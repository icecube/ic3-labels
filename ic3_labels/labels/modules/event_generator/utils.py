import numpy as np

from icecube import dataclasses

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils.cascade import convert_to_em_equivalent


def get_muon_from_frame(frame, primary):
    """Get muon from frame

    Parameters
    ----------
    frame : I3Frame
        The current frame.
    primary : I3Particle
        The primary particle.

    Returns
    -------
    I3Particle
        The muon from the frame

    Raises
    ------
    ValueError
        If not muon is found.
    """

    # NuGen Dataset
    if primary.is_neutrino:
        muon = mu_utils.get_muon_of_inice_neutrino(frame)

        if muon is None:
            print(frame["I3MCTree"])
            raise ValueError("Did not find a muon!")

    # MuonGun Dataset
    elif (
        primary.type_string == "unknown" and primary.pdg_encoding == 0
    ) or mu_utils.is_muon(primary):

        if mu_utils.is_muon(primary):
            muon = primary

            # -----------------------------
            # muon primary: MuonGun dataset
            # -----------------------------
            daugters = frame["I3MCTree"].get_daughters(muon)
            if len(daugters) == 1:
                daughter = daugters[0]
                if mu_utils.is_muon(daughter) and (
                    (daughter.id == primary.id)
                    and (daughter.dir == primary.dir)
                    and (daughter.pos == primary.pos)
                    and (daughter.energy == primary.energy)
                ):
                    muon = daughter

        else:
            daughters = frame["I3MCTree"].get_daughters(primary)
            muon = daughters[0]

            # Perform some safety checks to make sure that this is MuonGun
            assert (
                len(daughters) == 1
            ), "Expected 1 daughter for MuonGun, but got {!r}".format(
                daughters
            )
            assert mu_utils.is_muon(muon), "Expected muon but got {!r}".format(
                muon
            )

    return muon


def get_track_energy_depositions(
    mc_tree,
    track,
    num_to_remove,
    correct_for_em_loss=True,
    energy_threshold=1.0,
    extend_boundary=None,
    atol_time=1e-2,
    atol_pos=0.5,
    fix_muon_pair_production_bug=False,
):
    """Get a list of track energy updates and a number of highest energy
    cascades that were removed from the track.

    Note: this function has a lot of additional code and asserts to verify
    that the assumptions made hold. The I3MCTree is not well specified and
    may change between software revisions. In this case, the asserts will help
    in letting this crash loudly.
    The main driving assumption is that the corresponding track update particle
    has a minor particle ID +1 from the stochastic loss. This is checked via
    asserts on the delta time and position.

    Parameters
    ----------
    mc_tree : I3MCTree
        The I3MCTree.
    track : I3Particle.
        The track particle (usually a muon or tau) for which to create
        the energy loss plots
    num_to_remove : int
        Number of energy losses to remove. The n highest energy depositions
        will be removed from the track energy losses and instead be handled
        as separate cascades.
    correct_for_em_loss : bool, optional
        If True, energy depositions will be in terms of EM equivalent deposited
        energy.
        If False, the actual (but possibly invisible) energy depositions is
        used..
    energy_threshold : float, optional
        The energy threshold under which an energy loss is considered to be
        removed from the track.
    extend_boundary : float, optional
        If provided only energy losses within convex hull + extend boundary
        are accepted and considered.
    atol_time : float, optional
        Tolerance for absolute delta of interaction and track segment times.
        These should technically be identical, but precision loss can result
        in deviations.
    atol_pos : float, optional
        Tolerance for absolute delta of interaction and track segment vertex
        positions, provided in meters. These should technically be identical,
        but precision loss can result in deviations.
    fix_muon_pair_production_bug : bool, optional
        Older IceTray versions incorrectly double-counted energy losses of
        muon-pair production. This results in bare muon track segments in the
        I3MCTree to have less energy than they should.
        If set to True, the collected energies of the track segments are
        updated to account for these muon-pair productions. Note: this only
        works if the muon track segments don't reach down to zero energy, as
        the energy is capped at this value and does not turn negative.

    Raises
    ------
    NotImplementedError
        Description

    Returns
    -------
    dict
        update_distances : array_like
            The distances for the energy updates wrt the muon vertex.
        update_energies : array_like
            The energies for at the energy update positions.
        pair_production_muons : list of I3Particle
            List of muons created in muon-pair production losses.
        cascades : list of I3Particle
            List of removed cascades. This list is sorted from highest to lowest
            energies.
            Note this list may be smaller than `num_to_remove` if the number of
            energy losses of the muon are smaller than this number.
        track_updates : List of I3Particle
            List of track updates.
        relative_energy_losses : array_like
            The relative energy loss of each cascade energy deposition with
            respect to the total track energy at that point.
            Same length as `cascades`.
    """

    # sanity check
    assert num_to_remove >= 0

    # Other tracks such at taus might require additional handling of edge
    # cases. Remove for now
    if track.type not in [
        dataclasses.I3Particle.MuMinus,
        dataclasses.I3Particle.MuPlus,
    ]:
        raise NotImplementedError(
            "Particle type {} not yet supported".format(track.type)
        )

    # get all daughters of track
    daughters = mc_tree.get_daughters(track)

    # sort daughters according to minor PID
    sorted_idx = np.argsort([d.id.minorID for d in daughters])
    daughters = [daughters[i] for i in sorted_idx]

    n_daughters = len(daughters)

    # gather all track updates
    # (these define rest track energy at a certain point)
    update_distances = []
    update_times = []
    update_energies = []
    update_ids = []
    track_updates = []
    stoch_daughters = []
    stoch_energies = []
    last_update_outside = None
    track_entered_volume = False
    pair_production_muons = []

    # these variables are required for fix of muon-pair production bug fix
    muon_pair_prod_energy = 0
    has_hit_floor = False
    last_positive_track_update = None

    for index, daughter in enumerate(daughters):

        # check if the current daughter is part of a muon-pair production.
        # We'll do this check by checking if a second muon exists in the
        # I3MCTree with the same vertex and by checking that it's a
        # muon and anti-muon
        is_pair_production = False
        if daughter.pdg_encoding in (-13, 13):

            # collect neighbouring muons in I3MCTree,
            # stop when something else found
            check_indices = []
            for i in range(index, index - 3, -1):
                if i >= 0 and daughters[i].pdg_encoding in (-13, 13):
                    check_indices.append(i)
                else:
                    break
            for i in range(index + 1, index + 3):
                if i < n_daughters and daughters[i].pdg_encoding in (-13, 13):
                    check_indices.append(i)
                else:
                    break

            # collect all potential pair-production pairs by selecting
            # muons at the same vertex time
            mu_pairs = []
            for i in check_indices:
                if np.abs(daughters[i].time - daughter.time) < atol_time:
                    mu_pairs.append(daughters[i])

            # this cannot be a pair production
            if len(mu_pairs) == 1:
                pass

            # two muons should only be found by random chance of two losses
            # overlapping. In this case the two muons should have the
            # same pdg_encoding. A proper muon and anti-muon pair should
            # only happen outside of the active volume.
            elif len(mu_pairs) == 2:
                if mu_pairs[0].pdg_encoding == -mu_pairs[1].pdg_encoding:

                    # make sure this case only happens outside
                    assert (
                        daughter.pos.rho >= 800
                        or np.abs(daughter.pos.z) >= 800
                    ), daughter

                    is_pair_production = True
                else:
                    is_pair_production = False

            # this should only happen if there is a muon pair production
            elif len(mu_pairs) == 3:
                # sort based on minor ID
                # Assumption: muon track segment is included after
                # muons generated in pair-production!!
                sorted_idx = np.argsort([p.id.minorID for p in mu_pairs])
                mu_pairs = [mu_pairs[i] for i in sorted_idx]

                assert mu_pairs[0].pdg_encoding == -mu_pairs[1].pdg_encoding

                if daughter == mu_pairs[0] or daughter == mu_pairs[1]:
                    is_pair_production = True
            else:
                raise ValueError("Unexpected number of muons: ", mu_pairs)

            # if we want to fix for the potential muon-pair-production but
            # we need to account for when the track hits the floor of 0 GeV
            if fix_muon_pair_production_bug and not is_pair_production:

                # once an energy of 0 is reached, a simple addition of
                # double-counted energy loss will not fix things
                # We will fudge the energy of the subsequent track updates
                # to have a delta energy of at least the stochastic energy
                # losses in between.
                if daughter.energy <= 1e-5:
                    # find corresponding stochastic loss by checking
                    # if times of vertices match in neighboring daughters
                    delta_energy = 0.0
                    if index > 0 and (
                        np.abs(daughters[index - 1].time - daughter.time)
                        < atol_time
                    ):
                        delta_energy = daughters[index - 1].energy
                    elif index < n_daughters - 1 and (
                        np.abs(daughters[index + 1].time - daughter.time)
                        < atol_time
                    ):
                        delta_energy = daughters[index + 1].energy

                    # correct for remaining energy of track before it hit
                    # the floor of 0 GeV
                    # This is only possible if we have one update before that
                    if not has_hit_floor:
                        has_hit_floor = True

                        if last_positive_track_update is not None:
                            assert (
                                last_positive_track_update.pdg_encoding
                                == daughter.pdg_encoding
                            )
                            assert last_positive_track_update.energy > 0
                            delta_energy -= last_positive_track_update.energy

                    muon_pair_prod_energy -= delta_energy + 1e-4

                # keep track of the last track update with positive energy
                if daughter.energy > 0:
                    last_positive_track_update = daughter

        # if desired, fix potential muon-pair-productino bug
        if is_pair_production and fix_muon_pair_production_bug:
            # keep track of double-counted energy losses due to
            # faulty tracking of muon pair production.
            # Note that the double-counting of muon-pair production
            # losses only occurs if these happen outside of the active
            # volume. This is usually set as a cylinder with 800m
            # radius + 1600m height
            if daughter.pos.rho >= 800 or np.abs(daughter.pos.z) >= 800:
                muon_pair_prod_energy += daughter.energy

                # this bug should only happen when outside and only when
                # the two muons are included, but not a track update
                assert len(mu_pairs) == 2, mu_pairs
            else:
                # correctly handled pair production should have 3
                # consecutive muons in I3MCTree
                assert len(mu_pairs) == 3, mu_pairs

        # check if these points are inside defined volume
        if extend_boundary is not None:

            # due to slight deviations in particle positions of the
            # corresponding track updates for each stochastic loss it
            # can happen that the track update is just outside the
            # defined volume while the stochastic loss is just inside.
            # We want to avoid this and make sure that the track update
            # is always inside (it does not hurt much if only the
            # stochastic loss falls outside)
            if daughter.type == track.type:
                eps_boundary = 1.0
            else:
                eps_boundary = 0.0

            # use IceCube boundary + extent_boundary [meters] to check
            if not geometry.is_in_detector_bounds(
                daughter.pos, extend_boundary=extend_boundary + eps_boundary
            ):
                if daughter.type == track.type:
                    if not track_entered_volume:
                        last_update_outside = daughter

                        if fix_muon_pair_production_bug:
                            last_update_outside = dataclasses.I3Particle(
                                last_update_outside
                            )
                            last_update_outside.energy += muon_pair_prod_energy
                continue

        track_entered_volume = True

        if daughter.type == track.type and not is_pair_production:

            # this is probably a track segment updated
            update_distances.append((daughter.pos - track.pos).magnitude)
            update_times.append(daughter.time)
            update_ids.append(daughter.id.minorID)
            track_updates.append(daughter)

            if fix_muon_pair_production_bug:
                update_energies.append(daughter.energy + muon_pair_prod_energy)
            else:
                update_energies.append(daughter.energy)

        elif not is_pair_production:
            stoch_daughters.append(daughter)
            stoch_energies.append(daughter.energy)

        # muon pair-production
        else:
            # we do not consider these as stochastic losses in the sense
            # that the entire energy is deposited in a short-ranged shower.
            pair_production_muons.append(daughter)

    update_distances = np.array(update_distances)
    update_energies = np.array(update_energies)
    update_times = np.array(update_times)
    update_ids = np.array(update_ids)

    # check that everything is sorted
    assert (np.diff(update_distances) >= 0).all()
    assert (np.diff(update_times) >= 0).all()
    assert (np.diff(update_ids) > 0).all()

    # find the n highest energy depositions and remove these
    indices = np.argsort(stoch_energies)
    sorted_stoch_daughters = [stoch_daughters[i] for i in indices]
    num_removed = min(
        num_to_remove,
        len([d for d in stoch_daughters if d.energy > energy_threshold]),
    )

    if num_removed == 0:
        cascades = []
        cascades_left = sorted_stoch_daughters
    elif num_removed == len(sorted_stoch_daughters):
        cascades = sorted_stoch_daughters[::-1]
        cascades_left = []
    else:
        cascades_left = sorted_stoch_daughters[:-num_removed]
        cascades = sorted_stoch_daughters[-num_removed:][::-1]
    assert len(cascades) == num_removed

    # keep track of returned energy
    returned_energy = 0

    # keep track of unaccounted daughters, e.g. energy losses that do not have
    # a matching track update. This should only happen for the decay point
    unaccounted_daughters = []

    # compute relative energy loss (momentum transfer q) of each cascade
    relative_energy_losses = []

    if len(update_distances) > 0:
        # values for sanity check
        previous_energy = float(update_energies[-1])

        # fix the track updates by adding back the energy from the
        # removed cascades
        for cascade in cascades:

            # find the track update at the point of the stochastic loss
            index = get_update_index(
                update_times,
                update_energies,
                update_ids,
                cascade,
                atol=atol_time,
            )

            # the index should only be None if this cascade is part of the
            # decay products, e.g. at the end of the track
            if index is None and np.allclose(
                cascade.time, daughters[-1].time, atol=atol_time
            ):
                unaccounted_daughters.append((cascade, True))

                # we would need to consider the continuous losses to estimate
                # the relative energy loss. Instead of doing this, we'll just
                # add a NaN for now.
                relative_energy_losses.append(float("nan"))
            else:
                assert index is not None
                assert np.allclose(
                    update_times[index], cascade.time, atol=atol_time
                )
                assert np.allclose(
                    update_distances[index],
                    (cascade.pos - track.pos).magnitude,
                    atol=atol_pos,
                )

                # the energy of the muon update is already reduced by the loss.
                # To obtain the muon energy prior to the loss, we need to add
                # it back
                relative_energy_losses.append(
                    cascade.energy
                    / (cascade.energy + track_updates[index].energy)
                )

                # update all of the remaining track updates
                # (add energy back since we assume this did not get depsosited)
                update_energies[index:] += cascade.energy

                # keep track of returned energy
                returned_energy += cascade.energy

        # sanity checks
        assert np.allclose(
            update_energies[-1] - returned_energy, previous_energy
        )
        assert (np.diff(update_energies) <= 1e-4).all(), update_energies

    else:

        # No track updates exist. We would need to consider the continuous
        # losses to estimate the relative energy loss. Instead of doing this,
        # we'll just add NaNs for now.
        for cascade in cascades:
            relative_energy_losses.append(float("nan"))

    relative_energy_losses = np.array(relative_energy_losses)
    assert len(relative_energy_losses) == len(cascades)

    # Now walk through the leftover stochastic energy losses and make sure
    # that they are all covered by the track updates, possibly correct
    # for EM equivalent light yield if `correct_for_em_loss` is set to True.
    for daughter in cascades_left:

        # distance to stochastic energy loss
        distance = (daughter.pos - track.pos).magnitude

        # find the track update at the point of the stochastic loss
        index = get_update_index(
            update_times,
            update_energies,
            update_ids,
            daughter,
            atol=atol_time,
        )
        if index is not None:

            # perform some sanity checks
            assert np.allclose(
                update_times[index], daughter.time, atol=atol_time
            )
            assert np.allclose(
                update_distances[index],
                (daughter.pos - track.pos).magnitude,
                atol=atol_pos,
            )

            # sanity check to see if energy loss is included
            if index == 0:
                if last_update_outside is None:
                    # Sometimes there are no muons inserted previous to
                    # the first stochastic energy loss.
                    # use the track energy in this case
                    previous_energy = track.energy
                else:
                    previous_energy = last_update_outside.energy
            else:
                previous_energy = update_energies[index - 1]
            delta_energy = previous_energy - update_energies[index]
            assert delta_energy >= daughter.energy - 1e-3

            if correct_for_em_loss:
                em_energy = convert_to_em_equivalent(daughter)
                delta_energy = daughter.energy - em_energy

                assert delta_energy > -1e-5, delta_energy
                delta_energy = np.clip(delta_energy, 0.0, np.inf)

                # need to update additional delta_energy form
                # update all of the remaining track updates
                # (add energy back since we assume this did not get depsosited)
                update_energies[index:] += delta_energy
                # keep track of returned energy
                returned_energy += delta_energy

        else:
            # This seems to be an unaccounted stochastic energy loss
            # These should only be at end of track when muon decays
            # or in some unlucky cases in which the track update happens
            # to get cut away, while the stochastic energy is still inside.
            # However, we account for the latter case by increasing the
            # convex hull when checking for contained track updates.
            assert np.allclose(
                daughter.time, daughters[-1].time, atol=atol_time
            )
            unaccounted_daughters.append((daughter, False))

    # If there are unnaccounted stochastic energy losses, make sure these
    # are the particle decay
    if len(unaccounted_daughters) > 0:
        assert len(unaccounted_daughters) == 3
        assert (
            unaccounted_daughters[0][0].pos == unaccounted_daughters[1][0].pos
        )
        assert (
            unaccounted_daughters[0][0].pos == unaccounted_daughters[2][0].pos
        )

        # add an update distance with the rest of the deposited energy
        if len(update_energies) == 0:

            # this should only be the case if the only energy losses in the
            # I3MCTree are the ones from the decay
            assert len(stoch_daughters) == 3
            previous_energy = track.energy
        else:
            previous_energy = update_energies[-1]

        energy_dep = previous_energy - returned_energy

        # subtract off energy carried away by neutrinos or not visible
        for daughter, is_accounted_for in unaccounted_daughters:
            if (
                daughter.type
                in [
                    dataclasses.I3Particle.NuE,
                    dataclasses.I3Particle.NuMu,
                    dataclasses.I3Particle.NuTau,
                    dataclasses.I3Particle.NuEBar,
                    dataclasses.I3Particle.NuMuBar,
                    dataclasses.I3Particle.NuTauBar,
                ]
                or is_accounted_for
            ):
                energy_dep -= daughter.energy

            elif correct_for_em_loss:
                em_energy = convert_to_em_equivalent(daughter)
                delta_energy = daughter.energy - em_energy
                energy_dep -= delta_energy

        assert energy_dep <= previous_energy

        update_distances = np.append(
            update_distances,
            (track.pos - unaccounted_daughters[0][0].pos).magnitude,
        )
        update_energies = np.append(
            update_energies, previous_energy - energy_dep
        )

    # If there is only one track update in the detector, prepend the last one
    # before the detector
    if len(update_distances) == 1:

        # add last existing track update if it exists
        if last_update_outside is not None:

            distance = (track.pos - last_update_outside.pos).magnitude
            energy = last_update_outside.energy

            update_distances = np.insert(update_distances, 0, distance)
            update_energies = np.insert(update_energies, 0, energy)
            track_updates = [last_update_outside] + track_updates

        # otherwise add the starting track position and energy
        else:
            update_distances = np.insert(update_distances, 0, 0.0)
            update_energies = np.insert(update_energies, 0, track.energy)
            track_updates = [track] + track_updates

    # energies should be monotonously decreasing except if updates are
    # extremely close to each other
    assert (
        np.diff(update_distances)[np.diff(update_energies) >= 0] < 1e-1
    ).all()

    # Fix monoticity of energy updates that might have gotten broken due
    # to numerical issues
    energy_corrections = np.diff(update_energies)
    mask = energy_corrections <= 0.0
    energy_corrections[mask] = 0.0
    assert (np.abs(energy_corrections) <= 1e-2).all()
    update_energies[1:] -= energy_corrections

    assert (np.diff(update_energies) <= 0).all()
    assert np.all(update_energies >= 0), update_energies
    assert (np.diff([c.energy for c in cascades]) < 0).all()

    return {
        "update_distances": update_distances,
        "update_energies": update_energies,
        "pair_production_muons": pair_production_muons,
        "cascades": cascades,
        "track_updates": track_updates,
        "relative_energy_losses": relative_energy_losses,
    }


def get_bundle_energy_depositions(
    mc_tree,
    tracks,
    primary,
    correct_for_em_loss=True,
    energy_threshold=1.0,
    extend_boundary=None,
    atol_time=1e-2,
    atol_pos=0.5,
    fix_muon_pair_production_bug=False,
):
    """Get a list of track energy updates for a list of tracks

    Combines track updates from the provided list of tracks.
    Note: no checks are performed to ensure that the tracks are all part of
    the same bundle! This has to be ensured by the user!

    Parameters
    ----------
    mc_tree : I3MCTree
        The I3MCTree.
    tracks : list of I3Particle
        The list of tracks for which to compute combined energy depositions.
        Note: these must be coming from tracks of the same bundle. No checks
        are made to ensure this. This must be ensured by the user!
    primary : I3Particle
        The primary particle. Energy deposition distances will be relative
        to the vertex of the primary particle.
    correct_for_em_loss : bool, optional
        If True, energy depositions will be in terms of EM equivalent deposited
        energy.
        If False, the actual (but possibly invisible) energy depositions is
        used..
    energy_threshold : float, optional
        The energy threshold under which an energy loss is considered to be
        removed from the track.
    extend_boundary : float, optional
        If provided only energy losses within convex hull + extend boundary
        are accepted and considered.
    atol_time : float, optional
        Tolerance for absolute delta of interaction and track segment times.
        These should technically be identical, but precision loss can result
        in deviations.
    atol_pos : float, optional
        Tolerance for absolute delta of interaction and track segment vertex
        positions, provided in meters. These should technically be identical,
        but precision loss can result in deviations.
    fix_muon_pair_production_bug : bool, optional
        Older IceTray versions incorrectly double-counted energy losses of
        muon-pair production. This results in bare muon track segments in the
        I3MCTree to have less energy than they should.
        If set to True, the collected energies of the track segments are
        updated to account for these muon-pair productions. Note: this only
        works if the muon track segments don't reach down to zero energy, as
        the energy is capped at this value and does not turn negative.

    Returns
    -------
    dict
        update_distances : array_like
            The distances for the energy updates wrt the primary vertex.
        update_energies : array_like
            The energies for at the energy update positions.
        update_delta_energies : array_like
            The energy loss at each corresponding distance.
            The first energy loss is zero by default.
            Same shape as `update_distances` and `update_energies`.
    """

    dep_distances = []
    dep_energies = []

    bundle_energy_start = 0
    bundle_dist_start = np.inf

    for track in tracks:
        e_dep_dict = get_track_energy_depositions(
            mc_tree=mc_tree,
            track=track,
            num_to_remove=0,
            correct_for_em_loss=correct_for_em_loss,
            energy_threshold=energy_threshold,
            extend_boundary=extend_boundary,
            atol_time=atol_time,
            atol_pos=atol_pos,
            fix_muon_pair_production_bug=fix_muon_pair_production_bug,
        )

        if len(e_dep_dict["update_distances"]) > 0:

            # correct distances to be relative to primary vertex
            dist_offset = (primary.pos - track.pos).magnitude
            update_distances = (
                np.array(e_dep_dict["update_distances"]) + dist_offset
            )

            # take first point among all tracks to be the initial bundle start
            if bundle_dist_start > update_distances[0]:
                bundle_dist_start = update_distances[0]

            # accumulate energy of all tracks to be the bundle energy
            bundle_energy_start += e_dep_dict["update_energies"][0]

            # compute energy losses
            energy_losses = np.diff(e_dep_dict["update_energies"])

            dep_distances.append(update_distances[1:])
            dep_energies.append(energy_losses)

    if len(dep_distances) > 0:

        dep_distances = np.concatenate(dep_distances, axis=0)
        dep_energies = np.concatenate(dep_energies, axis=0)

        # sort losses along distance
        # Note: this assumes that all tracks are traveling on same trajectory!
        sorted_idx = np.argsort(dep_distances)
        dep_distances = dep_distances[sorted_idx]
        dep_energies = dep_energies[sorted_idx]

        dep_energies_cum = bundle_energy_start + np.cumsum(dep_energies)

        # create energy deposition updates for bundle
        update_distances = np.insert(dep_distances, 0, bundle_dist_start)
        update_energies = np.insert(dep_energies_cum, 0, bundle_energy_start)
        update_delta_energies = np.insert(dep_energies, 0, 0.0)

        # some basic sanity checks
        assert np.all(update_energies >= 0), update_energies
        assert np.all(np.diff(update_energies) <= 0), np.diff(update_energies)
        assert np.all(update_delta_energies <= 0), update_delta_energies

    else:
        update_distances = np.atleast_1d([])
        update_energies = np.atleast_1d([])
        update_delta_energies = np.atleast_1d([])

    return {
        "update_distances": update_distances,
        "update_energies": update_energies,
        "update_delta_energies": update_delta_energies,
    }


def get_update_index(
    update_times, update_energies, update_ids, cascade, atol=1e-2
):
    """Find the track update index at the given distance

    Parameters
    ----------
    update_times : array_like
        The times of the track updates.
    update_energies : array_like
        The energies of the track updates.
    update_ids : array_like
        The minor particle ids of the track updates.
    cascade : I3Particle
        The cascade for which to find the equivalent track update.
    atol : float, optional
        The maximum allowed difference in time in order to accept a match.

    Returns
    -------
    int or None
        The index of the track update if there is a corresponding update.
    """
    if len(update_times) == 0:
        return None

    # find the track update at the point of the stochastic loss
    index = np.searchsorted(update_ids, cascade.id.minorID)

    # check if after last update time
    if index == len(update_times):
        return None

    # perform sanity checks
    if not np.allclose(update_times[index], cascade.time, atol=atol):
        raise ValueError(
            "Times do not match: {} != {}!".format(
                update_times[index], cascade.time
            )
        )

    if index > 0:
        delta_energy = update_energies[index - 1] - update_energies[index]
        if delta_energy < convert_to_em_equivalent(cascade) - 1e-3:
            msg = "Energy loss is larger than available energy: {} !> {}!"
            raise ValueError(msg.format(delta_energy, cascade.energy))

    return index


def compute_stochasticity(update_distances, update_energies):
    """Compute the stochasticity for a given set of energy losses.

    The stochasticity is defined here as the area between the diagonal of
    the relative distance-CDF-plot and the curve for the cumulative energy
    losses (CDF) of the track divided by the maximum area of 0.5.

    Parameters
    ----------
    update_distances : array_like
        The distances of the track updates.
    update_distances : array_like
        The energies of the track updates.

    Returns
    -------
    float
        The computed stochasticity.
    """
    delta_distances = np.diff(update_distances)
    delta_energies = np.diff(update_energies)
    cum_energies = np.cumsum(-delta_energies)
    cum_distances = np.cumsum(delta_distances)

    cdf_energies = cum_energies / cum_energies[-1]
    rel_distances = cum_distances / cum_distances[-1]

    patch_starts, patch_ends, patch_areas = compute_area_difference(
        rel_distances, cdf_energies
    )

    stochasticity = np.sum(np.abs(patch_areas)) / 0.5
    area_above = np.sum(patch_areas[patch_areas > 0])
    area_below = np.sum(patch_areas[patch_areas < 0])

    return stochasticity, area_above, area_below


def compute_area_difference(x_rel, y_rel):
    """Compute positive and negative area differences

    Parameters
    ----------
    x_rel : array_like
        The relative x values which range from [0, 1].
    y_rel : array_like
        The relative y values which range from [0, 1].

    Returns
    -------
    array_like
        The start position of the area patch.
    array_like
        The end position of the area patch.
    array_like
        The area difference patches.
    """
    x_rel_list = [x for x in x_rel] + [1]
    y_rel_list = [y for y in y_rel] + [1]

    # make sure input is ordered in x
    assert (np.diff(x_rel_list) >= 0).all()

    patch_areas = []
    patch_starts = []
    patch_ends = []

    x_last = 0.0
    y_last = 0.0
    for x, y in zip(x_rel_list, y_rel_list):

        # check if there is a crossing over the diagonal
        if (y_last - x_last) * (y - x) >= 0:

            # both points are either above or below the diagonal
            # (or one or both are exactly on the diagonal)
            area_patch = get_area_of_patch(x_last, y_last, x, y)
            area_diagonal = get_area_of_patch(x_last, x_last, x, x)

            # append to list
            patch_areas.append(area_patch - area_diagonal)
            patch_starts.append(x_last)
            patch_ends.append(x)

        else:
            # there is a cross-over point

            # compute point at which it crosses over
            m = (y - y_last) / (x - x_last)
            x_cross = (m * x_last - y_last) / (m - 1)
            y_cross = y_last + m * (x_cross - x_last)

            # patch before cross-over point
            area_patch = get_area_of_patch(x_last, y_last, x_cross, y_cross)
            area_diagonal = get_area_of_patch(x_last, x_last, x_cross, x_cross)

            # append to list
            patch_areas.append(area_patch - area_diagonal)
            patch_starts.append(x_last)
            patch_ends.append(x_cross)

            # patch after cross-over point
            area_patch = get_area_of_patch(x_cross, y_cross, x, y)
            area_diagonal = get_area_of_patch(x_cross, x_cross, x, x)

            # append to list
            patch_areas.append(area_patch - area_diagonal)
            patch_starts.append(x_cross)
            patch_ends.append(x)

        # move on to next patch
        x_last = x
        y_last = y

    patch_starts = np.array(patch_starts)
    patch_ends = np.array(patch_ends)
    patch_areas = np.array(patch_areas)

    return patch_starts, patch_ends, patch_areas


def get_area_of_patch(x_last, y_last, x, y):
    """Calculate area of a patch provided by (x_last, y_last) and (x, y).

    The area of the polygon (x_last, 0), (x_last, y_last), (x, y), (x, 0)
    is computed and returned.

    Parameters
    ----------
    x_last : float
        The first x-point.
    y_last : float
        The first y-point.
    x : float
        The second x-point.
    y : float
        The second y-point.

    Returns
    -------
    float
        The area of the defined patch.
    """

    # make sure input is ordered in x
    assert np.all(x >= x_last)

    # area of square (x_last, 0), (x_last, y_last), (x, y_last), (x, 0)
    area_square = (x - x_last) * y_last

    # area of triangle (x_last, y_last), (x, y_last), (x, y) [can be negative]
    area_triangle = (x - x_last) * (y - y_last) / 2.0

    return area_square + area_triangle
