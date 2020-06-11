import numpy as np

from icecube import dataclasses

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils.cascade import get_cascade_em_equivalent


def get_track_energy_depositions(mc_tree, track, num_to_remove,
                                 correct_for_em_loss=True,
                                 energy_threshold=1.,
                                 extend_boundary=None):
    """Get a list of track energy updates and a number of highest energy
    cascades that were removed from the track.

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

    Raises
    ------
    NotImplementedError
        Description

    Returns
    -------
    array_like
        The distances for the energy updates wrt the muon vertex.
    array_like
        The energies for at the energy update positions.
    list of I3Particle
        List of removed cascades. This list is sorted from highest to lowest
        energies.
        Note this list may be smaller than `num_to_remove` if the number of
        energy losses of the muon are smaller than this number.
    List of I3Particle
        List of track updates.
    """

    # sanity check
    assert num_to_remove >= 0

    # Other tracks such at taus might require additional handling of edge
    # cases. Remove for now
    if track.type not in [dataclasses.I3Particle.MuMinus,
                          dataclasses.I3Particle.MuPlus]:
        raise NotImplementedError(
            'Particle type {} not yet supported'.format(track.type))

    # get all daughters of track
    daughters = mc_tree.get_daughters(track)

    # gather all track updates
    # (these define rest track energy at a certain point)
    update_distances = []
    update_energies = []
    track_updates = []
    stoch_daughters = []
    stoch_energies = []
    last_update_outside = None
    track_entered_volume = False
    for daughter in daughters:

        # check if these points are inside defined volumen
        if extend_boundary is not None:

            # use IceCube boundary + extent_boundary [meters] to check
            if not geometry.is_in_detector_bounds(
                    daughter.pos, extend_boundary=extend_boundary):
                if daughter.type == track.type:
                    if not track_entered_volume:
                        last_update_outside = daughter
                continue

        track_entered_volume = True

        if daughter.type == track.type:

            # this is probably a track segment updated
            update_distances.append((daughter.pos - track.pos).magnitude)
            update_energies.append(daughter.energy)
            track_updates.append(daughter)
        else:
            stoch_daughters.append(daughter)
            stoch_energies.append(daughter.energy)

    update_distances = np.array(update_distances)
    update_energies = np.array(update_energies)
    assert (np.diff(update_distances) > 0).all()

    # find the n highest energy depositions and remove these
    indices = np.argsort(stoch_energies)
    sorted_stoch_daughters = [stoch_daughters[i] for i in indices]
    num_removed = min(num_to_remove, len([d for d in stoch_daughters if
                                          d.energy > energy_threshold]))

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

    if len(update_distances) > 0:
        # values for sanity check
        previous_energy = float(update_energies[-1])

        # fix the track updates by adding back the energy from the
        # removed cascades
        for cascade in cascades:

            # distance to cascade energy loss
            distance = (cascade.pos - track.pos).magnitude

            # find the track update at the point of the stochastic loss
            index = get_update_index(
                update_distances, update_energies, distance, cascade)
            assert index is not None
            assert np.allclose(update_distances[index], distance, atol=0.01)

            # update all of the remaining track updates
            # (add energy back since we assume this did not get depsosited)
            update_energies[index:] += cascade.energy

            # keep track of returned energy
            returned_energy += cascade.energy

        # sanity checks
        assert np.allclose(
            update_energies[-1] - returned_energy, previous_energy)
        assert (np.diff(update_energies) < 0).all()

    # Now walk through the leftover stochastic energy losses and make sure
    # that they are all covered by the track updates, possibly correct
    # for EM equivalent light yield if `correct_for_em_loss` is set to True.
    unaccounted_daughters = []
    for daughter in cascades_left:

        # distance to stochastic energy loss
        distance = (daughter.pos - track.pos).magnitude

        # find the track update at the point of the stochastic loss
        index = get_update_index(
            update_distances, update_energies, distance, daughter)
        if (index is not None and
                np.allclose(update_distances[index], distance, atol=0.01)):

            # sanity check to see if energy loss is included
            if index == 0:
                previous_energy = last_update_outside.energy
            else:
                previous_energy = update_energies[index - 1]
            delta_energy = previous_energy - update_energies[index]
            assert delta_energy >= daughter.energy

            if correct_for_em_loss:
                em_energy = get_cascade_em_equivalent(daughter)
                delta_energy = daughter.energy - em_energy

                assert delta_energy > -1e-7
                delta_energy = np.clip(delta_energy, 0., np.inf)

                # need to update additional delta_energy form
                # update all of the remaining track updates
                # (add energy back since we assume this did not get depsosited)
                update_energies[index:] += delta_energy
                # keep track of returned energy
                returned_energy += delta_energy

        else:
            # This seems to be an unaccounted stochastic energy loss
            # These should only be at end of track when muon decays
            index = np.searchsorted(update_distances, distance)
            assert len(update_distances) == index
            unaccounted_daughters.append(daughter)

    # If there are unnaccounted stochastic energy losses, make sure these
    # are the particle decay
    if len(unaccounted_daughters) > 0:
        assert len(unaccounted_daughters) == 3
        assert unaccounted_daughters[0].pos == unaccounted_daughters[1].pos
        assert unaccounted_daughters[0].pos == unaccounted_daughters[2].pos

        # add an update distance with the rest of the deposited energy
        energy_dep = update_energies[-1] - returned_energy

        # subtract off energy carried away by neutrinos or not visible
        for daughter in unaccounted_daughters:
            if daughter.type in [
                    dataclasses.I3Particle.NuE,
                    dataclasses.I3Particle.NuMu,
                    dataclasses.I3Particle.NuTau,
                    dataclasses.I3Particle.NuEBar,
                    dataclasses.I3Particle.NuMuBar,
                    dataclasses.I3Particle.NuTauBar,
                    ]:
                energy_dep -= daughter.energy

            elif correct_for_em_loss:
                em_energy = get_cascade_em_equivalent(daughter)
                delta_energy = daughter.energy - em_energy
                energy_dep -= delta_energy

        assert energy_dep <= update_energies[-1]

        update_distances = np.append(
            update_distances,
            (track.pos - unaccounted_daughters[0].pos).magnitude)
        update_energies = np.append(
            update_energies, update_energies[-1] - energy_dep)

    assert (np.diff(update_energies) < 0).all()
    assert (np.all(update_energies) >= 0)
    assert (np.diff([c.energy for c in cascades]) < 0).all()

    return update_distances, update_energies, cascades, track_updates


def get_update_index(update_distances, update_energies, distance, cascade,
                     atol=0.01):
    """Find the track update index at the given distance

    Parameters
    ----------
    update_distances : array_like
        The distances of the track updates.
    update_energies : array_like
        The energies of the track updates.
    distance : float
        The distance at which to find the index.
    cascade : I3Particle
        The cascade for which to find the equivalent track update.
    atol : float, optional
        The maximum allowed difference in distance in order to accetp a match.

    Returns
    -------
    int or None
        The index of the track update if there is a corresponding update.
    """
    if len(update_distances) == 0:
        return None

    # find the track update at the point of the stochastic loss
    index = np.searchsorted(update_distances, distance)

    # due to numerical issues, this can pick up the wrong index
    # Check if neighbouring indices match better
    delta_distances = []
    indices = []
    for idx in [index - 2, index - 1, index, index + 1, index + 2]:
        if idx >= 0 and idx < len(update_distances):
            delta_distance = np.abs(update_distances[idx] - distance)

            if idx > 0:
                delta_energy = update_energies[idx - 1] - update_energies[idx]
                passed_energy_check = delta_energy >= cascade.energy
            else:
                passed_energy_check = True
            if delta_distance < 1 and passed_energy_check:
                delta_distances.append(delta_distance)
                indices.append(idx)

    # choose best index if suitable matches were found
    if len(indices) > 0:
        index = indices[np.argmin(delta_distances)]

    # check if after last update distance
    if index == len(update_distances):
        if np.allclose(update_distances[index-1], distance, atol=atol):
            return index - 1
        else:
            return None

    # check if the found index matches the distance
    if np.allclose(update_distances[index], distance, atol=atol):
        return index
    elif index > 0 and np.allclose(
            update_distances[index-1], distance, atol=atol):
        return index - 1
    elif (index < len(update_distances) and
          np.allclose(update_distances[index+1], distance, atol=atol)):
        return index + 1

    return None


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
        rel_distances, cdf_energies)

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

    x_last = 0.
    y_last = 0.
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
    area_triangle = (x - x_last) * (y - y_last) / 2.

    return area_square + area_triangle
