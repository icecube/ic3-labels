#!/usr/bin/env python
# -*- coding: utf-8 -*
'''Helper functions for icecube specific labels.
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from ic3_labels.labels.utils import geometry


def get_significant_energy_loss(
        frame, pulse_key='InIceDSTPulses', mctree_name='I3MCTree'):
    """Get the most significant energy loss from the I3MCTree that may have
    caused the hits given by the pulses 'pulse_key'.

    Note: this is only approximative. The energy losses are weighted according
    to their energy, their distance and angle to the DOM,
    and the charge of the DOM.

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame. Must contain the I3MCTree, I3Geometry, and the
        specified pulses.
    pulse_key : str, optional
        The pulses to use.
    mctree_name : str, optional
        The name of the I3MCTree to use.

    Returns
    -------
    I3Particle
        An energy loss of type Cascade and length 0, that has the position,
        direction, energy, and time set.
    """
    # get pulses
    pulses = frame[pulse_key]
    if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
            isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
        pulses = pulses.apply(frame)

    # collect positions and charge of hit DOMs
    positions = []
    charges = []
    for om_key, dom_pulses in pulses:
        positions.append(frame['I3Geometry'].omgeo[om_key].position)
        charges.append(np.sum([p.charge for p in dom_pulses]))
    total_charge = np.sum(charges)

    def get_angle_factor(angle):
        if angle < np.deg2rad(45):
            return 1.
        else:
            return 1. - 0.9*((angle - np.deg2rad(45)) / np.deg2rad(135))

    def calc_factor(particle):
        factor = 0.

        for pos, charge in zip(positions, charges):
            diff = pos - particle.pos
            diff_p = dataclasses.I3Particle()
            diff_p.dir = dataclasses.I3Direction(diff.x, diff.y, diff.z)
            angle = I3Calculator.angle(diff_p, particle)

            distance = max(10, diff.magnitude)
            if distance < 500:
                factor += charge * get_angle_factor(angle) / (distance**3)
        return factor * particle.energy

    mctree = frame[mctree_name]

    def get_relevant_children(parent):

        allowed_types = [dataclasses.I3Particle.NuclInt,
                         dataclasses.I3Particle.PairProd,
                         dataclasses.I3Particle.Brems,
                         dataclasses.I3Particle.DeltaE,
                         dataclasses.I3Particle.EMinus,
                         dataclasses.I3Particle.EPlus,
                         dataclasses.I3Particle.Hadrons,
                         ]
        # Rekursion stop
        if parent.type in allowed_types and parent.pos.magnitude < 2000:
            return [parent]

        children = []
        daughters = mctree.get_daughters(parent)
        for daughter in daughters:
            children.extend(get_relevant_children(daughter))
        return children

    # Now walk through energy losses and calculate factor for each
    max_factor = -float('inf')
    cascade = None

    relevant_particles = []
    for primary in mctree.get_primaries():
        # go through all daughter particles
        for p in get_relevant_children(primary):
            factor = calc_factor(p)
            if factor > max_factor:
                cascade = p
                max_factor = factor

    # found our energy loss
    energy_loss = dataclasses.I3Particle()
    energy_loss.pos = dataclasses.I3Position(cascade.pos)
    energy_loss.dir = dataclasses.I3Direction(cascade.dir)
    energy_loss.time = cascade.time
    energy_loss.length = cascade.length
    energy_loss.energy = cascade.energy
    energy_loss.shape = dataclasses.I3Particle.Cascade
    return energy_loss


def get_num_coincident_events(frame, mctree_name='I3MCTree'):
    '''Get Number of coincident events (= number of primaries in I3MCTree).

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...
    mctree_name : str, optional
        The name of the I3MCTree to use.

    Returns
    -------
    int
    '''
    return len(frame[mctree_name].get_primaries())


def get_weighted_primary(frame, mctree_name=None):
    """Return weighted primary from I3MCTree

    Weighted CORSIKA simulation (as well as some NuGen simulation) can have
    coincidences mixed in that should not be used to calculate weights, as they
    were chosen at "natural" frequency. Find the primary that was chosen from a
    biased spectrum, and put it in the frame.

    Note: This code is adopted from old icecube.simweights project

    Parameters
    ----------
    frame : I3Frame
        The I3Frame from which to retrieve the weighted primary particle.
    mctree_name : str, optional
        The name of the I3MCTree to use.
        If None is provided, one of 'I3MCTree_preMuonProp', 'I3MCTree'
        will be used.

    Returns
    -------
    I3Particle
        The primary particle
    """

    if mctree_name is None:
        for mctree in ['I3MCTree_preMuonProp', 'I3MCTree']:
            if (mctree in frame) and (len(frame[mctree].primaries) != 0):
                mctree_name = mctree
                break

    primaries = frame[mctree_name].primaries

    if len(primaries) == 0:
        return None

    if len(primaries) == 1:
        idx = 0

    elif 'I3MCWeightDict' in frame:
        idx = [i for i in range(len(primaries)) if primaries[i].is_neutrino]
        assert len(idx) == 0, (idx, primaries)
        idx = idx[0]

    elif 'CorsikaWeightMap' in frame:
        wmap = frame['CorsikaWeightMap']

        if len(primaries) == 0:
            return None

        elif len(primaries) == 1:
            idx = 0

        elif 'PrimaryEnergy' in wmap:
            prim_e = wmap['PrimaryEnergy']
            idx = int(np.nanargmin([abs(p.energy-prim_e) for p in primaries]))

        elif 'PrimarySpectralIndex' in wmap:
            prim_e = wmap['Weight']**(-1./wmap['PrimarySpectralIndex'])
            idx = int(np.nanargmin([abs(p.energy-prim_e) for p in primaries]))

        else:
            idx = 0

    return primaries[idx]


def particle_is_inside(particle, convex_hull):
    '''Checks if a particle is inside the convex hull.

    The particle is considered inside if any part of its track is inside
    the convex hull. In the case of point like particles with length zero,
    the particle will be considered to be inside if the vertex is inside
    the convex hull.

    Parameters
    ----------
    particle : I3Particle
        The Particle to check.
    convex_hull : scipy.spatial.ConvexHull
        Defines the desired convex volume.

    Returns
    -------
    bool
        True if particle is inside, otherwise False.
    '''
    intersection_ts = geometry.get_intersections(
        convex_hull, particle.pos, particle.dir)

    # particle didn't hit convex_hull
    if intersection_ts.size == 0:
        return False

    # particle hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting event
        return True
    if max_ts < 0:
        # particle created after the convex hull
        return False
    if min_ts > particle.length + 1e-8:
        # particle stops before convex hull
        return False
    # everything else
    return True


def get_ids_of_particle_and_daughters(
        frame, particle, ids, mctree_name='I3MCTree'):
    '''Get particle ids of particle and all its daughters.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...

    particle : I3Particle
        Any particle type.

    ids : list,
        List in which to save all ids.
    mctree_name : str, optional
        The name of the I3MCTree to use.

    Returns
    -------
    ids : list
        List of all particle ids
    '''
    if particle is None:
        return ids
    ids.append(particle.id)
    daughters = frame[mctree_name].get_daughters(particle)
    for daughter in daughters:
        get_ids_of_particle_and_daughters(frame, daughter, ids)
    return ids


def get_all_parents(
        frame, particle, mctree_name='I3MCTree', reorder=True):
    '''Get all parents of the specified particle

    Parameters
    ----------
    frame : I3Frame
        The I3Frame to use.
    particle : I3Particle
        The particle for which all parents will be collected.
    mctree_name : str, optional
        The name of the I3MCTree to use.
    reorder : bool, optional
        If True, reorder list to go from primary -> particle.

    Returns
    -------
    list of I3Particle
        List of all parent particles
    '''

    parents = []
    while frame[mctree_name].has_parent(particle):
        parent = frame[mctree_name].parent(particle)
        parents.append(parent)
        particle = parent

    if reorder:
        parents = parents[::-1]
    return parents


def get_pulse_map(frame, particle,
                  pulse_map_string='InIcePulses',
                  mcpe_series_map_name='I3MCPESeriesMap',
                  max_time_dif=10):
    '''Get map of pulses induced by a specific particle.
       Pulses to be used can be specified through
       pulse_map_string.
        [This is only a guess on which reco Pulses
         could be originated from the particle.
         Naively calculated by looking at time diffs.]

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...
    particle : I3Particle
        Any particle type.
    pulse_map_string : str
        The name of the pulse series map in the frame that should
        be used.
    mcpe_series_map_name : str
        The key of the mcpe series map in the frame.
    max_time_dif : int, optional
        The maximal time difference for which to still match a
        reconstructed pulse to the underlying MC photon.

    Returns
    -------
    pulse_map : I3RecoPulseSeriesMap or I3MCPulseSeriesMap
        Map of pulses.
        ----- Better if done over I3RecoPulseSeriesMapMask ... ----

    Raises
    ------
    ValueError
        Description

    '''
    if particle.id.majorID == 0 and particle.id.minorID == 0:
        raise ValueError('Can not get pulse map for particle\
                            with id == (0,0)\n{}'.format(particle))

    particle_pulse_series_map = {}
    if pulse_map_string in frame:
        # make a list of all ids
        ids = get_ids_of_particle_and_daughters(frame, particle, [])
        # older versions of icecube dont have correct hash for I3ParticleID
        # Therefore need tuple of major and minor ID
        # [works directly with I3ParticleID in  Version combo.trunk r152630]
        ids = {(i.majorID, i.minorID) for i in ids}

        assert (0, 0) not in ids, \
            'Daughter particle with id (0,0) should not exist'

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # get candidate keys
        valid_keys = set(frame[mcpe_series_map_name].keys())

        # find all pulses resulting from particle or daughters of particle
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}
        for key in shared_keys:
            mc_pulse_times = [p.time for p in frame[mcpe_series_map_name][key]
                              if (p.ID.majorID, p.ID.minorID) in ids]
            particle_in_ice_pulses = []
            if mc_pulse_times:
                # speed things up:
                # pulses are sorted in time. Therefore we
                # can start from the last match
                last_index = 0
                for pulse in in_ice_pulses[key]:
                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, t in enumerate(mc_pulse_times[last_index:]):
                        if abs(pulse.time - t) < max_time_dif:
                            last_index = last_index + i
                            particle_in_ice_pulses.append(pulse)
                            break
            if particle_in_ice_pulses:
                particle_pulse_series_map[key] = particle_in_ice_pulses
    return type(in_ice_pulses)(particle_pulse_series_map)


def get_noise_pulse_map(frame,
                        pulse_map_string='InIcePulses',
                        mcpe_series_map_name='I3MCPESeriesMap',
                        empty_id=dataclasses.I3ParticleID(),
                        max_time_dif=10):
    '''Get map of pulses induced by noise.
        [This is only a guess on which reco Pulses
         could be originated from noise.]

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame to work on.
    pulse_map_string : str
        The name of the pulse series map in the frame that should
        be used.
    mcpe_series_map_name : str, optional
        Description
    empty_id : I3ParticleID, optional
        The particle ID used for noise pulses.
        Note, it is assumed here that an empty particle ID
        is used for noise in the simulation, but this is not
        verified.
    max_time_dif : int, optional
        The maximal time difference for which to still match a
        reconstructed pulse to the underlying MC photon.
    mcpe_series_map_name : str
        The key of the mcpe series map in the frame.

    Returns
    -------
    pulse_map : I3RecoPulseSeriesMap
        Map of pulses.
        ----- Better if done over I3RecoPulseSeriesMapMask ... ----
    '''

    noise_pulse_series_map = {}
    if pulse_map_string in frame:

        # get candidate keys
        valid_keys = set(frame[mcpe_series_map_name].keys())

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # find all pulses resulting from noise
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}
        for key in shared_keys:
            mc_pulse_times = [p.time for p in frame[mcpe_series_map_name][key]
                              if p.ID == empty_id]
            noise_in_ice_pulses = []
            if mc_pulse_times:
                # speed things up:
                # pulses are sorted in time. Therefore we
                # can start from the last match
                last_index = 0
                for pulse in in_ice_pulses[key]:
                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, t in enumerate(mc_pulse_times[last_index:]):
                        if abs(pulse.time - t) < max_time_dif:
                            last_index = last_index + i
                            noise_in_ice_pulses.append(pulse)
                            break
            if noise_in_ice_pulses:
                noise_pulse_series_map[key] = noise_in_ice_pulses
    return type(in_ice_pulses)(noise_pulse_series_map)


def get_pulse_primary_mapping(
        frame, primaries,
        pulse_map_string='InIcePulses',
        mcpe_series_map_name='I3MCPESeriesMap',
        max_time_dif=10):
    '''Get mapping of pulses to primary particles

       Pulses to be used can be specified through
       pulse_map_string.
        [This is only a guess on which reco Pulses
         could be originated from the particle.
         Naively calculated by looking at time diffs.]

    Parameters
    ----------
    frame : I3Frame
        The I3Frame to work on.
    primaries : list of I3Particle
        The particles for which the mapping will be generated.
        Mapping will have the following format:
            -1: pulses without a matching MC photon.
            0: pulses with matching MC photon, but not associated to any of
                the defined particles in the `primaries` parameter.
            1: pulses associated to the first particle in `primaries`.
            ...
            n: pulses associated to the n-th particle in `primaries`.
    pulse_map_string : str
        The pulse series (or MCPESeries) to generate the mapping for.
    mcpe_series_map_name : str, optional
        The name of the MCPESeriesMap, which will be used to generate
        the mapping.
    max_time_dif : int, optional
        The maximal time difference for which to still match a
        reconstructed pulse to the underlying MC photon.
        Note: the first MC photon in the corresponding time window
        will be selected to perform the mapping. Ideally, the `max_time_dif`
        is chosen as small as possible to assure a proper matching.

    Returns
    -------
    I3MapKeyVectorInt
        An I3MapKeyVectorInt object with a vector of int for each hit DOM.
        The ordering of the values in the vector corresponds to the same
        ordering of the pulse series map `pulse_map_string`.
        The integer values utilized in the mapping are defined as:
            -1: pulses without a matching MC photon.
            0: pulses with matching MC photon, but not associated to any of
                the defined particles in the `primaries` parameter.
            1: pulses associated to the first particle in `primaries`.
            ...
            n: pulses associated to the n-th particle in `primaries`.
    '''
    mapping = dataclasses.I3MapKeyVectorInt()
    if pulse_map_string in frame:

        # collect the particle IDs of all daughters from the defined
        # primary particles to which we will perform the pulse mapping
        mapping_ids = {}
        for i, particle in enumerate(primaries):

            # make a list of all ids
            ids = get_ids_of_particle_and_daughters(frame, particle, [])
            # older versions of icecube dont have correct hash for I3ParticleID
            # Therefore need tuple of major and minor ID
            # [works directly with I3ParticleID  > combo.trunk r152630]
            ids = {(i.majorID, i.minorID) for i in ids}

            assert (0, 0) not in ids, \
                'Daughter particle with id (0,0) should not exist'

            assert (0, -1) not in ids, \
                'Daughter particle with id (0,-1) should not exist'

            mapping_ids[i + 1] = ids

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # now walk through all pulses
        for key, pulses in in_ice_pulses.items():
            mapping[key] = []

            # get MC pulses for this OMKey
            mc_pulses = frame[mcpe_series_map_name][key]
            mc_pulse_times = [p.time for p in mc_pulses]

            # speed things up:
            # pulses are sorted in time. Therefore we
            # can start from the last match
            last_index = 0

            # go through the pulses and create mapping
            for pulse in pulses:

                # find the corresponding MC pulse (if it exists)
                mc_pulse = None
                if mc_pulse_times:

                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, t in enumerate(mc_pulse_times[last_index:]):
                        if abs(pulse.time - t) < max_time_dif:
                            last_index = last_index + i
                            mc_pulse = mc_pulses[last_index]
                            break

                if mc_pulse is None:

                    # we did not find a matching photon
                    mapping[key].append(-1)

                else:

                    # get id of the MC photon
                    mc_id = (mc_pulse.ID.majorID, mc_pulse.ID.minorID)

                    # check if it belongs to any of the specified primaries
                    found_mapping = False
                    for value, ids in mapping_ids.items():
                        if mc_id in ids:
                            mapping[key].append(value)
                            found_mapping = True

                    if not found_mapping:
                        # the corresponding MC photon is not one of the
                        # specified primaries
                        mapping[key].append(0)

            # make sure that we have only mapped existing pulses
            assert len(mapping[key]) == len(pulses), (mapping[key], pulses)

        return mapping
