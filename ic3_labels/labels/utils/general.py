#!/usr/bin/env python
# -*- coding: utf-8 -*
'''Helper functions for icecube specific labels.
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from ic3_labels.labels.utils import geometry


def get_ids_of_particle_and_daughters(frame, particle, ids):
    '''Get particle ids of particle and all its daughters.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...

    particle : I3Particle
        Any particle type.

    ids : list,
        List in which to save all ids.

    Returns
    -------
    ids: list
        List of all particle ids
    '''
    if particle is None:
        return ids
    ids.append(particle.id)
    daughters = frame['I3MCTree'].get_daughters(particle)
    for daughter in daughters:
        get_ids_of_particle_and_daughters(frame, daughter, ids)
    return ids


def get_pulse_map(frame, particle,
                  pulse_map_string='InIcePulses',
                  max_time_dif=100):
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

    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for

    Returns
    -------
    pulse_map : I3RecoPulseSeriesMap or I3MCPulseSeriesMap
        Map of pulses.

    ----- Better if done over I3RecoPulseSeriesMapMask ----

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
        valid_keys = set(frame['I3MCPESeriesMap'].keys())

        # find all pulses resulting from particle or daughters of particle
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}
        for key in shared_keys:
            mc_pulse_times = [p.time for p in frame['I3MCPESeriesMap'][key]
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
    return dataclasses.I3RecoPulseSeriesMap(particle_pulse_series_map)


def get_noise_pulse_map(frame,
                        pulse_map_string='InIcePulses',
                        max_time_dif=100):
    '''Get map of pulses induced by noise.
        [This is only a guess on which reco Pulses
         could be originated from noise.]

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...

    pulse_map_string : key of pulse map in frame,
        of which the mask should be computed for

    Returns
    -------
    pulse_map : I3RecoPulseSeriesMap
        Map of pulses.

    ----- Better if done over I3RecoPulseSeriesMapMask ----

    '''

    noise_pulse_series_map = {}
    if pulse_map_string in frame:
        # pulses with no particle ID are likely from noise
        empty_id = dataclasses.I3ParticleID()

        # get candidate keys
        valid_keys = set(frame['I3MCPESeriesMap'].keys())

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # find all pulses resulting from noise
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}
        for key in shared_keys:
            mc_pulse_times = [p.time for p in frame['I3MCPESeriesMap'][key]
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
    return dataclasses.I3RecoPulseSeriesMap(noise_pulse_series_map)
