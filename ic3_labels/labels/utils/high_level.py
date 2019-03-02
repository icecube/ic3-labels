'''Helper functions for icecube specific labels.
'''
from __future__ import print_function, division
#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils import general
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils.cascade import get_cascade_of_primary_nu
from ic3_labels.labels.utils.cascade import get_cascade_energy_deposited
from ic3_labels.labels.utils.neutrino import get_interaction_neutrino
from ic3_labels.labels.utils.muon import get_muon_energy_deposited


def get_energy_deposited(frame, convex_hull, particle):
    '''Function to get the total energy a particle deposited in the
    volume defined by the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    particle : I3Particle
        Particle.
        (Particle can be of any type: Muon, Cascade, Neutrino...)

    Returns
    -------
    energy : float
        Deposited Energy.
    '''

    raise NotImplementedError


def get_energy_deposited_including_daughters(frame,
                                             convex_hull,
                                             particle,
                                             muongun_primary_neutrino_id=None,
                                             ):
    '''Function to get the total energy a particle or any of its
    daughters deposited in the volume defined by the convex hull.
    Assumes that Cascades lose all of their energy in the convex
    hull if their vetex is in the hull. Otherwise the energy
    deposited by a cascade will be 0.
    (naive: There is possibly a better solution to this)

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    particle : I3Particle
        Particle.
        (Particle can be of any type: Muon, Cascade, Neutrino...)

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    energy : float
        Accumulated deposited Energy of the mother particle and
        all of the daughters.
    '''
    energy_loss = 0
    # Calculate EnergyLoss of current particle
    if particle.is_cascade:
        # cascade
        energy_loss = get_cascade_energy_deposited(frame, convex_hull,
                                                   particle)
    elif particle.is_neutrino \
            or particle.id == muongun_primary_neutrino_id:  # MuonGunFix
        # neutrinos
        for daughter in frame['I3MCTree'].get_daughters(particle):
            energy_loss += get_energy_deposited_including_daughters(
                                                frame, convex_hull, daughter)
    elif particle.pdg_encoding in (13, -13):  # CC [Muon +/-]
        energy_loss = get_muon_energy_deposited(frame, convex_hull, particle)

    # sanity Checks
    else:
        raise ValueError('Particle of type {} was not handled.'.format(
                                                                particle.type))
    assert energy_loss >= 0, 'Energy deposited is negativ'
    assert (energy_loss <= particle.energy + 1e-8 or
            particle.id == muongun_primary_neutrino_id), \
        'Deposited E is higher than total E'  # MuonGunFix
    return energy_loss


def get_muon_entry_info(frame, muon, convex_hull):
    """Helper function for 'get_cascade_labels'.

    Get muon information for point of entry, or closest approach point,
    if muon does not enter the volume defined by the convex_hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    muon : I3Particle
        Muon I3Particle for which to get the entry information.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.

    Returns
    -------
    I3Position, double, double
        Entry Point (or closest approach point)
        Time of entry point (or closest approach point)
        Energy at entry point (or closest approach point)
        Warning: If 'I3MCTree' does not exist in frame, this
                 will instead return the muon energy
    """
    entry = mu_utils.get_muon_initial_point_inside(frame, muon, convex_hull)
    if entry is None:
        # get closest approach point as entry approximation
        entry = mu_utils.get_muon_closest_approach_to_center(frame, muon)
    time = mu_utils.get_muon_time_at_position(frame, muon, entry)

    # Nancy's MuonGun simulation datasets do not have I3MCTree or MMCTrackList
    # included: use muon energy instead
    # This might be an ok approximation, since MuonGun muons are often injected
    # not too far out of detector volume
    if 'I3MCTree' not in frame:
        energy = muon.energy
    else:
        energy = mu_utils.get_muon_energy_at_position(frame, muon, entry)
    return entry, time, energy


def get_muon_bundle_information(frame, convex_hull, energy_threshold=100):
    """Calculate muon bundle information:

    Number of muons for certain selections, relative leading muon energy,
    bundle energy.

    This will calculate all muons in MMCTrackList, e.g. muons created inside
    the detector will also be considered and counted.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve MMCTrackList
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
    energy_threshold : int, optional
        Energy threshold in GeV at which to count muons.
        Muons below this threshold will be discarded.

    Returns
    -------
    dict
        A dictionary with the calculated labels.
    """
    bundle_info = {}

    energies_at_entry = []
    energies_at_cyl = []
    num_muons = 0

    for particle in frame['MMCTrackList']:
        muon = particle.particle
        # Check if particle is a muon
        if not mu_utils.is_muon(muon):
            continue

        # Determine entrance point into the convex hull
        initial_point = mu_utils.get_muon_initial_point_inside(
                                                    frame, muon, convex_hull)

        # Get energy at entry point
        if initial_point is not None:
            entry_energy = mu_utils.get_muon_energy_at_position(
                                                    frame, muon, initial_point)
        else:
            entry_energy = 0
        energies_at_entry.append(entry_energy)

        cyl_energy = particle.Ei
        energies_at_cyl.append(cyl_energy)
        num_muons += 1

    energies_at_entry = np.array(energies_at_entry)
    energies_at_entry = energies_at_entry[np.isfinite(energies_at_entry)]
    mult_mask = energies_at_entry >= energy_threshold
    bundle_info['num_muons_at_entry'] = len(energies_at_entry)
    bundle_info['num_muons_at_entry_above_threshold'] = len(
        energies_at_entry[mult_mask])
    bundle_info['leading_energy_rel_entry'] = np.max(
        energies_at_entry) / np.sum(energies_at_entry)

    energies_at_cyl = np.array(energies_at_cyl)
    energies_at_cyl = energies_at_cyl[np.isfinite(energies_at_cyl)]
    mult_mask = energies_at_cyl >= energy_threshold
    bundle_info['num_muons_at_cyl'] = len(energies_at_cyl)
    bundle_info['num_muons_at_cyl_above_threshold'] = len(
        energies_at_cyl[mult_mask])

    bundle_info['leading_energy_rel_cyl'] = np.max(
        energies_at_cyl) / np.sum(energies_at_cyl)

    bundle_info['bundle_energy_at_entry'] = np.sum(energies_at_entry)
    bundle_info['bundle_energy_at_cyl'] = np.sum(energies_at_cyl)
    bundle_info['num_muons'] = num_muons

    return bundle_info


def get_muon_information(frame, muon, dom_pos_dict,
                         convex_hull, pulse_map_string='InIcePulses'):
    '''Function to get labels for a muon

    Parameters
    ----------
    muon : I3Particle
        Muon.

    dom_pos_dict : dict
        Dictionary with key of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    '''

    # check if muon exists
    if muon is None:
        # create and return nan Values
        zero = dataclasses.I3Position(0, 0, 0)
        zero_dist_icecube = geometry.distance_to_icecube_hull(zero)
        zero_dist_deepcore = geometry.distance_to_deepcore_hull(zero)

        zero_dict = {
            'NoOfHitDOMs': 0,
            'NoOfPulses': 0,
            'TotalCharge': 0.,

            'COGDistanceToBorder': zero_dist_icecube,
            'COGDistanceToDeepCore': zero_dist_deepcore,
            'COGx': zero.x,
            'COGy': zero.y,
            'COGz': zero.z,

            'EntryDistanceToDeepCore': zero_dist_deepcore,
            'TimeAtEntry': 0.,
            'Entryx': zero.x,
            'Entryy': zero.y,
            'Entryz': zero.z,
            'EnergyEntry': 0.,

            'CenterDistanceToBorder': zero_dist_icecube,
            'CenterDistanceToDeepCore': zero_dist_deepcore,
            'TimeAtCenter': 0.,
            'Centerx': zero.x,
            'Centery': zero.y,
            'Centerz': zero.z,
            'EnergyCenter': 0.,

            'InDetectorTrackLength': 0.,
            'InDetectorEnergyLoss': 0.,

            'Azimuth': 0.,
            'Zenith': 0.,
            'Energy': 0.,
            'TotalTrackLength': 0.,
            'Vertexx': zero.x,
            'Vertexy': zero.y,
            'Vertexz': zero.z,
            'VertexDistanceToBorder': zero_dist_icecube,
            'VertexDistanceToDeepCore': zero_dist_deepcore,
        }
        return zero_dict

    # create empty information dictionary
    info_dict = {}

    # get labels depending on pulse map
    pulse_map = general.get_pulse_map(frame, muon,
                                      pulse_map_string=pulse_map_string)

    NoOfHitDOMs = len(pulse_map.keys())
    NoOfPulses = 0
    TotalCharge = 0.
    COG = np.array([0., 0., 0.])

    if NoOfHitDOMs > 0:
        for key in pulse_map.keys():
            for pulse in pulse_map[key]:
                NoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                COG += pos*pulse.charge
        COG = COG / TotalCharge
    COG = dataclasses.I3Position(*COG)

    COGDistanceToBorder = geometry.distance_to_icecube_hull(COG)
    COGDistanceToDeepCore = geometry.distance_to_deepcore_hull(COG)

    # get entry point labels
    Entry = mu_utils.get_muon_initial_point_inside(frame, muon, convex_hull)
    if Entry:
        TimeAtEntry = mu_utils.get_muon_time_at_position(frame, muon, Entry)
        EntryDistanceToDeepCore = geometry.distance_to_deepcore_hull(Entry)
        EnergyEntry = mu_utils.get_muon_energy_at_position(frame, muon, Entry)
    else:
        # handle missing values
        Entry = dataclasses.I3Position(0, 0, 0)
        TimeAtEntry = 0
        EntryDistanceToDeepCore = 0
        EnergyEntry = 0

    # get center point labels
    Center = mu_utils.get_muon_closest_approach_to_center(frame, muon)
    TimeAtCenter = mu_utils.get_muon_time_at_position(frame, muon, Center)
    CenterDistanceToBorder = geometry.distance_to_icecube_hull(Center)
    CenterDistanceToDeepCore = geometry.distance_to_deepcore_hull(Center)
    EnergyCenter = mu_utils.get_muon_energy_at_position(frame, muon, Center)

    # other labels
    InDetectorTrackLength = mu_utils.get_muon_track_length_inside(
                                                    frame, muon, convex_hull)
    InDetectorEnergyLoss = mu_utils.get_muon_energy_deposited(
                                                    frame, convex_hull, muon)

    # add labels to info_dict
    info_dict['NoOfHitDOMs'] = NoOfHitDOMs
    info_dict['NoOfPulses'] = NoOfPulses
    info_dict['TotalCharge'] = TotalCharge

    info_dict['COGDistanceToBorder'] = COGDistanceToBorder
    info_dict['COGDistanceToDeepCore'] = COGDistanceToDeepCore
    info_dict['COGx'] = COG.x
    info_dict['COGy'] = COG.y
    info_dict['COGz'] = COG.z

    info_dict['EntryDistanceToDeepCore'] = EntryDistanceToDeepCore
    info_dict['TimeAtEntry'] = TimeAtEntry
    info_dict['Entryx'] = Entry.x
    info_dict['Entryy'] = Entry.y
    info_dict['Entryz'] = Entry.z
    info_dict['EnergyEntry'] = EnergyEntry

    info_dict['CenterDistanceToBorder'] = CenterDistanceToBorder
    info_dict['CenterDistanceToDeepCore'] = CenterDistanceToDeepCore
    info_dict['TimeAtCenter'] = TimeAtCenter
    info_dict['Centerx'] = Center.x
    info_dict['Centery'] = Center.y
    info_dict['Centerz'] = Center.z
    info_dict['EnergyCenter'] = EnergyCenter

    info_dict['InDetectorTrackLength'] = InDetectorTrackLength
    info_dict['InDetectorEnergyLoss'] = InDetectorEnergyLoss

    info_dict['Azimuth'] = muon.dir.azimuth
    info_dict['Zenith'] = muon.dir.zenith
    info_dict['Energy'] = muon.energy
    info_dict['TotalTrackLength'] = muon.length
    info_dict['Vertexx'] = muon.pos.x
    info_dict['Vertexy'] = muon.pos.y
    info_dict['Vertexz'] = muon.pos.z
    info_dict['VertexDistanceToBorder'] = geometry.distance_to_icecube_hull(
                                                                    muon.pos)
    info_dict['VertexDistanceToDeepCore'] = geometry.distance_to_deepcore_hull(
                                                                    muon.pos)

    return info_dict


def get_primary_information(frame, primary,
                            dom_pos_dict, convex_hull,
                            pulse_map_string='InIcePulses',
                            muongun_primary_neutrino_id=None):
    '''Function to get labels for the primary

    Parameters
    ----------
    frame : frame

    primary : I3Particle
        Primary particle

    dom_pos_dict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    '''
    info_dict = {}

    # get labels depending on pulse map
    pulse_map = general.get_pulse_map(frame, primary,
                                      pulse_map_string=pulse_map_string)

    NoOfHitDOMs = len(pulse_map.keys())
    NoOfPulses = 0
    TotalCharge = 0.
    COG = np.array([0., 0., 0.])

    if NoOfHitDOMs > 0:
        for key in pulse_map.keys():
            for pulse in pulse_map[key]:
                NoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                COG += pos*pulse.charge
        COG = COG / TotalCharge
    COG = dataclasses.I3Position(*COG)

    COGDistanceToBorder = geometry.distance_to_icecube_hull(COG)
    COGDistanceToDeepCore = geometry.distance_to_deepcore_hull(COG)

    # other labels
    daughters = frame['I3MCTree'].get_daughters(primary)
    codes = [p.pdg_encoding for p in daughters]
    if -13 in codes or 13 in codes:
        # CC Interaction: nu + N -> mu + hadrons
        IsCCInteraction = True
    else:
        # NC Interaction: nu + N -> nu + hadrons
        IsCCInteraction = False

    if geometry.is_in_detector_bounds(daughters[0].pos):
        # Interaction of Primary is in Detector
        IsStartingTrack = True
    else:
        # Interaction outside of Detector
        IsStartingTrack = False
    InDetectorEnergyLoss = general.get_energy_deposited_including_daughters(
                    frame, convex_hull, primary,
                    muongun_primary_neutrino_id=muongun_primary_neutrino_id)

    # add labels to info_dict
    info_dict['NoOfHitDOMs'] = NoOfHitDOMs
    info_dict['NoOfPulses'] = NoOfPulses
    info_dict['TotalCharge'] = TotalCharge

    info_dict['COGDistanceToBorder'] = COGDistanceToBorder
    info_dict['COGDistanceToDeepCore'] = COGDistanceToDeepCore
    info_dict['COGx'] = COG.x
    info_dict['COGy'] = COG.y
    info_dict['COGz'] = COG.z

    info_dict['Azimuth'] = primary.dir.azimuth
    info_dict['Zenith'] = primary.dir.zenith
    info_dict['Energy'] = primary.energy
    info_dict['InDetectorEnergyLoss'] = InDetectorEnergyLoss
    info_dict['IsCCInteraction'] = IsCCInteraction
    info_dict['IsStartingTrack'] = IsStartingTrack

    return info_dict


def get_misc_information(frame,
                         dom_pos_dict, convex_hull,
                         pulse_map_string='InIcePulses'):
    '''Function to misc labels

    Parameters
    ----------
    frame : frame

    pulse_map_string : key of pulse map in frame,
        of which the mask should be computed for

    dom_pos_dict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    '''
    info_dict = {}
    in_ice_pulses = frame[pulse_map_string].apply(frame)

    TotalNoOfHitDOMs = len(in_ice_pulses.keys())
    TotalNoOfPulses = 0
    TotalCharge = 0.
    TotalCOG = np.array([0., 0., 0.])
    noise_pulses = []

    if TotalNoOfHitDOMs > 0:
        for key in in_ice_pulses.keys():
            for pulse in in_ice_pulses[key]:
                TotalNoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                TotalCOG += pos*pulse.charge
        TotalCOG = TotalCOG / TotalCharge
    TotalCOG = dataclasses.I3Position(*TotalCOG)

    noise_pulses = general.get_noise_pulse_map(
                                    frame, pulse_map_string=pulse_map_string)
    NoiseNoOfHitDOMs = len(noise_pulses.keys())
    NoiseNoOfPulses = 0
    NoiseTotalCharge = 0
    for key in noise_pulses.keys():
        for pulse in noise_pulses[key]:
            NoiseNoOfPulses += 1
            NoiseTotalCharge += pulse.charge

    info_dict['TotalNoOfHitDOMs'] = TotalNoOfHitDOMs
    info_dict['TotalNoOfPulses'] = TotalNoOfPulses
    info_dict['TotalCharge'] = TotalCharge
    info_dict['TotalCOGx'] = TotalCOG.x
    info_dict['TotalCOGy'] = TotalCOG.y
    info_dict['TotalCOGz'] = TotalCOG.z

    info_dict['NoiseNoOfHitDOMs'] = NoiseNoOfHitDOMs
    info_dict['NoiseNoOfPulses'] = NoiseNoOfPulses
    info_dict['NoiseTotalCharge'] = NoiseTotalCharge

    info_dict['NoOfPrimaries'] = len(frame['I3MCTree'].primaries)

    return info_dict


def get_labels(frame, convex_hull,
               domPosDict, primary,
               pulse_map_string='InIcePulses',
               is_muongun=False):
    '''Function to get extensive labels for muons, primary and general event
    data.

    Parameters
    ----------
    frame : frame

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    domPosDict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    primary : I3Particle
        Primary particle

    pulse_map_string : key of pulse map in frame,
        of which the mask should be computed for

    is_muongun : bool
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along to sub-functions.
        Technically, this could be done implicity, by setting
        the primary id. However, this will loosen up sanity
        checks. Therefore, an explicit decision to use MuonGun
        is prefered.

    Returns
    -------
    labels : I3MapStringDouble
        Dictionary with all labels
    '''

    if primary is None:
        raise ValueError('Primary does not exist!')

    assert primary.id is not None, 'MuonGunFix will not work if this is not true'

    # Check if MuonGun dataset
    if is_muongun:
        # This loosens up sanity checks, therefore
        # better to use only if it is really a
        # MuonGun set.
        # Should work for all datasets though,
        # as long as a primary exists

        # make sure it is a MuonGun dataset
        assert primary.type_string == 'unknown', 'Expected unknown, got {}'.format(primary.type_string)
        assert primary.pdg_encoding == 0, 'Expected 0,got {}'.format(primary.pdg_encoding)

        # set primary particle id
        muongun_primary_neutrino_id = primary.id
    else:
        muongun_primary_neutrino_id = None

    # create empty labelDict
    labels = dataclasses.I3MapStringDouble()

    # get misc info
    misc_info = get_misc_information(frame, domPosDict, convex_hull,
                                     pulse_map_string=pulse_map_string)
    labels.update(misc_info)

    muons_inside = mu_utils.get_muons_inside(frame, convex_hull)
    labels['NoOfMuonsInside'] = len(muons_inside)

    # get muons
    mostEnergeticMuon = mu_utils.get_most_energetic_muon_inside(
                                                frame, convex_hull,
                                                muons_inside=muons_inside)
    highestEDepositMuon = mu_utils.get_highest_deposit_muon_inside(
                                                frame, convex_hull,
                                                muons_inside=muons_inside)
    mostVisibleMuon = mu_utils.get_most_visible_muon_inside(
                                            frame, convex_hull,
                                            pulse_map_string=pulse_map_string)
    primaryMuon = mu_utils.get_next_muon_daughter_of_nu(
                    frame, primary,
                    muongun_primary_neutrino_id=muongun_primary_neutrino_id)

    labels['PrimaryMuonExists'] = not (primaryMuon is None)
    labels['VisibleStartingTrack'] = False
    for m in [mostEnergeticMuon, highestEDepositMuon, mostVisibleMuon,
              primaryMuon]:
        if m:
            if geometry.is_in_detector_bounds(m.pos, extend_boundary=60):
                labels['VisibleStartingTrack'] = True

    # get labels for most energetic muon
    mostEnergeticMuon_info = get_muon_information(
                            frame, mostEnergeticMuon, domPosDict, convex_hull,
                            pulse_map_string=pulse_map_string)
    for key in mostEnergeticMuon_info.keys():
        labels['MostEnergeticMuon'+key] = mostEnergeticMuon_info[key]

    # # get labels for highest deposit muon
    # if highestEDepositMuon == mostEnergeticMuon:
    #     highestEDepositMuon_info = mostEnergeticMuon_info
    # else:
    #     highestEDepositMuon_info = get_muon_information(frame,
    #             highestEDepositMuon, domPosDict, convex_hull,
    #             pulse_map_string=pulse_map_string)
    # for key in highestEDepositMuon_info.keys():
    #     labels['HighestEDepositMuon'+key] = highestEDepositMuon_info[key]

    # get labels for most visible muon
    if mostVisibleMuon == mostEnergeticMuon:
        mostVisibleMuon_info = mostEnergeticMuon_info
    else:
        mostVisibleMuon_info = get_muon_information(
                            frame, mostVisibleMuon, domPosDict, convex_hull,
                            pulse_map_string=pulse_map_string)
    for key in mostVisibleMuon_info.keys():
        labels['MostVisibleMuon'+key] = mostVisibleMuon_info[key]

    # get labels for muon from primary
    if primaryMuon == mostEnergeticMuon:
        primaryMuon_info = mostEnergeticMuon_info
    elif primaryMuon == mostVisibleMuon:
        primaryMuon_info = mostVisibleMuon_info
    else:
        primaryMuon_info = get_muon_information(
                                frame, primaryMuon, domPosDict, convex_hull,
                                pulse_map_string=pulse_map_string)
    for key in primaryMuon_info.keys():
        labels['PrimaryMuon'+key] = primaryMuon_info[key]

    # get labels for primary particle
    primary_info = get_primary_information(
                    frame, primary, domPosDict, convex_hull,
                    pulse_map_string=pulse_map_string,
                    muongun_primary_neutrino_id=muongun_primary_neutrino_id)
    for key in primary_info.keys():
        labels['Primary'+key] = primary_info[key]

    return labels


def get_cascade_labels(frame, primary, convex_hull, extend_boundary=0):
    """Get cascade labels.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        Will be used to compute muon entry point for an entering muon.
    extend_boundary : float, optional
        Extend boundary of convex_hull by this distance [in meters].

    Returns
    -------
    I3MapStringDouble
        Labels for cascade of primary neutrino.
    """
    labels = dataclasses.I3MapStringDouble()

    labels['PrimaryEnergy'] = primary.energy
    labels['PrimaryAzimuth'] = primary.dir.azimuth
    labels['PrimaryZenith'] = primary.dir.zenith
    labels['PrimaryDirectionX'] = primary.dir.x
    labels['PrimaryDirectionY'] = primary.dir.y
    labels['PrimaryDirectionZ'] = primary.dir.z

    # set pid variables to false per default
    labels['p_starting'] = 0
    labels['p_starting_300m'] = 0
    labels['p_starting_glashow'] = 0
    labels['p_starting_nc'] = 0
    labels['p_starting_cc'] = 0
    labels['p_starting_cc_e'] = 0
    labels['p_starting_cc_mu'] = 0
    labels['p_starting_cc_tau'] = 0
    labels['p_starting_cc_tau_muon_decay'] = 0
    labels['p_starting_cc_tau_double_bang'] = 0

    labels['p_entering'] = 0
    labels['p_entering_muon_single'] = 0
    labels['p_entering_muon_bundle'] = 0

    labels['p_outside_cascade'] = 0

    if primary.is_neutrino:
        # --------------------
        # NuGen dataset
        # --------------------
        mctree = frame['I3MCTree']
        cascade = get_cascade_of_primary_nu(frame, primary,
                                            convex_hull=None,
                                            extend_boundary=extend_boundary)

        # ---------------------------
        # 300m detector boundary test
        # ---------------------------
        cascade_300 = get_cascade_of_primary_nu(frame, primary,
                                                convex_hull=None,
                                                extend_boundary=300)
        if cascade_300 is not None:
            labels['p_starting_300m'] = 1
        # ---------------------------

        if cascade is None:
            # --------------------
            # not a starting event
            # --------------------
            muon = mu_utils.get_next_muon_daughter_of_nu(frame, primary)

            if muon is None:
                # --------------------
                # Cascade interaction outside of defined volume
                # --------------------
                # get first in ice neutrino
                nu_in_ice = None
                for p in mctree:
                    if p.is_neutrino and p.location_type_string == 'InIce':
                        nu_in_ice = p
                        break

                assert nu_in_ice is not None, 'Expected at least one in ice nu'

                daughters = mctree.get_daughters(nu_in_ice)
                visible_energy = 0.
                for d in daughters:
                    if d.is_neutrino:
                        # skip neutrino: the energy is not visible
                        continue
                    visible_energy += d.energy
                assert len(daughters) > 0, 'Expected at least one daughter!'

                labels['p_outside_cascade'] = 1
                labels['VertexX'] = daughters[0].pos.x
                labels['VertexY'] = daughters[0].pos.y
                labels['VertexZ'] = daughters[0].pos.z
                labels['VertexTime'] = daughters[0].time
                labels['EnergyVisible'] = visible_energy
            else:
                # ------------------------------
                # NuMu CC Muon entering detector
                # ------------------------------
                entry, time, energy = get_muon_entry_info(frame, muon,
                                                          convex_hull)
                labels['p_entering'] = 1
                labels['p_entering_muon_single'] = 1
                labels['VertexX'] = entry.x
                labels['VertexY'] = entry.y
                labels['VertexZ'] = entry.z
                labels['VertexTime'] = time
                labels['EnergyVisible'] = energy

        else:
            # --------------------
            # starting NuGen event
            # --------------------
            labels['VertexX'] = cascade.pos.x
            labels['VertexY'] = cascade.pos.y
            labels['VertexZ'] = cascade.pos.z
            labels['VertexTime'] = cascade.time
            labels['EnergyVisible'] = cascade.energy

            labels['p_starting'] = 1

            if frame['I3MCWeightDict']['InteractionType'] == 1:
                    # charged current
                    labels['p_starting_cc'] = 1

                    if cascade.type_string[:3] == 'NuE':
                        # cc NuE
                        labels['p_starting_cc_e'] = 1

                    elif cascade.type_string[:4] == 'NuMu':
                        # cc NuMu
                        labels['p_starting_cc_mu'] = 1

                    elif cascade.type_string[:5] == 'NuTau':
                        # cc Tau
                        labels['p_starting_cc_tau'] = 1

                        nu_tau = get_interaction_neutrino(
                                            frame, primary,
                                            convex_hull=None,
                                            extend_boundary=extend_boundary)
                        tau = [t for t in mctree.get_daughters(nu_tau)
                               if t.type_string in ['TauMinus', 'TauPlus']]

                        assert len(tau) == 1, 'Expected exactly 1 tau!'

                        mu = [m for m in mctree.get_daughters(tau[0])
                              if m.type_string in ['MuMinus', 'MuPlus']]

                        if len(mu) > 0:
                            # tau decays into muon: No Double bang signature!
                            labels['p_starting_cc_tau_muon_decay'] = 1
                        else:
                            # Double bang signature
                            labels['p_starting_cc_tau_double_bang'] = 1

                    else:
                        raise ValueError('Unexpected type: {!r}'.format(
                                                    cascade.type_string))

            elif frame['I3MCWeightDict']['InteractionType'] == 2:
                # neutral current (2)
                labels['p_starting_nc'] = 1

            elif frame['I3MCWeightDict']['InteractionType'] == 3:
                # glashow resonance (3)
                labels['p_starting_glashow'] = 1

            else:
                #  GN -- Genie
                print('InteractionType: {!r}'.format(
                                frame['I3MCWeightDict']['InteractionType']))

    elif mu_utils.is_muon(primary):
        # -----------------------------
        # muon primary: MuonGun dataset
        # -----------------------------
        entry, time, energy = get_muon_entry_info(frame, primary, convex_hull)
        labels['p_entering'] = 1
        labels['p_entering_muon_single'] = 1
        labels['VertexX'] = entry.x
        labels['VertexY'] = entry.y
        labels['VertexZ'] = entry.z
        labels['VertexTime'] = time
        labels['EnergyVisible'] = energy

    else:
        # ---------------------------------------------
        # No neutrino or muon primary: Corsika dataset?
        # ---------------------------------------------
        '''
        if single muon:
            entry, time, energy = get_muon_entry_info(frame, muon, convex_hull)
            labels['p_entering'] = 1
            labels['p_entering_muon_single'] = 1
            labels['VertexX'] = entry.pos.x
            labels['VertexY'] = entry.pos.y
            labels['VertexZ'] = entry.pos.z
            labels['VertexTime'] = time
            labels['EnergyVisible'] = energy
        elif muon bundle:
            muon = get_leading_muon()
            entry, time, energy = get_muon_entry_info(frame, muon, convex_hull)
            labels['p_entering'] = 1
            labels['p_entering_muon_bundle'] = 1
            labels['VertexX'] = entry.pos.x
            labels['VertexY'] = entry.pos.y
            labels['VertexZ'] = entry.pos.z
            labels['VertexTime'] = time
            labels['EnergyVisible'] = energy
        '''
        raise NotImplementedError('Primary type {!r} is not supported'.format(
                                                            primary.type))
    return labels


def get_cascade_parameters(frame, primary, convex_hull, extend_boundary=200):
    """Get cascade parameters.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        Will be used to compute muon entry point for an entering muon.
    extend_boundary : float, optional
        Extend boundary of convex_hull by this distance [in meters].

    Returns
    -------
    I3MapStringDouble
        Cascade parameters of primary neutrino: x, y, z, t, azimuth, zenith, E
    """
    labels = dataclasses.I3MapStringDouble()
    cascade = get_cascade_of_primary_nu(frame, primary,
                                        convex_hull=None,
                                        extend_boundary=extend_boundary)
    if cascade is None:
        # --------------------
        # not a starting event
        # --------------------
        muon = mu_utils.get_next_muon_daughter_of_nu(frame, primary)

        if muon is None:
            # --------------------
            # Cascade interaction outside of defined volume
            # --------------------
            mctree = frame['I3MCTree']
            # get first in ice neutrino
            nu_in_ice = None
            for p in mctree:
                if p.is_neutrino and p.location_type_string == 'InIce':
                    nu_in_ice = p
                    break

            assert nu_in_ice is not None, 'Expected at least one in ice nu'

            daughters = mctree.get_daughters(nu_in_ice)
            visible_energy = 0.
            for d in daughters:
                if d.is_neutrino:
                    # skip neutrino: the energy is not visible
                    continue
                visible_energy += d.energy
            assert len(daughters) > 0, 'Expected at least one daughter!'

            cascade = dataclasses.I3Particle()
            cascade.pos.x = daughters[0].pos.x
            cascade.pos.y = daughters[0].pos.y
            cascade.pos.z = daughters[0].pos.z
            cascade.time = daughters[0].time
            cascade.energy = visible_energy
            cascade.dir = dataclasses.I3Direction(nu_in_ice.dir)
        else:
            # ------------------------------
            # NuMu CC Muon entering detector
            # ------------------------------
            # set cascade parameters to muon entry information
            entry, time, energy = get_muon_entry_info(frame, muon,
                                                      convex_hull)
            cascade = dataclasses.I3Particle()
            cascade.pos.x = entry.x
            cascade.pos.y = entry.y
            cascade.pos.z = entry.z
            cascade.time = time
            cascade.energy = energy
            cascade.dir = dataclasses.I3Direction(muon.dir)

    frame['MCCascade'] = cascade

    labels['cascade_x'] = cascade.pos.x
    labels['cascade_y'] = cascade.pos.y
    labels['cascade_z'] = cascade.pos.z
    labels['cascade_t'] = cascade.time
    labels['cascade_energy'] = cascade.energy
    labels['cascade_azimuth'] = cascade.dir.azimuth
    labels['cascade_zenith'] = cascade.dir.zenith

    return labels
