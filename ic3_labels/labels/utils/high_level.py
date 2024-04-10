"""Helper functions for icecube specific labels.
"""

from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils import general
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils import tau as tau_utils
from ic3_labels.labels.utils.cascade import get_cascade_of_primary_nu
from ic3_labels.labels.utils.cascade import get_cascade_energy_deposited
from ic3_labels.labels.utils.cascade import get_interaction_extension_length
from ic3_labels.labels.utils.cascade import convert_to_em_equivalent
from ic3_labels.labels.utils.neutrino import get_interaction_neutrino


def get_total_deposited_energy(
    frame, convex_hull=None, extend_boundary=None, cylinder_ext=None
):
    """Get total deposited energy in an event.

    Traverses the I3MCTree and collects energies of particles.
    The particles are handled in the following:

        dark particles: ignore
        particles not InIce or in convex hull (if provided): ignore
        neutrinos: ignore
        taus and muons: ignore
            --> energy losses and decay products are collected
            --> ionisation energy losses are disregarded
            --> low energy muons created in cascades are disregarded
        electron, hadrons, ...: collect EM equivalent energy

    Note: the InIce volume is rather large. To provide additional and
    more stringent definitions of the detector volume, a convex hull, an
    extended IceCube boundary, or a simple cut on the radius can be applied.
    In this case, the InIce check will be performed in addition to:

        If convex_hull is not None: check if particle is in convex hull
        If extend_boundary is not None: check if particle is in extended
                                        IceCube boundary.
        If cylinder_ext is not None: check if particle is within the extended
                                     cylinder (z +- 500 + ext, r=500 + ext)


    Parameters
    ----------
    frame : I3Frame
        Current I3Frame.
    convex_hull : scipy.spatial.ConvexHull or None, optional
        Defines the desired convex volume to check whether an energy deposit
        was inside the detector volume.
    extend_boundary : float or None, optional
        Use a convex hull around the IceCube detector and extend it by this
        distance [in meters] to check if an energy deposit was in the detector
    cylinder_ext : float or None, optional
        If provided, energy losses with a radius in x-y > 500 + cylinder_ext
        and abs(z) > 500 + cylinder_ext will be discarded.

    Returns
    -------
    double
        The deposited energy.
    """
    deposited_energy = 0.0

    for p in frame["I3MCTree"]:

        # skip dark particles
        if p.shape == dataclasses.I3Particle.ParticleShape.Dark:
            continue

        # skip neutrino: the energy is not visible
        if p.is_neutrino:
            continue

        # skip muons and taus:
        # --> energy losses and decay products are still collected
        # --> ionisation energy losses are disregarded
        # --> low energy muons created in cascades are disregarded
        if p.type in [
            dataclasses.I3Particle.ParticleType.MuPlus,
            dataclasses.I3Particle.ParticleType.MuMinus,
            dataclasses.I3Particle.ParticleType.TauMinus,
            dataclasses.I3Particle.ParticleType.TauPlus,
        ]:
            continue

        # Check if the energy deposit was inside the detector.
        # Ignore it, if it was outside.

        if p.location_type != dataclasses.I3Particle.LocationType.InIce:
            # skip particles that are way outside of the detector volume
            continue

        # use a basic cylinder to determine if particle was inside
        if cylinder_ext is not None:
            dist = 500 + cylinder_ext
            if (
                p.pos.z > dist
                or p.pos.z < -dist
                or p.pos.x**2 + p.pos.y**2 > dist**2
            ):
                continue

        if convex_hull is not None:
            # use convex hull to determine if inside detector
            if not geometry.point_is_inside(convex_hull, p.pos):
                continue

        if extend_boundary is not None:
            # use IceCube boundary + extent_boundary [meters] to check
            if not geometry.is_in_detector_bounds(
                p.pos, extend_boundary=extend_boundary
            ):
                continue

        # scale energy of cascades to EM equivalent
        deposited_energy += convert_to_em_equivalent(p)

    return deposited_energy


def get_energy_deposited(frame, convex_hull, particle):
    """Function to get the total energy a particle deposited in the
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
    """

    raise NotImplementedError


def get_energy_deposited_including_daughters(
    frame,
    convex_hull,
    particle,
    muongun_primary_neutrino_id=None,
    mctree_name="I3MCTree",
    track_cache=None,
):
    """Function to get the total energy a particle or any of its
    daughters deposited in the volume defined by the convex hull.
    Assumes that Cascades lose all of their energy in the convex
    hull if their vertex is in the hull. Otherwise the energy
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
    mctree_name : str, optional
        The name of the I3MCTree to use.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    energy : float
        Accumulated deposited Energy of the mother particle and
        all of the daughters.

    Raises
    ------
    ValueError
        Description
    """
    energy_loss = 0
    # Calculate EnergyLoss of current particle
    if particle.is_cascade:
        # cascade
        energy_loss = get_cascade_energy_deposited(
            frame, convex_hull, particle
        )
    elif (
        particle.is_neutrino or particle.id == muongun_primary_neutrino_id
    ):  # MuonGunFix
        # neutrinos
        for daughter in frame[mctree_name].get_daughters(particle):
            energy_loss += get_energy_deposited_including_daughters(
                frame, convex_hull, daughter, track_cache=track_cache
            )
    elif particle.pdg_encoding in (13, -13):  # CC [Muon +/-]
        energy_loss = mu_utils.get_muon_energy_deposited(
            frame, convex_hull, particle, track_cache=track_cache
        )

    # sanity Checks
    else:
        raise ValueError(
            "Particle of type {} was not handled.".format(particle.type)
        )
    assert energy_loss >= 0, "Energy deposited is negative"
    assert (
        energy_loss <= particle.energy + 1e-8
        or particle.id == muongun_primary_neutrino_id
    ), "Deposited E is higher than total E"  # MuonGunFix
    return energy_loss


def get_tau_entry_info(frame, tau, convex_hull, mctree_name="I3MCTree"):
    """Helper function for 'get_cascade_labels'.

    Get tau information for point of entry, or closest approach point,
    if tau does not enter the volume defined by the convex_hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    tau : I3Particle
        Tau I3Particle for which to get the entry information.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
    mctree_name : str, optional
        The name of the I3MCTree to use.

    Returns
    -------
    I3Position, double, double
        Entry Point (or closest approach point)
        Time of entry point (or closest approach point)
        Energy at entry point (or closest approach point)
        Warning: If 'I3MCTree' does not exist in frame, this
                 will instead return the muon energy
    """
    entry = mu_utils.get_muon_initial_point_inside(tau, convex_hull)
    if entry is None:
        # get closest approach point as entry approximation
        entry = mu_utils.get_particle_closest_approach_to_position(
            tau, dataclasses.I3Position(0, 0, 0)
        )
    time = mu_utils.get_muon_time_at_position(tau, entry)

    # Todo: calculate energy at point of entry for tau as it is done for muon
    # For now, just provide the total tau energy
    if mctree_name not in frame:
        energy = tau.energy
    else:
        energy = tau.energy
        # energy = mu_utils.get_muon_energy_at_position(frame, tau, entry)
    return entry, time, energy


def get_muon_bundle_information(
    frame,
    convex_hull,
    energy_threshold=20,
    track_cache=None,
):
    """Calculate muon bundle information:

    Number of muons for certain selections, relative leading muon energy,
    bundle energy.

    This will calculate all muons in MMCTrackList for 'cyl', but for 'entry'
    starting muons will not be considered.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve MMCTrackList
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
    energy_threshold : int, optional
        Energy threshold in GeV at which to count muons.
        Muons below this threshold will be discarded.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    dict
        A dictionary with the calculated labels.
    """
    bundle_info = {}

    energies_at_entry = []
    energies_at_cyl = []

    for particle in frame["MMCTrackList"]:
        muon = particle.particle
        # Check if particle is a muon
        if not mu_utils.is_muon(muon):
            continue

        # Determine entrance point into the convex hull
        initial_point = mu_utils.get_muon_initial_point_inside(
            muon, convex_hull
        )

        # Get energy at entry point
        if initial_point is not None:
            # check if it is a starting muon, e.g. if initial point inside
            # is the same as the vertex (Discard muon in this case)
            if (initial_point - muon.pos).magnitude > 1:
                entry_energy = mu_utils.get_muon_energy_at_position(
                    frame, muon, initial_point, track_cache=track_cache
                )
                energies_at_entry.append(entry_energy)

        cyl_energy = particle.Ei
        energies_at_cyl.append(cyl_energy)

    # collect all muons in I3MCTree that aren't generated by other muons
    muons = mu_utils.get_parent_muons(frame)
    energies_in_mctree = np.array(sorted([m.energy for m in muons]))
    bundle_info["num_muons_in_mctree"] = len(muons)

    energies_at_entry = np.array(sorted(energies_at_entry))
    energies_at_entry = energies_at_entry[np.isfinite(energies_at_entry)]
    mult_mask = energies_at_entry >= energy_threshold
    bundle_info["num_muons_at_entry"] = len(energies_at_entry)
    bundle_info["num_muons_at_entry_above_threshold"] = len(
        energies_at_entry[mult_mask]
    )

    energies_at_cyl = np.array(sorted(energies_at_cyl))
    energies_at_cyl = energies_at_cyl[np.isfinite(energies_at_cyl)]
    mult_mask = energies_at_cyl >= energy_threshold
    bundle_info["num_muons_at_cyl"] = len(energies_at_cyl)
    bundle_info["num_muons_at_cyl_above_threshold"] = len(
        energies_at_cyl[mult_mask]
    )

    # add info on relative energy fractions of most and second most energetic
    for energies, name in zip(
        [energies_in_mctree, energies_at_cyl, energies_at_entry],
        ["_mctree", "_cyl", "_entry"],
    ):
        total_energy = np.sum(energies)

        if len(energies) > 1:
            bundle_info["leading_energy_rel_2nd" + name] = (
                energies[-2] / total_energy
            )
        else:
            bundle_info["leading_energy_rel_2nd" + name] = float("NaN")

        if len(energies) > 0:
            bundle_info["leading_energy_rel" + name] = (
                energies[-1] / total_energy
            )
        else:
            bundle_info["leading_energy_rel" + name] = float("NaN")

    bundle_info["bundle_energy_at_entry"] = np.sum(energies_at_entry)
    bundle_info["bundle_energy_at_cyl"] = np.sum(energies_at_cyl)
    bundle_info["bundle_energy_in_mctree"] = np.sum(energies_in_mctree)

    return bundle_info


def get_muon_information(
    frame,
    muon,
    dom_pos_dict,
    convex_hull,
    pulse_map_string="InIcePulses",
    mcpe_series_map_name="I3MCPESeriesMap",
    track_cache=None,
):
    """Function to get labels for a muon

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame.
    muon : I3Particle
        Muon.
    dom_pos_dict : dict
        Dictionary with key of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume
    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for
    mcpe_series_map_name : str, optional
        The name if the I3MCPESeriesMap
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    """

    # check if muon exists
    if muon is None:
        # create and return nan Values
        zero = dataclasses.I3Position(0, 0, 0)
        zero_dist_icecube = geometry.distance_to_icecube_hull(zero)
        zero_dist_deepcore = geometry.distance_to_deepcore_hull(zero)

        zero_dict = {
            "NoOfHitDOMs": 0,
            "NoOfPulses": 0,
            "TotalCharge": 0.0,
            "COGDistanceToBorder": zero_dist_icecube,
            "COGDistanceToDeepCore": zero_dist_deepcore,
            "COGx": zero.x,
            "COGy": zero.y,
            "COGz": zero.z,
            "EntryDistanceToDeepCore": zero_dist_deepcore,
            "TimeAtEntry": 0.0,
            "Entryx": zero.x,
            "Entryy": zero.y,
            "Entryz": zero.z,
            "EnergyEntry": 0.0,
            "CenterDistanceToBorder": zero_dist_icecube,
            "CenterDistanceToDeepCore": zero_dist_deepcore,
            "TimeAtCenter": 0.0,
            "Centerx": zero.x,
            "Centery": zero.y,
            "Centerz": zero.z,
            "EnergyCenter": 0.0,
            "ExitDistanceToDeepCore": zero_dist_deepcore,
            "TimeAtExit": 0.0,
            "Exitx": zero.x,
            "Exity": zero.y,
            "Exitz": zero.z,
            "EnergyExit": 0.0,
            "InDetectorTrackLength": 0.0,
            "InDetectorEnergyLoss": 0.0,
            "Azimuth": 0.0,
            "Zenith": 0.0,
            "Energy": 0.0,
            "TotalTrackLength": 0.0,
            "Vertexx": zero.x,
            "Vertexy": zero.y,
            "Vertexz": zero.z,
            "VertexDistanceToBorder": zero_dist_icecube,
            "VertexDistanceToDeepCore": zero_dist_deepcore,
        }
        return zero_dict

    # create empty information dictionary
    info_dict = {}

    # get labels depending on pulse map
    pulse_map = general.get_pulse_map(
        frame,
        muon,
        pulse_map_string=pulse_map_string,
        mcpe_series_map_name=mcpe_series_map_name,
    )

    NoOfHitDOMs = len(pulse_map.keys())
    NoOfPulses = 0
    TotalCharge = 0.0
    COG = np.array([0.0, 0.0, 0.0])

    if NoOfHitDOMs > 0:
        for key in pulse_map.keys():
            for pulse in pulse_map[key]:
                NoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                COG += pos * pulse.charge
        COG = COG / TotalCharge
    COG = dataclasses.I3Position(*COG)

    COGDistanceToBorder = geometry.distance_to_icecube_hull(COG)
    COGDistanceToDeepCore = geometry.distance_to_deepcore_hull(COG)

    # get entry point labels
    Entry = mu_utils.get_muon_initial_point_inside(muon, convex_hull)
    if Entry:
        TimeAtEntry = mu_utils.get_muon_time_at_position(muon, Entry)
        EntryDistanceToDeepCore = geometry.distance_to_deepcore_hull(Entry)
        EnergyEntry = mu_utils.get_muon_energy_at_position(
            frame, muon, Entry, track_cache=track_cache
        )
    else:
        # handle missing values
        Entry = dataclasses.I3Position(0, 0, 0)
        TimeAtEntry = 0
        EntryDistanceToDeepCore = 0
        EnergyEntry = 0

    # get exit point labels
    Exit = mu_utils.get_muon_exit_point(muon, convex_hull)
    if Exit:
        TimeAtExit = mu_utils.get_muon_time_at_position(muon, Exit)
        ExitDistanceToDeepCore = geometry.distance_to_deepcore_hull(Exit)
        EnergyExit = mu_utils.get_muon_energy_at_position(
            frame, muon, Exit, track_cache=track_cache
        )
    else:
        # handle missing values
        Exit = dataclasses.I3Position(0, 0, 0)
        TimeAtExit = 0
        ExitDistanceToDeepCore = 0
        EnergyExit = 0

    # get center point labels
    Center = mu_utils.get_muon_closest_approach_to_center(frame, muon)
    TimeAtCenter = mu_utils.get_muon_time_at_position(muon, Center)
    CenterDistanceToBorder = geometry.distance_to_icecube_hull(Center)
    CenterDistanceToDeepCore = geometry.distance_to_deepcore_hull(Center)
    EnergyCenter = mu_utils.get_muon_energy_at_position(
        frame, muon, Center, track_cache=track_cache
    )

    # other labels
    InDetectorTrackLength = mu_utils.get_muon_track_length_inside(
        muon, convex_hull
    )
    InDetectorEnergyLoss = mu_utils.get_muon_energy_deposited(
        frame, convex_hull, muon, track_cache=track_cache
    )

    # add labels to info_dict
    info_dict["NoOfHitDOMs"] = NoOfHitDOMs
    info_dict["NoOfPulses"] = NoOfPulses
    info_dict["TotalCharge"] = TotalCharge

    info_dict["COGDistanceToBorder"] = COGDistanceToBorder
    info_dict["COGDistanceToDeepCore"] = COGDistanceToDeepCore
    info_dict["COGx"] = COG.x
    info_dict["COGy"] = COG.y
    info_dict["COGz"] = COG.z

    info_dict["EntryDistanceToDeepCore"] = EntryDistanceToDeepCore
    info_dict["TimeAtEntry"] = TimeAtEntry
    info_dict["Entryx"] = Entry.x
    info_dict["Entryy"] = Entry.y
    info_dict["Entryz"] = Entry.z
    info_dict["EnergyEntry"] = EnergyEntry

    info_dict["CenterDistanceToBorder"] = CenterDistanceToBorder
    info_dict["CenterDistanceToDeepCore"] = CenterDistanceToDeepCore
    info_dict["TimeAtCenter"] = TimeAtCenter
    info_dict["Centerx"] = Center.x
    info_dict["Centery"] = Center.y
    info_dict["Centerz"] = Center.z
    info_dict["EnergyCenter"] = EnergyCenter

    info_dict["ExitDistanceToDeepCore"] = ExitDistanceToDeepCore
    info_dict["TimeAtExit"] = TimeAtExit
    info_dict["Exitx"] = Exit.x
    info_dict["Exity"] = Exit.y
    info_dict["Exitz"] = Exit.z
    info_dict["EnergyExit"] = EnergyExit

    info_dict["InDetectorTrackLength"] = InDetectorTrackLength
    info_dict["InDetectorEnergyLoss"] = InDetectorEnergyLoss

    info_dict["Azimuth"] = muon.dir.azimuth
    info_dict["Zenith"] = muon.dir.zenith
    info_dict["Energy"] = muon.energy
    info_dict["TotalTrackLength"] = muon.length
    info_dict["Vertexx"] = muon.pos.x
    info_dict["Vertexy"] = muon.pos.y
    info_dict["Vertexz"] = muon.pos.z
    info_dict["VertexDistanceToBorder"] = geometry.distance_to_icecube_hull(
        muon.pos
    )
    info_dict["VertexDistanceToDeepCore"] = geometry.distance_to_deepcore_hull(
        muon.pos
    )

    return info_dict


def get_primary_information(
    frame,
    primary,
    dom_pos_dict,
    convex_hull,
    pulse_map_string="InIcePulses",
    mcpe_series_map_name="I3MCPESeriesMap",
    muongun_primary_neutrino_id=None,
    mctree_name="I3MCTree",
    track_cache=None,
):
    """Function to get labels for the primary

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
    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for
    mcpe_series_map_name : key of mcpe series map in frame
    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.
    mctree_name : str, optional
        The name of the I3MCTree to use.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    """
    info_dict = {}

    # get labels depending on pulse map
    pulse_map = general.get_pulse_map(
        frame,
        primary,
        pulse_map_string=pulse_map_string,
        mcpe_series_map_name=mcpe_series_map_name,
    )

    NoOfHitDOMs = len(pulse_map.keys())
    NoOfPulses = 0
    TotalCharge = 0.0
    COG = np.array([0.0, 0.0, 0.0])

    if NoOfHitDOMs > 0:
        for key in pulse_map.keys():
            for pulse in pulse_map[key]:
                NoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                COG += pos * pulse.charge
        COG = COG / TotalCharge
    COG = dataclasses.I3Position(*COG)

    COGDistanceToBorder = geometry.distance_to_icecube_hull(COG)
    COGDistanceToDeepCore = geometry.distance_to_deepcore_hull(COG)

    # other labels
    daughters = frame[mctree_name].get_daughters(primary)
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
    InDetectorEnergyLoss = get_energy_deposited_including_daughters(
        frame,
        convex_hull,
        primary,
        muongun_primary_neutrino_id=muongun_primary_neutrino_id,
        track_cache=track_cache,
    )

    # add labels to info_dict
    info_dict["NoOfHitDOMs"] = NoOfHitDOMs
    info_dict["NoOfPulses"] = NoOfPulses
    info_dict["TotalCharge"] = TotalCharge

    info_dict["COGDistanceToBorder"] = COGDistanceToBorder
    info_dict["COGDistanceToDeepCore"] = COGDistanceToDeepCore
    info_dict["COGx"] = COG.x
    info_dict["COGy"] = COG.y
    info_dict["COGz"] = COG.z

    info_dict["Azimuth"] = primary.dir.azimuth
    info_dict["Zenith"] = primary.dir.zenith
    info_dict["Energy"] = primary.energy
    info_dict["InDetectorEnergyLoss"] = InDetectorEnergyLoss
    info_dict["IsCCInteraction"] = IsCCInteraction
    info_dict["IsStartingTrack"] = IsStartingTrack

    return info_dict


def get_misc_information(
    frame,
    dom_pos_dict,
    convex_hull,
    pulse_map_string="InIcePulses",
    mcpe_series_map_name="I3MCPESeriesMap",
    mctree_name="I3MCTree",
):
    """Function to misc labels

    Parameters
    ----------
    frame : I3Frame
        The I3Frame to work on
    dom_pos_dict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume
    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for
    mcpe_series_map_name : str, optional
        Description
    mctree_name : str, optional
        The name of the I3MCTree to use.
    mcpe_series_map_name : key of mcpe series map in frame

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    """
    info_dict = {}
    in_ice_pulses = frame[pulse_map_string].apply(frame)

    TotalNoOfHitDOMs = len(in_ice_pulses.keys())
    TotalNoOfPulses = 0
    TotalCharge = 0.0
    TotalCOG = np.array([0.0, 0.0, 0.0])
    noise_pulses = []

    if TotalNoOfHitDOMs > 0:
        for key in in_ice_pulses.keys():
            for pulse in in_ice_pulses[key]:
                TotalNoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                TotalCOG += pos * pulse.charge
        TotalCOG = TotalCOG / TotalCharge
    TotalCOG = dataclasses.I3Position(*TotalCOG)

    noise_pulses = general.get_noise_pulse_map(
        frame,
        pulse_map_string=pulse_map_string,
        mcpe_series_map_name=mcpe_series_map_name,
    )
    NoiseNoOfHitDOMs = len(noise_pulses.keys())
    NoiseNoOfPulses = 0
    NoiseTotalCharge = 0
    for key in noise_pulses.keys():
        for pulse in noise_pulses[key]:
            NoiseNoOfPulses += 1
            NoiseTotalCharge += pulse.charge

    info_dict["TotalNoOfHitDOMs"] = TotalNoOfHitDOMs
    info_dict["TotalNoOfPulses"] = TotalNoOfPulses
    info_dict["TotalCharge"] = TotalCharge
    info_dict["TotalCOGx"] = TotalCOG.x
    info_dict["TotalCOGy"] = TotalCOG.y
    info_dict["TotalCOGz"] = TotalCOG.z

    info_dict["NoiseNoOfHitDOMs"] = NoiseNoOfHitDOMs
    info_dict["NoiseNoOfPulses"] = NoiseNoOfPulses
    info_dict["NoiseTotalCharge"] = NoiseTotalCharge

    info_dict["NoOfPrimaries"] = len(frame[mctree_name].primaries)

    return info_dict


def get_labels(
    frame,
    convex_hull,
    domPosDict,
    primary,
    pulse_map_string="InIcePulses",
    mcpe_series_map_name="I3MCPESeriesMap",
    is_muongun=False,
    track_cache=None,
):
    """Function to get extensive labels for muons, primary and general event
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
    mcpe_series_map_name : key of mcpe series map in frame
    is_muongun : bool
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along to sub-functions.
        Technically, this could be done implicitly, by setting
        the primary id. However, this will loosen up sanity
        checks. Therefore, an explicit decision to use MuonGun
        is preferred.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    labels : I3MapStringDouble
        Dictionary with all labels
    """

    if primary is None:
        raise ValueError("Primary does not exist!")

    assert (
        primary.id is not None
    ), "MuonGunFix will not work if this is not true"

    # Check if MuonGun dataset
    if is_muongun:
        # This loosens up sanity checks, therefore
        # better to use only if it is really a
        # MuonGun set.
        # Should work for all datasets though,
        # as long as a primary exists

        # make sure it is a MuonGun dataset
        assert (
            primary.type_string == "unknown"
        ), "Expected unknown, got {}".format(primary.type_string)
        assert primary.pdg_encoding == 0, "Expected 0,got {}".format(
            primary.pdg_encoding
        )

        # set primary particle id
        muongun_primary_neutrino_id = primary.id
    else:
        muongun_primary_neutrino_id = None

    # create empty labelDict
    labels = dataclasses.I3MapStringDouble()

    # get misc info
    misc_info = get_misc_information(
        frame,
        domPosDict,
        convex_hull,
        pulse_map_string=pulse_map_string,
        mcpe_series_map_name=mcpe_series_map_name,
    )
    labels.update(misc_info)

    muons_inside = mu_utils.get_muons_inside(frame, convex_hull)
    labels["NoOfMuonsInside"] = len(muons_inside)

    # get muons
    mostEnergeticMuon = mu_utils.get_most_energetic_muon_inside(
        frame, convex_hull, muons_inside=muons_inside, track_cache=track_cache
    )

    highestEDepositMuon = mu_utils.get_highest_deposit_muon_inside(
        frame, convex_hull, muons_inside=muons_inside, track_cache=track_cache
    )

    mostVisibleMuon = mu_utils.get_most_visible_muon_inside(
        frame,
        convex_hull,
        pulse_map_string=pulse_map_string,
        mcpe_series_map_name=mcpe_series_map_name,
    )
    primaryMuon = mu_utils.get_muon_of_inice_neutrino(
        frame, muongun_primary_neutrino_id=muongun_primary_neutrino_id
    )

    labels["PrimaryMuonExists"] = primaryMuon is not None
    labels["VisibleStartingTrack"] = False
    for m in [
        mostEnergeticMuon,
        highestEDepositMuon,
        mostVisibleMuon,
        primaryMuon,
    ]:
        if m:
            if geometry.is_in_detector_bounds(m.pos, extend_boundary=60):
                labels["VisibleStartingTrack"] = True

    # get labels for most energetic muon
    mostEnergeticMuon_info = get_muon_information(
        frame,
        mostEnergeticMuon,
        domPosDict,
        convex_hull,
        pulse_map_string=pulse_map_string,
        track_cache=track_cache,
    )
    for key in mostEnergeticMuon_info.keys():
        labels["MostEnergeticMuon" + key] = mostEnergeticMuon_info[key]

    # get labels for most visible muon
    if mostVisibleMuon == mostEnergeticMuon:
        mostVisibleMuon_info = mostEnergeticMuon_info
    else:
        mostVisibleMuon_info = get_muon_information(
            frame,
            mostVisibleMuon,
            domPosDict,
            convex_hull,
            pulse_map_string=pulse_map_string,
            track_cache=track_cache,
        )
    for key in mostVisibleMuon_info.keys():
        labels["MostVisibleMuon" + key] = mostVisibleMuon_info[key]

    # get labels for muon from primary
    if primaryMuon == mostEnergeticMuon:
        primaryMuon_info = mostEnergeticMuon_info
    elif primaryMuon == mostVisibleMuon:
        primaryMuon_info = mostVisibleMuon_info
    else:
        primaryMuon_info = get_muon_information(
            frame,
            primaryMuon,
            domPosDict,
            convex_hull,
            pulse_map_string=pulse_map_string,
            track_cache=track_cache,
        )
    for key in primaryMuon_info.keys():
        labels["PrimaryMuon" + key] = primaryMuon_info[key]

    # get labels for primary particle
    primary_info = get_primary_information(
        frame,
        primary,
        domPosDict,
        convex_hull,
        pulse_map_string=pulse_map_string,
        muongun_primary_neutrino_id=muongun_primary_neutrino_id,
        track_cache=track_cache,
    )
    for key in primary_info.keys():
        labels["Primary" + key] = primary_info[key]

    return labels


def get_cascade_labels(
    frame,
    primary,
    convex_hull,
    extend_boundary=0,
    track_length_threshold=30,
    mctree_name="I3MCTree",
    track_cache=None,
):
    """Get general cascade labels.

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
    track_length_threshold : int, optional
        The minimum length (in meter) of a cascade/muon after which it is
        considered as a track event.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    I3MapStringDouble
        Labels for cascade of primary neutrino.

    Raises
    ------
    ValueError
        Description
    """
    labels = dataclasses.I3MapStringDouble()

    labels["num_coincident_events"] = general.get_num_coincident_events(frame)

    bundle_info = get_muon_bundle_information(
        frame=frame, convex_hull=convex_hull, track_cache=track_cache
    )
    for k in [
        "leading_energy_rel_entry",
        "num_muons_at_entry",
        "num_muons_at_entry_above_threshold",
    ]:
        labels[k] = bundle_info[k]

    if not np.isfinite(labels["leading_energy_rel_entry"]):
        labels["leading_energy_rel_entry"] = 0.0

    labels["TotalDepositedEnergy"] = get_total_deposited_energy(
        frame, extend_boundary=300
    )
    labels["PrimaryEnergy"] = primary.energy
    labels["PrimaryAzimuth"] = primary.dir.azimuth
    labels["PrimaryZenith"] = primary.dir.zenith
    labels["PrimaryDirectionX"] = primary.dir.x
    labels["PrimaryDirectionY"] = primary.dir.y
    labels["PrimaryDirectionZ"] = primary.dir.z

    # set pid variables to false per default
    labels["p_starting"] = 0
    labels["p_starting_300m"] = 0
    labels["p_starting_glashow"] = 0
    labels["p_starting_nc"] = 0
    labels["p_starting_cc"] = 0
    labels["p_starting_cc_e"] = 0
    labels["p_starting_cc_mu"] = 0
    labels["p_starting_cc_tau"] = 0
    labels["p_starting_cc_tau_muon_decay"] = 0
    labels["p_starting_cc_tau_double_bang"] = 0

    labels["p_entering"] = 0
    labels["p_entering_muon_single"] = 0
    labels["p_entering_muon_single_stopping"] = 0
    labels["p_entering_muon_bundle"] = 0

    labels["p_outside_cascade"] = 0

    if primary.is_neutrino:
        # --------------------
        # NuGen dataset
        # --------------------
        mctree = frame[mctree_name]
        cascade = get_cascade_of_primary_nu(
            frame, primary, convex_hull=None, extend_boundary=extend_boundary
        )[0]

        # ---------------------------
        # 300m detector boundary test
        # ---------------------------
        cascade_300 = get_cascade_of_primary_nu(
            frame, primary, convex_hull=None, extend_boundary=300
        )[0]
        if cascade_300 is not None:
            labels["p_starting_300m"] = 1
        # ---------------------------

        if cascade is None:
            # --------------------
            # not a starting event
            # --------------------
            muon = mu_utils.get_muon_of_inice_neutrino(frame)

            if muon is None:
                tau = tau_utils.get_tau_of_inice_neutrino(frame)

                if tau is None:
                    # --------------------
                    # Cascade interaction outside of defined volume
                    # Note: this could still be a muon created in a hadronic
                    # shower that can enter the detector and look like a
                    # normal track event! toDo: check this?
                    # --------------------
                    cascade = get_cascade_of_primary_nu(
                        frame,
                        primary,
                        convex_hull=None,
                        extend_boundary=float("inf"),
                        sanity_check=False,
                    )[0]

                    labels["p_outside_cascade"] = 1
                    labels["VertexX"] = cascade.pos.x
                    labels["VertexY"] = cascade.pos.y
                    labels["VertexZ"] = cascade.pos.z
                    labels["VertexTime"] = cascade.time
                    labels["EnergyVisible"] = cascade.energy
                    labels["Length"] = cascade.length
                    labels["LengthInDetector"] = (
                        mu_utils.get_muon_track_length_inside(
                            cascade, convex_hull
                        )
                    )
                else:
                    # --------------------
                    # CC Tau interaction
                    # --------------------
                    entry, time, energy = get_tau_entry_info(
                        frame, tau, convex_hull
                    )
                    labels["p_entering"] = 1
                    labels["VertexX"] = entry.x
                    labels["VertexY"] = entry.y
                    labels["VertexZ"] = entry.z
                    labels["VertexTime"] = time
                    labels["EnergyVisible"] = energy
                    labels["Length"] = tau.length
                    labels["LengthInDetector"] = (
                        mu_utils.get_muon_track_length_inside(tau, convex_hull)
                    )
            else:
                # ------------------------------
                # NuMu CC Muon entering detector
                # ------------------------------
                entry, time, energy = mu_utils.get_muon_entry_info(
                    frame, muon, convex_hull, track_cache=track_cache
                )

                labels["p_entering"] = 1
                labels["p_entering_muon_single"] = 1
                labels["p_entering_muon_single_stopping"] = (
                    mu_utils.is_stopping_muon(muon, convex_hull)
                )
                labels["VertexX"] = entry.x
                labels["VertexY"] = entry.y
                labels["VertexZ"] = entry.z
                labels["VertexTime"] = time
                labels["EnergyVisible"] = energy
                labels["Length"] = muon.length
                labels["LengthInDetector"] = (
                    mu_utils.get_muon_track_length_inside(muon, convex_hull)
                )

        else:
            # --------------------
            # starting NuGen event
            # --------------------
            labels["VertexX"] = cascade.pos.x
            labels["VertexY"] = cascade.pos.y
            labels["VertexZ"] = cascade.pos.z
            labels["VertexTime"] = cascade.time
            labels["EnergyVisible"] = cascade.energy
            labels["Length"] = cascade.length
            labels["LengthInDetector"] = mu_utils.get_muon_track_length_inside(
                cascade, convex_hull
            )

            labels["p_starting"] = 1

            if frame["I3MCWeightDict"]["InteractionType"] == 1:
                # charged current
                labels["p_starting_cc"] = 1

                if cascade.type_string[:3] == "NuE":
                    # cc NuE
                    labels["p_starting_cc_e"] = 1

                elif cascade.type_string[:4] == "NuMu":
                    # cc NuMu
                    labels["p_starting_cc_mu"] = 1

                elif cascade.type_string[:5] == "NuTau":
                    # cc Tau
                    labels["p_starting_cc_tau"] = 1

                    nu_tau = get_interaction_neutrino(
                        frame,
                        primary,
                        convex_hull=None,
                        extend_boundary=extend_boundary,
                    )
                    tau = [
                        t
                        for t in mctree.get_daughters(nu_tau)
                        if t.type_string in ["TauMinus", "TauPlus"]
                    ]

                    assert len(tau) == 1, "Expected exactly 1 tau!"

                    mu = [
                        m
                        for m in mctree.get_daughters(tau[0])
                        if m.type_string in ["MuMinus", "MuPlus"]
                    ]

                    if len(mu) > 0:
                        # tau decays into muon: No Double bang signature!
                        labels["p_starting_cc_tau_muon_decay"] = 1
                    else:
                        # Double bang signature
                        labels["p_starting_cc_tau_double_bang"] = 1

                else:
                    raise ValueError(
                        "Unexpected type: {!r}".format(cascade.type_string)
                    )

            elif frame["I3MCWeightDict"]["InteractionType"] == 2:
                # neutral current (2)
                labels["p_starting_nc"] = 1

            elif frame["I3MCWeightDict"]["InteractionType"] == 3:
                # glashow resonance (3)
                labels["p_starting_glashow"] = 1

            else:
                #  GN -- Genie
                print(
                    "InteractionType: {!r}".format(
                        frame["I3MCWeightDict"]["InteractionType"]
                    )
                )

    elif (
        primary.type_string == "unknown" and primary.pdg_encoding == 0
    ) or mu_utils.is_muon(primary):

        if mu_utils.is_muon(primary):
            muon = primary
            # -----------------------------
            # muon primary: MuonGun dataset
            # -----------------------------
            if len(frame[mctree_name]) > 1:
                daughter = frame[mctree_name][1]
                if mu_utils.is_muon(daughter) and (
                    (daughter.id == primary.id)
                    and (daughter.dir == primary.dir)
                    and (daughter.pos == primary.pos)
                    and (daughter.energy == primary.energy)
                ):
                    muon = daughter

        else:
            daughters = frame[mctree_name].get_daughters(primary)
            muon = daughters[0]

            # Perform some safety checks to make sure that this is MuonGun
            assert (
                len(daughters) == 1
            ), "Expected only 1 daughter for MuonGun, but got {!r}".format(
                daughters
            )
            assert mu_utils.is_muon(muon), "Expected muon but got {!r}".format(
                muon
            )

        entry, time, energy = mu_utils.get_muon_entry_info(
            frame, muon, convex_hull, track_cache=track_cache
        )

        labels["p_entering"] = 1
        labels["p_entering_muon_single"] = 1
        labels["p_entering_muon_single_stopping"] = mu_utils.is_stopping_muon(
            muon, convex_hull
        )
        labels["VertexX"] = entry.x
        labels["VertexY"] = entry.y
        labels["VertexZ"] = entry.z
        labels["VertexTime"] = time
        labels["EnergyVisible"] = energy
        labels["Length"] = muon.length
        labels["LengthInDetector"] = mu_utils.get_muon_track_length_inside(
            muon, convex_hull
        )

        # The primary particle for MuonGun simulation can have NaN energy
        # replace this if necessary
        # (Note: technically PrimarEnergy is not known for MuonGun)
        if not np.isfinite(labels["PrimaryEnergy"]):
            labels["PrimaryEnergy"] = muon.energy

    else:
        # ---------------------------------------------
        # No neutrino or muon primary: Corsika dataset?
        # ---------------------------------------------
        muons = mu_utils.get_muons_inside(frame, convex_hull)
        if len(muons) == 0:
            muons = [m.particle for m in frame["MMCTrackList"]]

        time_max = None
        entry_max = None
        energy_max = float("-inf")
        for m in muons:
            if mu_utils.is_muon(m):
                entry, time, energy = mu_utils.get_muon_entry_info(
                    frame, m, convex_hull, track_cache=track_cache
                )

                if energy > energy_max:
                    time_max = time
                    entry_max = entry
                    energy_max = energy
                    muon = m

        labels["p_entering"] = 1
        if entry_max is None:
            labels["VertexX"] = float("nan")
            labels["VertexY"] = float("nan")
            labels["VertexZ"] = float("nan")
            labels["VertexTime"] = float("nan")
            labels["Length"] = float("nan")
            labels["LengthInDetector"] = float("nan")
        else:
            labels["VertexX"] = entry_max.x
            labels["VertexY"] = entry_max.y
            labels["VertexZ"] = entry_max.z
            labels["VertexTime"] = time_max
            labels["Length"] = muon.length
            labels["LengthInDetector"] = mu_utils.get_muon_track_length_inside(
                muon, convex_hull
            )
        labels["EnergyVisible"] = bundle_info["bundle_energy_at_entry"]

        if bundle_info["num_muons_at_entry_above_threshold"] > 0:
            bundle_key = "num_muons_at_entry_above_threshold"
        elif bundle_info["num_muons_at_entry"] > 0:
            bundle_key = "num_muons_at_entry"
        elif bundle_info["num_muons_at_cyl_above_threshold"] > 0:
            bundle_key = "num_muons_at_cyl_above_threshold"
        elif bundle_info["num_muons_at_cyl"] > 0:
            bundle_key = "num_muons_at_cyl"
        else:
            bundle_key = None
            print("Expected at least one muon!", frame[mctree_name])

        if bundle_key is not None:
            if bundle_info[bundle_key] == 1:
                labels["p_entering_muon_single"] = 1
                labels["p_entering_muon_single_stopping"] = (
                    mu_utils.is_stopping_muon(muon, convex_hull)
                )
            else:
                labels["p_entering_muon_bundle"] = 1

    # Check if event is track. Definition used here:
    #   Event is track if at least one muon exists in cylinder and the length
    #   of the shower/muon is greater than 20m
    if bundle_info["num_muons_at_cyl"] < 1:
        labels["p_is_track"] = 0
    else:
        if labels["Length"] > track_length_threshold:
            labels["p_is_track"] = 1
        else:
            labels["p_is_track"] = 0

    return labels


def get_cascade_parameters(
    frame,
    primary,
    convex_hull,
    extend_boundary=200,
    write_mc_cascade_to_frame=True,
    track_cache=None,
):
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
    write_mc_cascade_to_frame : bool
        If true, the cascade will be written to the I3Frame.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    I3MapStringDouble
        Cascade parameters of primary neutrino: x, y, z, t, azimuth, zenith, E
    """
    labels = dataclasses.I3MapStringDouble()

    cascade, e_em, e_hadron, e_track = get_cascade_of_primary_nu(
        frame,
        primary,
        convex_hull=None,
        extend_boundary=extend_boundary,
    )

    if cascade is None:
        # --------------------
        # not a starting event
        # --------------------
        muon = mu_utils.get_muon_of_inice_neutrino(frame)

        if muon is None:
            # --------------------
            # Cascade interaction outside of defined volume
            # --------------------
            cascade, e_em, e_hadron, e_track = get_cascade_of_primary_nu(
                frame,
                primary,
                convex_hull=None,
                extend_boundary=float("inf"),
                sanity_check=False,
            )

        else:
            # ------------------------------
            # NuMu CC Muon entering detector
            # ------------------------------
            # set cascade parameters to muon entry information
            entry, time, energy = mu_utils.get_muon_entry_info(
                frame, muon, convex_hull, track_cache=track_cache
            )

            length = mu_utils.get_muon_track_length_inside(muon, convex_hull)
            cascade = dataclasses.I3Particle()
            cascade.pos.x = entry.x
            cascade.pos.y = entry.y
            cascade.pos.z = entry.z
            cascade.time = time
            cascade.energy = energy
            cascade.dir = dataclasses.I3Direction(muon.dir)
            cascade.length = length
            e_em = 0.0
            e_hadron = 0.0
            e_track = energy

    if write_mc_cascade_to_frame:
        frame["MCCascade"] = cascade

    labels["cascade_x"] = cascade.pos.x
    labels["cascade_y"] = cascade.pos.y
    labels["cascade_z"] = cascade.pos.z
    labels["cascade_t"] = cascade.time
    labels["cascade_energy"] = cascade.energy
    labels["cascade_azimuth"] = cascade.dir.azimuth
    labels["cascade_zenith"] = cascade.dir.zenith
    labels["cascade_max_extension"] = cascade.length

    # compute fraction of energy for each component: EM, hadronic, track
    labels["energy_fraction_em"] = e_em / cascade.energy
    labels["energy_fraction_hadron"] = e_hadron / cascade.energy
    labels["energy_fraction_track"] = e_track / cascade.energy

    return labels
