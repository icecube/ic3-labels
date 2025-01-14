"""Helper functions for icecube specific labels.
"""

from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator
from icecube.icetray.i3logging import log_error, log_warn, log_info

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils.general import get_ids_of_particle_and_daughters
from ic3_labels.labels.utils.general import particle_is_inside
from ic3_labels.labels.utils.general import get_weighted_primary
from ic3_labels.labels.utils.cascade import get_cascade_em_equivalent


def is_muon(particle):
    """Checks if particle is a Muon: MuPlus or MuMinus

    Parameters
    ----------
    particle : I3Particle or I3MMCTrack
        Particle to be checked.

    Returns
    -------
    is_muon : bool
        True if particle is a muon, else false.
    """
    if particle is None:
        return False
    if isinstance(particle, simclasses.I3MMCTrack):
        particle = particle.particle
    return particle.pdg_encoding in (-13, 13)


def get_muon_time_at_distance(muon, distance):
    """Function to get the time of a muon at a certain
        distance from the muon vertex.
        Assumes speed = c

    Parameters
    ----------
    muon : I3Particle
        Muon.

    distance : float
        Distance.

    Returns
    -------
    time : float
        Time.
    """
    c = dataclasses.I3Constants.c  # m / nano s
    dtime = distance / c  # in ns
    return muon.time + dtime


def get_muon_time_at_position(muon, position):
    """Function to get the time of a muon at a certain
        position.

    Parameters
    ----------
    muon : I3Particle
        Muon.

    position : I3Position
        Position.

    Returns
    -------
    time : float
        Time.
        If position is before muon vertex or if position is
        not on line defined by the track, this will
        return nan.
        If position is along the track, but after the end
        point of the muon, this will return the time
        the muon would have been at that position.
    """
    distance = get_distance_along_track_to_point(muon.pos, muon.dir, position)
    if distance < 0 or np.isnan(distance):
        return float("nan")
    return get_muon_time_at_distance(muon, distance)


def get_muongun_track_cache(
    frame, mctree_name="I3MCTree", mmctracklist_name="MMCTrackList"
):
    """Get a cache of MuonGun tracks from the I3MCTree and MMCTrackList

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame which is utilized to retrieve the MMCTrackList
        and I3MCTree.
    mctree_name : str, optional
        The frame key of the I3MCTree.
    mmctracklist_name : str, optional
        The frame key of the MMCTrackList.

    Returns
    -------
    dict
        A dictionary with items {particle_id: MuonGun.Track}.
        None if no I3MCTree or MMCTrackList in frame.
    MuonGun.TrackList
        The list of MuonGun.Tracks
        None if no I3MCTree or MMCTrackList in frame.
    """
    if mctree_name not in frame or mmctracklist_name not in frame:
        log_info(
            "No I3MCTree or MMCTrackList in frame. "
            "Returning empty track cache."
        )
        return None, None

    track_list = MuonGun.Track.harvest(
        frame[mctree_name],
        frame[mmctracklist_name],
    )
    track_cache = {track.id: track for track in track_list}
    return track_cache, track_list


def get_muongun_track(frame, particle_id, track_cache=None):
    """Function to get the MuonGun track corresponding
        to the particle with the id particle_id

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    particle_id : I3ParticleID
        Id of the particle of which the MuonGun
        track should be retrieved from
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    track : MuonGun.Track
            Returns None if no corresponding track
            exists
    """
    if track_cache is None:
        for track in MuonGun.Track.harvest(
            frame["I3MCTree"], frame["MMCTrackList"]
        ):
            if track.id == particle_id:
                return track
        return None
    else:
        if particle_id in track_cache:
            return track_cache[particle_id]
        else:
            return None


def get_track_energy_wrapper(frame, track, distance):
    """Wrapper around MuonGun track.get_energy function

    In rare cases the sum of energy losses is larger than the checkpoint.
    (Why is this the case? I3MCTree reproduced incorrectly?)
    Catch and handle the exception here.
    While we are at it: print debug information, such that this issue can
    be fixed.

    Parameters
    ----------
    track : MuonGun.Track
        The MuonGun track.
    distance : float
        The distance along the track at which to obtain the energy.
    """

    try:
        energy = track.get_energy(distance)

    except RuntimeError as e:
        log_error("Caught exception: " + str(e))
        print("-----------")
        print("Debug info:")
        print("-----------")
        print("Track:")
        print(track)
        print("Distance:", distance)
        print("I3EventHeader:")
        print(frame["I3EventHeader"])
        print("-----------")

        energy = float("nan")
        log_error("Setting energy to NaN and proceeding!")

    return energy


def get_muon_energy_at_position(frame, muon, position, track_cache=None):
    """Function to get the energy of a muon at a certain position.

    Parameters
    ----------
    frame : I3Frame
        Current frame.
    muon : I3Particle
        Muon.
    position : I3Position
        Position.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    energy : float
        Energy.
        If position is before muon vertex or if position is
        not on line defined by the track, this will
        return nan.
        If no corresponding MuonGun.Track can be found to the
        muon, then this will return nan.
        If position is along the track, but after the end
        point of the muon, this will return 0
    """
    track = get_muongun_track(frame, muon.id, track_cache=track_cache)
    if track is None:
        # no track exists [BUG?]
        # That means that muon is not in the frame['MMCTrackList']
        # or that it is not correctly harvested from MuonGun
        # Assuming that this is only the case, when the muon
        # is either outside of the inice-volume or that the muon
        # is too low energetic to be listed in the frame['MMCTrackList']
        # Need to fix this ----------------BUG

        # if position is close to the vertex, we can just assume
        # that the energy at position is the initial energy
        distance = np.linalg.norm(muon.pos - position)
        if distance < 60:
            # accept if less than 60m difference
            return muon.energy
        else:
            if muon.energy < 20 or muon.length < distance:
                return 0.0
        return float("nan")
    distance = get_distance_along_track_to_point(muon.pos, muon.dir, position)
    if distance < 0 or np.isnan(distance):
        return float("nan")
    return get_track_energy_wrapper(frame, track, distance)
    # return track.get_energy(distance)


def get_muon_energy_at_distance(frame, muon, distance, track_cache=None):
    """Function to get the energy of a muon at a certain
        distance from the muon vertex

    Parameters
    ----------
    frame : I3Frame
        Current frame.
    muon : I3Particle
        Muon.
    distance : float
        Distance.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    energy : float
        Energy.
    """
    track = get_muongun_track(frame, muon.id, track_cache=track_cache)
    if track is None:
        # no track exists [BUG?]
        # That means that muon is not in the frame['MMCTrackList']
        # or that it is not correctly harvested from MuonGun
        # Assuming that this is only the case, when the muon
        # is either outside of the inice-volume or that the muon
        # is too low energetic to be listed in the frame['MMCTrackList']
        # Need to fix this ----------------BUG

        # if position is close to the vertex, we can just assume
        # that the energy at position is the initial energy
        if distance < 60:
            # accept if less than 60m difference
            return muon.energy
        else:
            if muon.energy < 20 or muon.length < distance:
                return 0.0
        return float("nan")
    return get_track_energy_wrapper(frame, track, distance)
    # return track.get_energy(distance)


def bin_muon_energy_losses_along_track(
    frame, muon, bin_edges, track_cache=None
):
    """Bin energy losses of Muon along the track

    Parameters
    ----------
    frame : I3Frame
        Current I3 frame.
    muon : I3Particle
        The track I3Particle for which to bin the energy losses.
    bin_edges : array_like
        The bin edges in which the energy osses will be accumulated.
        These are defined as distances along the track relative to the
        particle vertex.
        The binned energy losses will have a length of len(bin_edges) - 1.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    array_like
        The binned energy losses.
        Shape: [len(bin_edges) - 1]
    """
    energies_at_edges = []
    for bin_edge in bin_edges:
        energies_at_edges.append(
            get_muon_energy_at_distance(
                frame, muon, bin_edge, track_cache=track_cache
            )
        )

    energy_lost = np.diff(energies_at_edges) * -1

    # Energy before track vertex may be zero, therefore the diffs can contain
    # negative numbers. These will now be set to zero
    energy_lost[energy_lost < 0] = 0.0

    return energy_lost


def get_binned_energy_losses_in_cylinder(
    frame,
    muon,
    bin_width=15.0,
    num_bins=100,
    cylinder_height=1000.0,
    cylinder_radius=600.0,
    track_cache=None,
):
    """Bin energy losses along inf track in axial cylinder

    The energy losses along the infinite track are binned in a total of
    `num_bins` bins, which have a spacing of `bin_width`. The first bin
    edge is computed as the entry point of the infinite track into the defined
    cylinder. If the infinite track does not intersect the cylinder, the
    closest approach point minus half of the binned track length will be used
    as the first bin edge.

    Parameters
    ----------
    frame : I3Frame
        Current I3 frame.
    muon : I3Particle
        The track I3Particle for which to bin the energy losses.
    bin_width : float, optional
        Defines the width of bins [in meters] along the track.
        Energy losses are binned along the track in bins of this width.
    num_bins : int, optional
        The number of bins to create. The total binned track length depends
        on the chosen number of bins and the bin width `bin_width`.
    cylinder_height : float, optional
        Height of the cylinder.
        The cylinder is aligned to the z-axis and centered at (0, 0, 0).
    cylinder_radius : float, optional
        Radius of the cylinder.
        The cylinder is aligned to the z-axis and centered at (0, 0, 0).
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    array_like
        The binned energy losses.
        Shape: [num_bins]
    """

    # compute intersection with cylinder
    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_axial_cylinder_intersections(
        v_pos, v_dir, height=cylinder_height, radius=cylinder_radius
    )

    if len(intersection_ts) > 0:
        bin_start = intersection_ts[0]

    else:
        # The infinite track does not intersect the axial cylinder
        # Compute the closest approach distance to the detector center
        dist_closest = I3Calculator.closest_approach_distance(
            muon, dataclasses.I3Position(0.0, 0.0, 0.0)
        )
        bin_start = dist_closest - 0.5 * (bin_width * max(0, num_bins - 1))

    # compute bin edges
    bin_edges = []
    for i in range(num_bins + 1):
        bin_edges.append(bin_start + i * bin_width)

    # Now bin energy losses in these bins
    binnned_energy_losses = bin_muon_energy_losses_along_track(
        frame=frame, muon=muon, bin_edges=bin_edges, track_cache=track_cache
    )

    return binnned_energy_losses


def get_binned_energy_losses_in_cube(
    frame,
    muon,
    bin_width=15.0,
    boundary=600.0,
    return_bin_centers=False,
    track_cache=None,
):
    """Bin energy losses along inf track in a cube around IceCubes center.

    The energy losses along the infinite track are binned with a spacing of
    `bin_width`. The first bin edge is computed as the entry point of the
    infinite track into the defined cube.
    If the infinite track does not intersect the cube, the
    closest approach point minus half of the binned track length will be used
    as the first bin edge.

    Parameters
    ----------
    frame : I3Frame
        Current I3 frame.
    muon : I3Particle
        The track I3Particle for which to bin the energy losses.
    bin_width : float, optional
        Defines the width of bins [in meters] along the track.
        Energy losses are binned along the track in bins of this width.
    boundary: float, optional
        Defines half the edge length of a cube positioned at IceCube's
        center.
    return_bin_centers: bool, optional
        Switch to also return the bin centers.

    Returns
    -------
    array_like
        The binned energy losses.
        Shape: [num_bins]

    array_like: optional
        Bin center positions for the binned energy losses.
        Shape: [num_bins]
    """

    # compute intersection with cube
    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_cube_intersections(
        v_pos, v_dir, boundary=boundary
    )

    if len(intersection_ts) > 0:
        entry_t = intersection_ts[0]
        exit_t = intersection_ts[1]
    else:
        # The infinite track does not intersect cube
        # Compute the closest approach distance to the detector center
        dist_closest = I3Calculator.closest_approach_distance(
            muon, dataclasses.I3Position(0.0, 0.0, 0.0)
        )
        entry_t = dist_closest - 0.5 * (
            bin_width * max(0, boundary // bin_width)
        )
        exit_t = dist_closest + 0.5 * (
            bin_width * max(0, boundary // bin_width)
        )

    # Shift the entry point to be an integer number of the bin width
    # away from v_pos
    bin_start = np.round(entry_t / bin_width) * bin_width
    # compute bin edges
    bin_edges = np.arange(bin_start, exit_t, bin_width)

    # Now bin energy losses in these bins
    binned_energy_losses = bin_muon_energy_losses_along_track(
        frame=frame, muon=muon, bin_edges=bin_edges, track_cache=track_cache
    )

    if not return_bin_centers:
        return binned_energy_losses
    else:
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        bin_center_pos = [
            np.array(v_pos) + bc * np.array(v_dir) for bc in bin_centers
        ]
        return binned_energy_losses, bin_center_pos


def get_inf_muon_binned_energy_losses(
    frame,
    convex_hull,
    muon,
    bin_width=10,
    extend_boundary=150,
    compute_em_equivalent=False,
    include_under_over_flow=False,
):
    """Function to get binned energy losses along an infinite track.
    The direction and vertex of the given muon is used to create an
    infinite track. This infinite track is then binned in bins of width
    bin_width along the track. The first bin will start at the point where
    the infinite track intersects the extended convex hull. The convex hull
    is defined by convex_hull and can optionally be extended by
    extend_boundary meters.
    The I3MCTree is traversed and the energy losses of the muon are
    accumulated in the corresponding bins along the infinite track.

    If the muon does not hit the convex hull (without extension)
    an empty list is returned.

    Parameters
    ----------
    frame : current frame
        needed to retrieve I3MCTree
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume
    muon : I3Particle
        Muon
    bin_width : float
        defines width of bins [in meters]
        Energy losses in I3MCtree are binned along the
        track in bins of this width
    extend_boundary : float
        Extend boundary of convex_hull by this distance [in meters].
        The first bin will be at convex_hull + extend_boundary
    compute_em_equivalent : bool
        If True, the energy losses are converted to EM-equivalent energy.
    include_under_over_flow : bool
        If True, an underflow and overflow bin is added for energy
        losses outside of convex_hull + extend_boundary

    Returns
    -------
    binnned_energy_losses : list of float
        Returns a list of energy losses for each bin


    Raises
    ------
    ValueError
        Description
    """

    if muon.pdg_encoding not in (13, -13):  # CC [Muon +/-]
        raise ValueError("Expected muon but got:", muon)

    intersection_ts = get_muon_convex_hull_intersections(muon, convex_hull)

    # muon didn't hit convex_hull
    if len(intersection_ts) == 0:
        return []

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)

    bin_start = muon.pos + min_ts * muon.dir - extend_boundary * muon.dir
    bin_end = muon.pos + max_ts * muon.dir + extend_boundary * muon.dir

    total_length = (bin_end - bin_start).magnitude

    bin_edges = np.arange(0, total_length + bin_width, bin_width)

    # include overflow bin
    bin_edges = np.append(bin_edges, float("inf"))

    # get distance and energy of each loss
    distances = []
    energies = []
    for daughter in frame["I3MCTree"].get_daughters(muon):

        if is_muon(daughter):
            # this is likely a track segment update: skip this
            continue

        # compute EM-equivalent energy of daughter
        if compute_em_equivalent:
            energy = get_cascade_em_equivalent(frame["I3MCTree"], daughter)[0]
        else:
            energy = daughter.energy

        distances.append((daughter.pos - bin_start).magnitude)
        energies.append(energy)

    # bin energy losses in bins along track
    binnned_energy_losses, _ = np.histogram(
        distances, weights=energies, bins=bin_edges
    )

    if not include_under_over_flow:
        # remove under and over flow bin
        binnned_energy_losses = binnned_energy_losses[1:-1]

    return binnned_energy_losses


def get_muon_entry_info(frame, muon, convex_hull, track_cache=None):
    """Get muon information for point of entry, or closest approach point,
    if muon does not enter the volume defined by the convex_hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    muon : I3Particle
        Muon I3Particle for which to get the entry information.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    I3Position, double, double
        Entry Point (or closest approach point)
        Time of entry point (or closest approach point)
        Energy at entry point (or closest approach point)
        Warning: If 'I3MCTree' does not exist in frame, this
                 will instead return the muon energy
    """
    entry = get_muon_initial_point_inside(muon, convex_hull)
    if entry is None:
        # get closest approach point as entry approximation
        entry = get_muon_closest_approach_to_center(frame, muon)
    time = get_muon_time_at_position(muon, entry)
    energy = get_muon_energy_at_position(
        frame, muon, entry, track_cache=track_cache
    )

    return entry, time, energy


def get_muon(
    frame,
    primary,
    convex_hull,
    mctree_name="I3MCTree",
    track_cache=None,
):
    """Get muon from MCPrimary.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        The primary I3Particle for which to find the muon.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
    mctree_name : str, optional
        The name of the I3MCTree.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    I3Particle
        The muon from the primary particle or None, if no muon exists.
    """
    # NuGen dataset
    if primary.is_neutrino:
        muon = get_muon_of_inice_neutrino(frame)

    # MuonGun dataset
    elif (
        primary.type_string == "unknown" and primary.pdg_encoding == 0
    ) or is_muon(primary):

        if is_muon(primary):
            muon = primary
            if len(frame[mctree_name]) > 1:
                daughter = frame[mctree_name][1]
                if is_muon(daughter) and (
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
            assert is_muon(muon), "Expected muon but got {!r}".format(muon)

    # No neutrino or muon primary: Corsika dataset?
    else:
        muons = get_muons_inside(frame, convex_hull)
        if len(muons) == 0:
            muons = [m.particle for m in frame["MMCTrackList"]]

        energy_max = float("-inf")
        for m in muons:
            if is_muon(m):
                entry, time, energy = get_muon_entry_info(
                    frame, m, convex_hull, track_cache=track_cache
                )
                if energy > energy_max:
                    energy_max = energy
                    muon = m

    return muon


def get_muon_scattering_info(
    frame,
    convex_hull,
    primary,
    min_length=1000,
    min_length_before=400,
    min_length_after=400,
    min_muon_entry_energy=10000,
    min_rel_loss_energy=0.5,
    track_cache=None,
):
    """Function to get labels that can be used to detect muon scattering.

    The labels returned include:

        'length':
            Length in convex hull
        'length_before':
            Length in convex hull before biggest energy loss
        'length_after':
            Length in convex hull after biggest energy loss
        'muon_energy':
            Primary muon energy
        'muon_energy_at_entry':
            Energy of muon as it entters the detector
        'max_muon_energy_loss':
            Maximum energy loss in detector.
        'p_scattering_candidate':
            Muons that pass cuts

    Parameters
    ----------
    frame : current frame
        needed to retrieve I3MCTree
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume
    primary : TYPE
        Description
    min_length : int, optional
        Description
    min_length_before : int, optional
        Description
    min_length_after : int, optional
        Description
    min_muon_entry_energy : int, optional
        Description
    min_rel_loss_energy : float, optional
        Description
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    labels : I3MapStringDouble
        The computed muon scattering labels.

    Raises
    ------
    ValueError
        Description

    Deleted Parameters
    ------------------
    muon : I3Particle
        Muon
    """

    # fill in default values
    labels = dataclasses.I3MapStringDouble()
    labels["length"] = 0.0
    labels["length_before"] = 0.0
    labels["length_after"] = 0.0
    labels["muon_zenith"] = 0.0
    labels["muon_azimuth"] = 0.0
    labels["muon_energy"] = 0.0
    labels["muon_entry_energy"] = 0.0
    labels["muon_entry_x"] = 0.0
    labels["muon_entry_y"] = 0.0
    labels["muon_entry_z"] = 0.0
    labels["muon_entry_time"] = 0.0
    labels["muon_exit_energy"] = 0.0
    labels["muon_exit_x"] = 0.0
    labels["muon_exit_y"] = 0.0
    labels["muon_exit_z"] = 0.0
    labels["muon_exit_time"] = 0.0
    labels["muon_loss_energy"] = 0.0
    labels["muon_loss_x"] = 0.0
    labels["muon_loss_y"] = 0.0
    labels["muon_loss_z"] = 0.0
    labels["muon_loss_time"] = 0.0
    labels["rel_muon_loss_energy"] = 0.0
    labels["p_scattering_candidate"] = 0.0

    muon = get_muon(frame, primary, convex_hull, track_cache=track_cache)
    if muon is None:
        return labels

    if muon.pdg_encoding not in (13, -13):  # CC [Muon +/-]
        raise ValueError("Expected muon but got:", muon)

    entry = get_muon_initial_point_inside(muon, convex_hull)

    # muon didn't hit convex_hull
    if entry is None:
        labels["muon_zenith"] = muon.dir.zenith
        labels["muon_azimuth"] = muon.dir.azimuth
        labels["muon_energy"] = muon.energy
        return labels

    exit = get_muon_exit_point(muon, convex_hull)
    total_length = (exit - entry).magnitude
    entry_energy = get_muon_energy_at_position(
        frame, muon, entry, track_cache=track_cache
    )
    exit_energy = get_muon_energy_at_position(
        frame, muon, exit, track_cache=track_cache
    )
    entry_time = get_muon_time_at_position(muon, entry)
    exit_time = get_muon_time_at_position(muon, exit)

    # get max energy loss inside
    max_energy_loss = None
    max_energy = -float("inf")
    for daughter in frame["I3MCTree"].get_daughters(muon):

        if is_muon(daughter):
            # these are probably muon segment updates and not actual losses
            continue

        if daughter.time > entry_time and daughter.time < exit_time:
            if daughter.energy > max_energy:
                max_energy = daughter.energy
                max_energy_loss = daughter

    if max_energy_loss is None:
        max_energy_loss = dataclasses.I3Particle()
        length_before = 0.0
        length_after = 0.0
        rel_loss_energy = 0.0
    else:

        # calculate labels
        length_before = (max_energy_loss.pos - entry).magnitude
        length_after = (exit - max_energy_loss.pos).magnitude
        rel_loss_energy = max_energy / entry_energy

    if (
        total_length >= min_length
        and length_before >= min_length_before
        and length_after >= min_length_after
        and rel_loss_energy >= min_rel_loss_energy
        and entry_energy > min_muon_entry_energy
    ):
        p_scattering_candidate = 1.0
    else:
        p_scattering_candidate = 0.0

    # fill in labels
    labels["length"] = total_length
    labels["length_before"] = length_before
    labels["length_after"] = length_after
    labels["muon_zenith"] = muon.dir.zenith
    labels["muon_azimuth"] = muon.dir.azimuth
    labels["muon_energy"] = muon.energy
    labels["muon_entry_energy"] = entry_energy
    labels["muon_entry_x"] = entry.x
    labels["muon_entry_y"] = entry.y
    labels["muon_entry_z"] = entry.z
    labels["muon_entry_time"] = entry_time
    labels["muon_exit_energy"] = exit_energy
    labels["muon_exit_x"] = exit.x
    labels["muon_exit_y"] = exit.y
    labels["muon_exit_z"] = exit.z
    labels["muon_exit_time"] = exit_time
    labels["muon_loss_energy"] = max_energy_loss.energy
    labels["muon_loss_x"] = max_energy_loss.pos.x
    labels["muon_loss_y"] = max_energy_loss.pos.y
    labels["muon_loss_z"] = max_energy_loss.pos.z
    labels["muon_loss_time"] = max_energy_loss.time
    labels["rel_muon_loss_energy"] = rel_loss_energy
    labels["p_scattering_candidate"] = p_scattering_candidate

    return labels


def get_muon_energy_deposited(frame, convex_hull, muon, track_cache=None):
    """Function to get the total energy a muon deposited in the
    volume defined by the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume
    muon : I3Particle
        muon.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    energy : float
        Deposited Energy.
    """
    intersection_ts = get_muon_convex_hull_intersections(muon, convex_hull)

    # muon didn't hit convex_hull
    if len(intersection_ts) == 0:
        return 0.0

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting track
        return muon.energy - get_muon_energy_at_distance(
            frame, muon, max_ts, track_cache=track_cache
        )
    if max_ts < 0:
        # muon created after the convex hull
        return 0.0

    energy_min_ts = get_muon_energy_at_distance(
        frame,
        muon,
        min_ts,
        track_cache=track_cache,
    )
    energy_max_ts = get_muon_energy_at_distance(
        frame,
        muon,
        max_ts,
        track_cache=track_cache,
    )
    return energy_min_ts - energy_max_ts


def get_muon_initial_point_inside(muon, convex_hull):
    """Get initial point of the muon inside
        the convex hull. This is either the
        vertex for a starting muon or the
        first intersection of the muon
        and the convex hull.

    Parameters
    ----------
    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    intial_point : I3Position
        Returns None if muon doesn't hit
        convex hull.
    """
    intersection_ts = get_muon_convex_hull_intersections(muon, convex_hull)

    # muon didn't hit convex_hull
    if len(intersection_ts) == 0:
        return None

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting track
        return muon.pos
    if max_ts < 0:
        # muon created after the convex hull
        return None
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return None
    return muon.pos + min_ts * muon.dir


def is_stopping_muon(muon, convex_hull):
    """Check if the the muon stops within the convex hull.

    If the muon does not hit the convex hull this will return False.
    This will return True for a starting muon that also stops within the
    convex hull.

    Parameters
    ----------
    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    bool
        Returns true if muon
    """
    intersection_ts = get_muon_convex_hull_intersections(muon, convex_hull)

    # muon didn't hit convex_hull
    if len(intersection_ts) == 0:
        return False

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return False
    if max_ts < 0:
        # muon created after the convex hull
        return False
    if max_ts > muon.length + 1e-8:
        # stopping track
        return True
    else:
        # muon exits detector
        return False


def get_muon_convex_hull_intersections(track, convex_hull):
    """Get the intersections of an infinite track.

    Parameters
    ----------
    track : I3Particle
        The infinite track for which to compute the intersections with the
        convex hull.
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    array_like
        The two intersections points
    """
    intersection_ts = geometry.get_intersections(
        convex_hull, track.pos, track.dir
    )

    if len(intersection_ts) == 1:
        # vertex is possible exactly on edge of convex hull
        # move vertex slightly by eps
        eps = 1e-4
        muon_pos_shifted = track.pos + eps * track.dir
        intersection_ts = geometry.get_intersections(
            convex_hull, muon_pos_shifted, track.dir
        )

    # track hit convex_hull:
    #   Expecting zero or two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) in [
        0,
        2,
    ], "Expected exactly 0 or 2 intersections: {}".format(intersection_ts)

    return intersection_ts


def get_muon_exit_point(muon, convex_hull):
    """Get point of the muon when it exits
        the convex hull. This is either the
        stopping point for a stopping muon or the
        second intersection of the muon
        and the convex hull.

    Parameters
    ----------
    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    intial_point : I3Position
        Returns None if muon doesn't hit
        convex hull.
    """
    intersection_ts = get_muon_convex_hull_intersections(muon, convex_hull)

    # muon didn't hit convex_hull
    if len(intersection_ts) == 0:
        return None

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return None
    if max_ts < 0:
        # muon created after the convex hull
        return None
    if min_ts < muon.length + 1e-8 and max_ts > muon.length + 1e-8:
        # stopping track
        return muon.pos + muon.length * muon.dir

    return muon.pos + max_ts * muon.dir


def get_mmc_particle(frame, muon):
    """Get corresponding I3MMCTrack
        object to the I3Particle muon

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    Returns
    -------
    muon : I3MMCTrack
        I3MMCTrack object of the I3Particle muon
        returns None if no corresponding I3MMCTrack
        object can be found
    """
    for p in frame["MMCTrackList"]:
        if p.particle.id == muon.id:
            return p
    return None


def get_distance_along_track_to_point(vertex, direction, point):
    """Get (signed) distance along a track (defined by position
        and direction) to a point. Negative distance means the
        point is before the vertex.
        Assumes that point is on the infinite track
        (within ~0.81 degree [rtol=1e-4]).

    Parameters
    ----------
    vertex : I3Position
        Vertex (starting point) of the track

    direction : I3Direction
        Direction of the track

    point : I3Position
        Point of which to calculate distance to

    Returns
    -------
    distance : float
        Distance along track to get to point starting
        from the vertex. Negative value indicates
        the point is before the vertex.
        Returns nan if point is not on track (within ~0.81 degree [rtol=1e-4]).
    """
    distanceX = (point.x - vertex.x) / direction.x
    distanceY = (point.y - vertex.y) / direction.y
    if not np.allclose(distanceX, distanceY, rtol=1e-4):
        return float("nan")
    distanceZ = (point.z - vertex.z) / direction.z
    if not np.allclose(distanceX, distanceZ, rtol=1e-4):
        return float("nan")
    else:
        return distanceX


def get_particle_closest_approach_to_position(particle, position):
    """Get closest approach to an I3Position position
        given a particle.

    Parameters
    ----------
    particle : I3Particle.
             Particle of which to compute the
             closest approach to position

    position : I3Position.
             Position to which the closest approach
             of the particle is to be calculated.

    Returns
    -------
    closest_position : I3Position
        I3Position of the point on the track
        that is closest to the position
    """
    closest_position = I3Calculator.closest_approach_position(
        particle, position
    )
    distance_to_position = get_distance_along_track_to_point(
        particle.pos, particle.dir, closest_position
    )
    if distance_to_position < 0:
        # closest_position is before vertex, so set
        # closest_position to vertex
        closest_position = particle.pos
    elif distance_to_position > particle.length:
        # closest_position is after end point of track,
        # so set closest_position to the endpoint
        closest_position = particle.pos + particle.dir * particle.length

    return closest_position


def get_mmc_closest_approach_to_center(mmc_track):
    """Get closest approach to center (0,0,0)
        of a an I3MMCTrack. Uses MMCTrackList center
        position, but checks, if particle is still on
        track. Assumes that the MMCTrackList center
        is somewhere on the line of the track.

    Parameters
    ----------
    mmc_track : I3MMCTrack

    Returns
    -------
    center : I3Position
        I3Position of the point on the track
        that is closest to the center (0,0,0)
    """
    center = dataclasses.I3Position(mmc_track.xc, mmc_track.yc, mmc_track.zc)
    distance_to_center = get_distance_along_track_to_point(
        mmc_track.particle.pos, mmc_track.particle.dir, center
    )
    if distance_to_center < 0:
        # mmc center is before vertex, so set center to vertex
        center = mmc_track.particle.pos
    elif distance_to_center > mmc_track.particle.length:
        # mmc center is after end point of track,
        # so set center to the endpoint
        center = (
            mmc_track.particle.pos
            + mmc_track.particle.dir * mmc_track.particle.length
        )

    return center


def get_muon_closest_approach_to_center(frame, muon):
    """Get closest approach to center (0,0,0)
        of a muon. Uses MMCTrackList center position,
        but checks, if particle is still on track.
        Assumes that the MMCTrackList center
        is somewhere on the line of the track.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    Returns
    -------
    center : I3Position
        I3Position of the point on the track
        that is closest to the center (0,0,0)
    """
    if not is_muon(muon):
        raise ValueError("Particle:\n{}\nis not a muon.".format(muon))

    if "MMCTrackList" in frame:
        mmc_muon = get_mmc_particle(frame, muon)
    else:
        mmc_muon = None

    if mmc_muon is None:
        # no mmc_muon exists [BUG?]
        # That means that muon is not in the frame['MMCTrackList']
        # Assuming that this is only the case, when the muon
        # is either outside of the inice-volume or that the muon
        # is too low energetic to be listed in the frame['MMCTrackList']

        return get_particle_closest_approach_to_position(
            muon, dataclasses.I3Position(0, 0, 0)
        )

    assert is_muon(mmc_muon), "mmc_muon should be a muon"

    return get_mmc_closest_approach_to_center(mmc_muon)


def is_mmc_particle_inside(mmc_particle, convex_hull):
    """Find out if mmc particle is inside volume
        defined by the convex hull

    Parameters
    ----------
    mmc_particle : I3MMCTrack

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    isInside : bool
        True if mmc muon is inside covex hull
        Returns False, if mmc_particle doesn't exist.
    """
    if mmc_particle is None:
        return False
    return particle_is_inside(
        particle=mmc_particle.particle, convex_hull=convex_hull
    )


def is_muon_inside(muon, convex_hull):
    """Find out if muon is inside volume defined by the convex hull

    Parameters
    ----------
    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    muon : I3Particle
        Muon.
    """
    if not is_muon(muon):
        raise ValueError("Particle:\n{}\nis not a muon.".format(muon))

    return particle_is_inside(particle=muon, convex_hull=convex_hull)


def get_mmc_particles_inside(frame, convex_hull):
    """Get mmc particles entering the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    particles : list of I3MMCTrack
        Particle mmcTracks that are inside
    """
    mmc_particles_inside = [
        m
        for m in frame["MMCTrackList"]
        if is_mmc_particle_inside(m, convex_hull)
    ]
    return mmc_particles_inside


def get_muons_inside(frame, convex_hull):
    """Get muons inside the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    muons : list of I3Particle
        Muons.
    """
    muons_inside = [
        m.particle
        for m in frame["MMCTrackList"]
        if is_mmc_particle_inside(m, convex_hull) and is_muon(m.particle)
    ]
    return muons_inside


def get_most_energetic_muon_inside(
    frame, convex_hull, muons_inside=None, track_cache=None
):
    """Get most energetic Muon that is within
    the convex hull. To decide which muon is
    the most energetic, the energy at the initial
    point in the volume is compared. This is either
    the muon vertex or the entry point.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muons_inside : list of I3Particle
        Muons inside the convex hull
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    most_energetic_muon : I3Particle
        Returns most energetic muon inside convex hull.
        Returns None, if no muon exists in convex hull.
    """
    if muons_inside is None:
        muons_inside = get_muons_inside(frame, convex_hull)

    most_energetic_muon = None
    most_energetic_muon_energy = 0

    for m in muons_inside:
        initial_point = get_muon_initial_point_inside(m, convex_hull)
        intial_energy = get_muon_energy_at_position(
            frame, m, initial_point, track_cache=track_cache
        )
        if intial_energy > most_energetic_muon_energy:
            most_energetic_muon = m
            most_energetic_muon_energy = intial_energy

    return most_energetic_muon


def get_highest_deposit_muon_inside(
    frame, convex_hull, muons_inside=None, track_cache=None
):
    """Get Muon with the most deposited energy
        that is inside or hits the convex_hull

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume
    muons_inside : list of I3Particle
        Muons inside the convex hull
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    muon : I3Particle
        Muon.
    """
    if muons_inside is None:
        muons_inside = get_muons_inside(frame, convex_hull)

    highest_deposit_muon = None
    highest_deposit = 0

    for m in muons_inside:
        deposit = get_muon_energy_deposited(
            frame, convex_hull, m, track_cache=track_cache
        )
        if deposit > highest_deposit:
            highest_deposit_muon = m
            highest_deposit = deposit

    return highest_deposit_muon


def get_most_visible_muon_inside(
    frame,
    convex_hull,
    pulse_map_string="InIcePulses",
    mcpe_series_map_name="I3MCPESeriesMap",
    max_time_dif=100,
    method="noOfPulses",
):
    """Get Muon with the most deposited charge
        inside the detector, e.g. the most visible

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for

    mcpe_series_map_name : key of mcpe series map in frame

    method : string 'charge','noOfPulses'
        'charge' : select muon that deposits the
                    highest sum of charges
        'noOfPulses' : select muon that has the
                    most no of pulses

    Returns
    -------
    muon : I3Particle
        Muon.
    """

    # get all muons
    muons = [m.particle for m in frame["MMCTrackList"] if is_muon(m.particle)]
    # muons = frame['I3MCTree'].get_filter(
    #                 lambda p: p.pdg_encoding in [13, -13])

    most_visible_muon = None
    if muons:
        if len(muons) == 1:
            # stop here if only one muon is inside
            return muons[0]
        ids = [m.id for m in muons]

        assert (0, 0) not in ids, "Muon with id (0,0) should not exist"

        # I3ParticleID can't be used as a dict key in older icecube software versions
        # [works in: Version combo.trunk     r152630 (with python 2.7.6)]
        # possible_ids = {m.id : set(get_ids_of_particle_and_daughters(frame,m,[]))
        #                                                          for m in muons}
        # # possible_ids = {(m.id.majorID,m.id.minorID) :
        # #                 { (i.majorID,i.minorID) for i in get_ids_of_particle_and_daughters(frame,m,[]) }
        # #                                         for m in muons}

        # create a dictionary that holds all daughter ids of each muon
        possible_ids = {}
        for m in muons:
            # get a set of daughter ids for muon m
            temp_id_set = {
                (i.majorID, i.minorID)
                for i in get_ids_of_particle_and_daughters(frame, m, [])
            }

            # fill dictionary
            possible_ids[(m.id.majorID, m.id.minorID)] = temp_id_set

            # sanity check
            assert (
                0,
                0,
            ) not in temp_id_set, (
                "Daughter particle with id (0,0) should not exist"
            )

        counter = {(i.majorID, i.minorID): 0.0 for i in ids}

        # get pulses defined by pulse_map_string
        in_ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame,
            pulse_map_string,
        )

        # get candidate keys
        valid_keys = set(frame[mcpe_series_map_name].keys())

        # find all pulses resulting from particle or daughters of particle
        shared_keys = {
            key for key in in_ice_pulses.keys() if key in valid_keys
        }

        for key in shared_keys:
            # mc_pulses = [ p for p in frame[mcpe_series_map_name][key]
            #                      if p.ID in ids_set]
            mc_pulses = frame[mcpe_series_map_name][key]
            pulses = in_ice_pulses[key]
            if mc_pulses:
                # speed things up:
                # pulses are sorted in time. Therefore we
                # can start from the last match
                last_index = 0
                for pulse in pulses:
                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, p in enumerate(mc_pulses[last_index:]):
                        if abs(pulse.time - p.time) < max_time_dif:
                            last_index = last_index + i
                            for ID in ids:
                                if (
                                    p.ID.majorID,
                                    p.ID.minorID,
                                ) in possible_ids[(ID.majorID, ID.minorID)]:
                                    if method == "charge":
                                        counter[
                                            (ID.majorID, ID.minorID)
                                        ] += pulse.charge
                                    elif method == "noOfPulses":
                                        counter[(ID.majorID, ID.minorID)] += 1
                            break

        most_visible_muon = muons[0]
        for k, i in enumerate(ids):
            if (
                counter[(i.majorID, i.minorID)]
                > counter[
                    (
                        most_visible_muon.id.majorID,
                        most_visible_muon.id.minorID,
                    )
                ]
            ):
                most_visible_muon = muons[k]
    return most_visible_muon


def get_muon_of_inice_neutrino(frame, muongun_primary_neutrino_id=None):
    """Get the muon daughter of the first in ice neutrino in the I3MCTree.

    Optionally a primary particle id can be passed for MuonGun simulation.
    Returns None if no muons can be found.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    muon : I3Particle
        Muon.
        Returns None if no muon daughter can be found.
    """
    nu_in_ice = None
    for p in frame["I3MCTree"]:
        if (
            p.is_neutrino and p.location_type_string == "InIce"
        ) or p.id == muongun_primary_neutrino_id:
            nu_in_ice = p
            break

    if nu_in_ice is None:
        log_warn('Could not find an "InIce" neutrino. Returning None')
        return None

    daughters = frame["I3MCTree"].get_daughters(nu_in_ice)
    muons = [p for p in daughters if p.pdg_encoding in (13, -13)]

    if muons:
        assert len(muons) == 1, "Found more or less than one expected muon."
        return muons[0]
    else:
        return None


def get_next_muon_daughter_of_nu(
    frame, particle, muongun_primary_neutrino_id=None
):
    """Get the next muon daughter of a muon-neutrino.
    This will return None for any neutrinos other than muon-neutrinos.
    Goes along I3MCTree to find the first muon daughter of the muon-neutrino.
    Returns None if none can be found.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    particle : I3Particle

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    muon : I3Particle
        Muon.
        Returns None if no muon daughter
        can be found.
    """
    if (
        particle.pdg_encoding == 14
        or particle.pdg_encoding == -14
        or particle.id == muongun_primary_neutrino_id
    ):  # nu # MuonGunFix
        daughters = frame["I3MCTree"].get_daughters(particle)
        if len(daughters) == 0:
            return None
        codes = [p.pdg_encoding for p in daughters]
        if -13 in codes or 13 in codes:  # muon
            # CC Interaction: nu + N -> mu + hadrons
            muons = [p for p in daughters if p.pdg_encoding in (13, -13)]
            assert (
                len(muons) == 1
            ), "Found more or less than one expected muon."
            return muons[0]
        elif -14 in codes or 14 in codes:
            # NC Interaction: nu + N -> nu + hadrons
            neutrinos = [p for p in daughters if p.pdg_encoding in (14, -14)]
            assert (
                len(neutrinos) == 1
            ), "Found more or less than one expected neutrino."
            return get_next_muon_daughter_of_nu(frame, neutrinos[0])
    else:
        return None


def get_parent_muons(frame, particle=None, mctree_name="I3MCTree"):
    """Get "parent" muons

    Definition of "parent" muon in this context:

    Any muon in the I3MCTree that does not have another muon
    as a parent.

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame. Must contain the specified I3MCTree.
    particle : I3Particle
        The primary particle. If None, the primary particle of the
        I3MCTree will be chose. Will throw an error if more than
        one primary particle is present.
    mctree_name : str, optional
        The name of the I3MCTree.

    Returns
    -------
    list of I3Particle
        A list of "parent" muons.
    """

    # get the primary particle of the I3MCTree if none provided
    if particle is None:
        particle = get_weighted_primary(frame=frame, mctree_name=mctree_name)

    muons = []
    # Now walk through each daughter particle recursively
    for daughter in frame[mctree_name].get_daughters(particle):
        if is_muon(daughter):
            muons.append(daughter)
        else:
            muons += get_parent_muons(
                frame=frame,
                particle=daughter,
                mctree_name=mctree_name,
            )

    return muons


def get_muon_track_length_inside(muon, convex_hull):
    """Get the track length of the muon
        inside the convex hull.
        Returns 0 if muon doesn't hit hull.

    Parameters
    ----------
    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    track_length : float
        Returns 0 if muon doesn't hit
        convex hull.
    """
    intersection_ts = get_muon_convex_hull_intersections(muon, convex_hull)

    # muon didn't hit convex_hull
    if len(intersection_ts) == 0:
        return 0

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting track
        return min(max_ts, muon.length)
    if max_ts < 0:
        # muon created after the convex hull
        return 0
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return 0
    return min(max_ts, muon.length) - min_ts


def get_bundle_radius(
    frame,
    primary=None,
    quantiles=[0.1, 0.5, 0.68, 0.9],
    ref_point=dataclasses.I3Position(0.0, 0.0, 0.0),
    mctree_name="I3MCTree",
    mmctracklist_name="MMCTrackList",
    track_cache=None,
):
    """Computes the bundle radius for given tracks

    The bundle radius is defined here as the distribution in distances
    to the most energetic muon at the closest approach point of the
    defined reference point `ref_point`. Note that this definition may be
    mis-leading in case muons distribute all along the same direction of
    the leading muon.

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame.
    primary : I3Particle, optional
        The primary particle for which to collect the bundle.
        If None is provided, the head of the I3MCTree is utilized.
    quantiles : list of float, optional
        The quantiles for which to compute the bundle radius.
    ref_point : I3Position, optional
        The radius of the bundle is compute based on the energy-weighted
        profile at the closest approach position of this reference point.
    mctree_name : str, optional
        The name of the propagated I3MCTree.
    mmctracklist_name : str, optional
        The name of the MMCTrackList.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    array_like
        The quantiles (fraction of total energy at the reference point) for
        which the bundle radii are calculated.
    array_like
        An array of distances [meters] that describe the energy-weighted
        radius of the bundle. Each entry corresponds to the distance from
        the leading muon at which a fraction of q of the total charge is
        included. The fraction q corresponds to the quantiles provided.
    """

    # get the primary if none is provided
    if primary is None:
        primary = frame[mctree_name].get_head()

    # get all tracks in active volume
    tracks = [
        m.particle
        for m in frame[mmctracklist_name]
        if frame[mctree_name].get_primary(m.particle) == primary
    ]

    return get_bundle_radius_from_tracks(
        tracks=tracks,
        frame=frame,
        quantiles=quantiles,
        ref_point=ref_point,
        track_cache=track_cache,
    )


def get_bundle_radius_from_tracks(
    tracks,
    frame,
    quantiles=[0.1, 0.5, 0.68, 0.9],
    ref_point=dataclasses.I3Position(0.0, 0.0, 0.0),
    mctree_name="I3MCTree",
    mmctracklist_name="MMCTrackList",
    track_cache=None,
):
    """Computes the bundle radius for given tracks

    The bundle radius is defined here as the distribution in distances
    to the most energetic muon at the closest approach point of the
    defined reference point `ref_point`. Note that this definition may be
    mis-leading in case muons distribute all along the same direction of
    the leading muon.

    Parameters
    ----------
    tracks : List of I3Particle
        A list of tracks belonging to the bundle.
    frame : I3Frame
        The current I3Frame.
    quantiles : list of float, optional
        The quantiles for which to compute the bundle radius.
    ref_point : I3Position, optional
        The radius of the bundle is compute based on the energy-weighted
        profile at the closest approach position of this reference point.
    mctree_name : str, optional
        The name of the propagated I3MCTree.
    mmctracklist_name : str, optional
        The name of the MMCTrackList.
    track_cache : dict[MuonGun.Track], optional
        A dictionary of the harvested MuonGun tracks in the frame.
        The structure of the dictionary is {particle_id: MuonGun.Track}.

    Returns
    -------
    array_like
        The quantiles (fraction of total energy at the reference point) for
        which the bundle radii are calculated.
    array_like
        An array of distances [meters] that describe the energy-weighted
        radius of the bundle. Each entry corresponds to the distance from
        the leading muon at which a fraction of q of the total charge is
        included. The fraction q corresponds to the quantiles provided.
    """
    max_energy = -np.inf
    closest_points = []
    energies = []
    for track in tracks:

        # compute closest approach position of infinite long track
        closest_pos = I3Calculator.closest_approach_position(track, ref_point)

        energy = get_muon_energy_at_position(
            frame=frame,
            muon=track,
            position=closest_pos,
            track_cache=track_cache,
        )

        if np.isfinite(energy) and energy > 0:
            energies.append(energy)
            closest_points.append(closest_pos)

            if energy > max_energy:
                center_point = closest_pos
                max_energy = energy

    if len(energies) == 0:
        return quantiles, np.ones(len(quantiles)) * np.nan

    # now compute distances to point of maximum energy shower
    distances = [(pos - center_point).magnitude for pos in closest_points]

    # sort along distance
    sorted_idx = np.argsort(distances)
    distances = np.array(distances)[sorted_idx]
    energies = np.array(energies)[sorted_idx]
    cum_energy_fraction = np.cumsum(energies) / np.sum(energies)

    # account for numerical issues
    assert np.allclose(cum_energy_fraction[-1], 1.0), cum_energy_fraction
    cum_energy_fraction = np.clip(cum_energy_fraction, 0.0, 1.0)
    cum_energy_fraction[-1] = 1.0

    # compute bundle radii for each of the quantiles
    radii = distances[np.searchsorted(cum_energy_fraction, quantiles)]

    return quantiles, radii
