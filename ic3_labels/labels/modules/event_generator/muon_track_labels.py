import numpy as np
from icecube import dataclasses

from ic3_labels.labels.base_module import MCLabelsBase
from ic3_labels.labels.utils import high_level as hl
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils import geometry as geo_utils
from ic3_labels.labels.utils.geometry_scipy import Sphere
from ic3_labels.labels.modules.event_generator.utils import (
    get_track_energy_depositions,
    compute_stochasticity,
    get_muon_from_frame,
)


def get_sphere_inf_track_geometry_values(muon, sphere_radius):
    """Generate geometry values for tracks in the sphere

    The values contain information of the track such as:

        - information at the entry point of the infinite
          track in the sphere: (x, y, z, theta, phi, t)
        - information at the exit point of the infinite
          track in the sphere: (x, y, z, theta, phi, t)
        - Direction of the track (theta, phi)
        - Length of infinite track in sphere
        - Length of the (finite) track in the sphere

    Parameters
    ----------
    muon : I3Particle
        The muon track for which to calculate the values.
    sphere_radius : float
        The radius of the sphere around IceCube [in meters].

    Returns
    -------
    values : I3MapStringDouble
        The values for the track.
    dist_entry : float
        The distance from the start of the track to the entry point
        of the sphere.
    dist_exit : float
        The distance from the start of the track to the exit point
        of the sphere.
    """

    # get intersectios with sphere
    intersections = geo_utils.get_sphere_intersection(
        radius=sphere_radius,
        anchor=muon.pos,
        direction=muon.dir,
    )
    if intersections is None:
        raise ValueError("No intersection with sphere found!")

    dist_entry, dist_exit = intersections
    entry_pos = muon.pos + dist_entry * muon.dir
    exit_pos = muon.pos + dist_exit * muon.dir

    # compute length in detector of finite track
    end_point = min(dist_exit, muon.length)
    finite_length = end_point - dist_entry

    # compute angle representation
    entry_dir = dataclasses.I3Direction(-entry_pos)
    exit_dir = dataclasses.I3Direction(-exit_pos)

    # gather labels
    labels = dataclasses.I3MapStringDouble()
    labels["entry_x"] = entry_pos.x
    labels["entry_y"] = entry_pos.y
    labels["entry_z"] = entry_pos.z
    labels["entry_t"] = muon.time + dist_entry / muon.speed
    labels["entry_zenith"] = entry_dir.zenith
    labels["entry_azimuth"] = entry_dir.azimuth
    labels["exit_x"] = exit_pos.x
    labels["exit_y"] = exit_pos.y
    labels["exit_z"] = exit_pos.z
    labels["exit_t"] = muon.time + dist_exit / muon.speed
    labels["exit_zenith"] = exit_dir.zenith
    labels["exit_azimuth"] = exit_dir.azimuth
    labels["zenith"] = muon.dir.zenith
    labels["azimuth"] = muon.dir.azimuth
    labels["inf_length"] = dist_exit - dist_entry
    labels["finite_length"] = finite_length

    return labels, dist_entry, dist_exit


class EventGeneratorSphereInfTrackLabels(MCLabelsBase):
    """Generate labels for tracks in the sphere

    The labels contain information of the track such as:

        - information at the entry point of the infinite
          track in the sphere: (x, y, z, theta, phi, E, t)
        - information at the exit point of the infinite
          track in the sphere: (x, y, z, theta, phi, E, t)
        - Direction of the track (theta, phi)
        - Length of infinite track in sphere
        - Length of the (finite) track in the sphere
        - Deposited energy in the sphere

    Note that some of this information is redundant, but it is
    included for convenience.

    """

    def __init__(self, context):
        MCLabelsBase.__init__(self, context)
        self.AddParameter(
            "SphereRadius",
            "The radius of the sphere around IceCube [in meters].",
            750,
        )
        self.AddParameter(
            "EnergyLossBinWidth",
            "If provided, the energy losses are binned in this "
            "width [in GeV] and added to the labels.",
            None,
        )

    def Configure(self):
        MCLabelsBase.Configure(self)
        self._sphere_radius = self.GetParameter("SphereRadius")
        self._bin_width = self.GetParameter("EnergyLossBinWidth")
        self._sphere_convex_hull = Sphere(radius=self._sphere_radius)
        self._max_bins = (2 * self._sphere_radius) // self._bin_width

    def add_labels(self, frame):
        """Add labels to frame

        Parameters
        ----------
        frame : I3Frame
            The frame to which to add the labels.
        """
        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        # get muon
        muon = get_muon_from_frame(frame, primary=frame[self._primary_key])

        # get geometry values based on the infinite track
        labels, dist_entry, dist_exit = get_sphere_inf_track_geometry_values(
            muon=muon,
            sphere_radius=self._sphere_radius,
        )

        # get energy at entry and exit point
        entry_energy = mu_utils.get_muon_energy_at_distance(
            frame=frame,
            muon=muon,
            distance=dist_entry,
            track_cache=track_cache,
        )
        exit_energy = mu_utils.get_muon_energy_at_distance(
            frame=frame,
            muon=muon,
            distance=dist_exit,
            track_cache=track_cache,
        )

        # -----------------
        # add energy losses
        # -----------------
        if self._bin_width is not None:
            energy_losses = mu_utils.get_inf_muon_binned_energy_losses(
                frame=frame,
                convex_hull=self._sphere_convex_hull,
                muon=muon,
                bin_width=self._bin_width,
                extend_boundary=0,
                compute_em_equivalent=True,
                include_under_over_flow=False,
            )
            for i in range(self._max_bins):
                if i >= len(energy_losses):
                    labels[f"energy_loss_{i:04d}"] = 0.0
                else:
                    labels[f"energy_loss_{i:04d}"] = energy_losses[i]
        # -----------------

        # gather labels
        labels["entry_energy"] = entry_energy
        labels["exit_energy"] = exit_energy
        labels["deposited_energy"] = entry_energy - exit_energy

        # write to frame
        frame.Put(self._output_key, labels)


class EventGeneratorMuonTrackLabels(MCLabelsBase):
    """Class to get track labels for Event-Generator

    The computed labels contain information on the n highest charge energy
    losses as well as quantile information of the remaining track energy
    depositions.
    """

    def __init__(self, context):
        # super(EventGeneratorMuonTrackLabels, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter(
            "ExtendBoundary",
            "Extend boundary of convex hull around IceCube " "[in meters].",
            300,
        )
        self.AddParameter(
            "UseEMEquivalenEnergy",
            "Correct energy losses to obtain EM equivalent " "energy.",
            True,
        )
        self.AddParameter(
            "NumCascades",
            "Number of energy losses to treat independently " "as cascades.",
            5,
        )
        self.AddParameter(
            "NumQuantiles", "Number of track energy quantiles to compute.", 20
        )

    def Configure(self):
        # super(EventGeneratorMuonTrackLabels, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._extend_boundary = self.GetParameter("ExtendBoundary")
        self._correct_for_em_loss = self.GetParameter("UseEMEquivalenEnergy")
        self._num_cascades = self.GetParameter("NumCascades")
        self._num_quantiles = self.GetParameter("NumQuantiles")

        if self._num_quantiles > 1000:
            raise ValueError("Only quantiles up to 1000 supported!")

        self._quantiles = np.linspace(
            1.0 / self._num_quantiles, 1, self._num_quantiles
        )

    def add_labels(self, frame):
        """Add labels to frame

        Parameters
        ----------
        frame : I3Frame
            The frame to which to add the labels.
        """
        # get muon
        muon = get_muon_from_frame(frame, primary=frame[self._primary_key])

        # compute energy updates and high energy losses
        energy_depositions_dict = get_track_energy_depositions(
            mc_tree=frame["I3MCTree"],
            track=muon,
            num_to_remove=self._num_cascades,
            correct_for_em_loss=self._correct_for_em_loss,
            extend_boundary=self._extend_boundary,
        )
        update_distances = energy_depositions_dict["update_distances"]
        update_energies = energy_depositions_dict["update_energies"]
        cascades = energy_depositions_dict["cascades"]
        track_updates = energy_depositions_dict["track_updates"]

        # compute starting point of track updates
        if len(track_updates) == 0:
            track_start = muon
            track_end = muon
            stochasticity = 0.0
            area_above = 0.0
            area_below = 0.0
        else:
            track_start = track_updates[0]
            track_end = track_updates[-1]

            # compute stochasticity
            stochasticity, area_above, area_below = compute_stochasticity(
                update_distances, update_energies
            )

        # add empty cascades if not enough were returned
        for i in range(self._num_cascades - len(cascades)):
            cascade = dataclasses.I3Particle()
            cascade.pos = track_start.pos
            cascade.dir = muon.dir
            cascade.time = track_start.time
            cascade.energy = 0.0
            cascades.append(cascade)

        # chose track anchor point
        if self._num_cascades == 0:
            track_anchor = track_start
        else:
            track_anchor = cascades[0]

        # compute total deposited energy from track and its quantiles
        if len(update_distances) > 0:
            cum_energies = np.cumsum(-np.diff(update_energies))
            energy_track = cum_energies[-1]
            rel_cum_energies = cum_energies / energy_track

            indices = np.searchsorted(
                rel_cum_energies, self._quantiles, side="right"
            )
            indices = np.clip(indices, 0, len(rel_cum_energies) - 1)

            # compute relative distance to highest energy cascade
            rel_distances = (
                update_distances[1:] - (track_anchor.pos - muon.pos).magnitude
            )

            quantile_distances = rel_distances[indices]
        else:
            energy_track = 0.0
            quantile_distances = np.zeros(self._num_quantiles)

        # gather labels
        labels = dataclasses.I3MapStringDouble()
        labels["zenith"] = muon.dir.zenith
        labels["azimuth"] = muon.dir.azimuth

        for i, cascade in enumerate(cascades):

            # calculate distance to highest charge cascade
            distance = (track_anchor.pos - cascade.pos).magnitude

            labels["cascade_{:04d}_x".format(i)] = cascade.pos.x
            labels["cascade_{:04d}_y".format(i)] = cascade.pos.y
            labels["cascade_{:04d}_z".format(i)] = cascade.pos.z
            labels["cascade_{:04d}_time".format(i)] = cascade.time
            labels["cascade_{:04d}_energy".format(i)] = cascade.energy
            labels["cascade_{:04d}_zenith".format(i)] = cascade.dir.zenith
            labels["cascade_{:04d}_azimuth".format(i)] = cascade.dir.azimuth
            labels["cascade_{:04d}_distance".format(i)] = distance

        labels["track_energy"] = energy_track
        labels["track_anchor_x"] = track_anchor.pos.x
        labels["track_anchor_y"] = track_anchor.pos.y
        labels["track_anchor_z"] = track_anchor.pos.z
        labels["track_anchor_time"] = track_anchor.time
        labels["track_distance_start"] = (
            track_start.pos - muon.pos
        ).magnitude - (track_anchor.pos - muon.pos).magnitude
        labels["track_distance_end"] = (track_end.pos - muon.pos).magnitude - (
            track_anchor.pos - muon.pos
        ).magnitude
        labels["track_start_time"] = track_start.time
        labels["track_end_time"] = track_end.time
        labels["track_stochasticity"] = stochasticity
        labels["track_area_above"] = area_above
        labels["track_area_below"] = area_below

        for q, dist in zip(self._quantiles, quantile_distances):
            labels["track_quantile_{:04d}".format(int(q * 1000))] = dist

        # write to frame
        frame.Put(self._output_key, labels)
