import numpy as np
from icecube import dataclasses, icetray

from ic3_labels.labels.base_module import MCLabelsBase
from ic3_labels.labels.utils import high_level as hl
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.modules.event_generator.utils import (
    get_track_energy_depositions,
    compute_stochasticity,
)


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
        muon = self.get_muon(frame, primary=frame[self._primary_key])

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

    def get_muon(self, frame, primary):
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
                assert mu_utils.is_muon(
                    muon
                ), "Expected muon but got {!r}".format(muon)

        return muon
