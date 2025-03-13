""" I3Modules to add Labels for deep Learning
"""

from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray
from icecube.icetray.i3logging import log_error, log_warn

from ic3_labels.labels.base_module import MCLabelsBase
from ic3_labels.labels.utils import high_level as hl
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils import general


class MCLabelsDeepLearning(MCLabelsBase):
    """Creates extensive Muon, primary and misc Labels."""

    def __init__(self, context):
        # super(MCLabelsDeepLearning, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter(
            "IsMuonGun", "Indicate whether this is a MuonGun dataset.", False
        )

    def Configure(self):
        # super(MCLabelsDeepLearning, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._is_muongun = self.GetParameter("IsMuonGun")

    def add_labels(self, frame):
        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        labels = hl.get_labels(
            frame=frame,
            convex_hull=self._convex_hull,
            domPosDict=self._dom_pos_dict,
            primary=frame[self._primary_key],
            pulse_map_string=self._pulse_map_string,
            mcpe_series_map_name=self._mcpe_series_map_name,
            is_muongun=self._is_muongun,
            track_cache=track_cache,
        )

        # write to frame
        frame.Put(self._output_key, labels)


class MCLabelsTau(MCLabelsBase):
    def add_labels(self, frame):
        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        labels = hl.get_tau_labels(
            frame=frame,
            convex_hull=self._convex_hull,
            track_cache=track_cache,
        )

        # write to frame
        frame.Put(self._output_key, labels)


class MCLabelsCascadeParameters(MCLabelsBase):

    def add_labels(self, frame):
        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        labels = hl.get_cascade_parameters(
            frame=frame,
            primary=frame[self._primary_key],
            convex_hull=self._convex_hull,
            extend_boundary=500,
            track_cache=track_cache,
        )

        # write to frame
        frame.Put(self._output_key, labels)


class MCLabelsCascades(MCLabelsBase):

    def __init__(self, context):
        # super(MCLabelsCascades, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter(
            "ExtendBoundary", "Extend boundary of convex hull [in meters].", 0
        )

    def Configure(self):
        # super(MCLabelsCascades, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._extend_boundary = self.GetParameter("ExtendBoundary")

    def add_labels(self, frame):

        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        labels = hl.get_cascade_labels(
            frame=frame,
            primary=frame[self._primary_key],
            convex_hull=self._convex_hull,
            extend_boundary=self._extend_boundary,
            track_cache=track_cache,
        )

        # write to frame
        frame.Put(self._output_key, labels)


class MCLabelsCorsikaMultiplicity(MCLabelsBase):
    def add_labels(self, frame):

        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        labels = hl.get_muon_bundle_information(
            frame=frame,
            convex_hull=self._convex_hull,
            track_cache=track_cache,
        )

        labels["num_coincident_events"] = general.get_num_coincident_events(
            frame
        )

        primary = frame[self._primary_key]
        labels["PrimaryEnergy"] = primary.energy
        labels["PrimaryAzimuth"] = primary.dir.azimuth
        labels["PrimaryZenith"] = primary.dir.zenith
        labels["PrimaryDirectionX"] = primary.dir.x
        labels["PrimaryDirectionY"] = primary.dir.y
        labels["PrimaryDirectionZ"] = primary.dir.z

        label_names = [
            "num_coincident_events",
            "num_muons",
            "num_muons_at_cyl",
            "num_muons_at_cyl_above_threshold",
            "num_muons_at_entry",
            "num_muons_at_entry_above_threshold",
        ]
        pid_names = [
            "p_is_coincident_event",
            "p_is_muon_bundle",
            "p_is_muon_bundle_at_cyl",
            "p_is_muon_bundle_at_cyl_above_threshold",
            "p_is_muon_bundle_at_entry",
            "p_is_muon_bundle_at_entry_above_threshold",
        ]

        for label, p_name in zip(label_names, pid_names):
            labels[p_name] = labels[label] > 1

        # write to frame
        frame.Put(self._output_key, dataclasses.I3MapStringDouble(labels))


class MCLabelsCorsikaAzimuthExcess(MCLabelsBase):
    def add_labels(self, frame):

        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        # create empty labelDict
        labels = dataclasses.I3MapStringDouble()

        muons_inside = mu_utils.get_muons_inside(frame, self._convex_hull)
        labels["NoOfMuonsInside"] = len(muons_inside)

        # get muons
        mostEnergeticMuon = mu_utils.get_most_energetic_muon_inside(
            frame,
            self._convex_hull,
            muons_inside=muons_inside,
            track_cache=track_cache,
        )

        if mostEnergeticMuon is None:
            labels["Muon_energy"] = np.nan
            labels["Muon_vertexX"] = np.nan
            labels["Muon_vertexY"] = np.nan
            labels["Muon_vertexZ"] = np.nan
            labels["Muon_vertexTime"] = np.nan
            labels["Muon_azimuth"] = np.nan
            labels["Muon_zenith"] = np.nan
        else:
            labels["Muon_energy"] = mostEnergeticMuon.energy
            labels["Muon_vertexX"] = mostEnergeticMuon.pos.x
            labels["Muon_vertexY"] = mostEnergeticMuon.pos.y
            labels["Muon_vertexZ"] = mostEnergeticMuon.pos.z
            labels["Muon_vertexTime"] = mostEnergeticMuon.time
            labels["Muon_azimuth"] = mostEnergeticMuon.dir.azimuth
            labels["Muon_zenith"] = mostEnergeticMuon.dir.zenith

        # write to frame
        frame.Put(self._output_key, labels)


class MCLabelsMuonScattering(MCLabelsBase):
    """Creates labels to identify muon scattering."""

    def __init__(self, context):
        # super(MCLabelsDeepLearning, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter(
            "MinLength",
            "Minimum required track length inside detector to "
            "qualify an event as a muon scattering event",
            1000,
        )
        self.AddParameter(
            "MinLengthBefore",
            "Minimum required track length inside detector "
            "before the energy loss to qualify an event as a "
            "muon scattering event",
            400,
        )
        self.AddParameter(
            "MinLengthAfter",
            "Minimum required track length inside detector "
            "after the energy loss to qualify an event as a "
            "muon scattering event",
            400,
        )
        self.AddParameter(
            "MinMuonEntryEnergy",
            "Minimum required muon energy at point of entry "
            "to qualify an event as a muon scattering event",
            10000,
        )
        self.AddParameter(
            "MinRelativeLossEnergy",
            "Minimum required relative energy of the muon loss "
            "to qualify an event as a muon scattering event. "
            "The relative energy loss is calculated as the "
            "loss energy / muon energy at entry",
            0.5,
        )

    def Configure(self):
        # super(MCLabelsDeepLearning, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._min_length = self.GetParameter("MinLength")
        self._min_length_before = self.GetParameter("MinLengthBefore")
        self._min_length_after = self.GetParameter("MinLengthAfter")
        self._min_muon_entry_energy = self.GetParameter("MinMuonEntryEnergy")
        self._min_rel_loss_energy = self.GetParameter("MinRelativeLossEnergy")

    def add_labels(self, frame):

        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        labels = mu_utils.get_muon_scattering_info(
            frame=frame,
            primary=frame[self._primary_key],
            convex_hull=self._convex_hull,
            min_length=self._min_length,
            min_length_before=self._min_length_before,
            min_length_after=self._min_length_after,
            min_muon_entry_energy=self._min_muon_entry_energy,
            min_rel_loss_energy=self._min_rel_loss_energy,
            track_cache=track_cache,
        )
        frame.Put(self._output_key, labels)


class MCLabelsMuonEnergyLosses(MCLabelsBase):
    def __init__(self, context):
        MCLabelsBase.__init__(self, context)
        self.AddParameter(
            "MuonKey",
            "Name of the muon to consider for the energy losses."
            "If None, the muon from the primary key is used.",
            None,
        )
        self.AddParameter("BinWidth", "Bin width [in meters].", 10)
        self.AddParameter(
            "ExtendBoundary",
            "Extend boundary of convex hull [in meters].",
            150,
        )
        self.AddParameter(
            "IncludeUnderOverFlow", "Include over and under flow bins.", False
        )
        self.AddParameter(
            "ForceNumBins",
            "Force number of bins to be this value."
            "Will append zeros or remove last bins.",
            None,
        )

    def Configure(self):
        MCLabelsBase.Configure(self)
        self._muon_key = self.GetParameter("MuonKey")
        self._bin_width = self.GetParameter("BinWidth")
        self._extend_boundary = self.GetParameter("ExtendBoundary")
        self._force_num_bins = self.GetParameter("ForceNumBins")
        self._include_under_over_flow = self.GetParameter(
            "IncludeUnderOverFlow"
        )

    def add_labels(self, frame):

        # get muon
        if self._muon_key is None:
            # get track_cache
            track_cache, _ = mu_utils.get_muongun_track_cache(frame)

            muon = mu_utils.get_muon(
                frame=frame,
                primary=frame[self._primary_key],
                convex_hull=self._convex_hull,
                track_cache=track_cache,
            )
        else:
            muon = frame[self._muon_key]

        labels = dataclasses.I3MapStringDouble()

        binned_energy_losses = mu_utils.get_inf_muon_binned_energy_losses(
            frame=frame,
            convex_hull=self._convex_hull,
            muon=muon,
            bin_width=self._bin_width,
            extend_boundary=self._extend_boundary,
            include_under_over_flow=self._include_under_over_flow,
        )

        # force the number of bins to match ForceNumBins
        if self._force_num_bins is not None:

            num_bins = len(binned_energy_losses)

            # too many bins: remove last bins
            if num_bins > self._force_num_bins:
                binned_energy_losses = binned_energy_losses[
                    : self._force_num_bins
                ]

            # too few bins: append zeros
            elif num_bins < self._force_num_bins:
                num_bins_to_add = self._force_num_bins - num_bins
                # print('Appending {} zeros'.format(num_bins_to_add))
                binned_energy_losses = np.concatenate(
                    (binned_energy_losses, np.zeros(num_bins_to_add))
                )

        # write to frame
        for i, energy_i in enumerate(binned_energy_losses):
            labels["EnergyLoss_{:04d}".format(i)] = energy_i

        frame.Put(self._output_key, labels)


class MCLabelsMuonEnergyLossesInCylinder(MCLabelsBase):
    def __init__(self, context):
        MCLabelsBase.__init__(self, context)
        self.AddParameter("BinWidth", "Bin width [in meters].", 15)
        self.AddParameter("NumBins", "Number of bins to create.", 100)
        self.AddParameter(
            "CylinderHeight",
            "The height (z) of the axial clinder [in meters].",
            1000.0,
        )
        self.AddParameter(
            "CylinderRadius",
            "The radius (x-y) of the axial clinder [in meters].",
            600.0,
        )

    def Configure(self):
        MCLabelsBase.Configure(self)
        self._bin_width = self.GetParameter("BinWidth")
        self._num_bins = self.GetParameter("NumBins")
        self._cylinder_height = self.GetParameter("CylinderHeight")
        self._cylinder_radius = self.GetParameter("CylinderRadius")

    def add_labels(self, frame):

        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        # get muon
        muon = mu_utils.get_muon(
            frame=frame,
            primary=frame[self._primary_key],
            convex_hull=self._convex_hull,
            track_cache=track_cache,
        )

        labels = dataclasses.I3MapStringDouble()

        binned_energy_losses = mu_utils.get_binned_energy_losses_in_cylinder(
            frame=frame,
            muon=muon,
            bin_width=self._bin_width,
            num_bins=self._num_bins,
            cylinder_height=self._cylinder_height,
            cylinder_radius=self._cylinder_radius,
            track_cache=track_cache,
        )

        # write to frame
        labels["track_anchor_x"] = muon.pos.x
        labels["track_anchor_y"] = muon.pos.y
        labels["track_anchor_z"] = muon.pos.z
        labels["track_anchor_time"] = muon.time
        labels["azimuth"] = muon.dir.azimuth
        labels["zenith"] = muon.dir.zenith

        for i, energy_i in enumerate(binned_energy_losses):
            labels["EnergyLoss_{:05d}".format(i)] = energy_i

        frame.Put(self._output_key, labels)


class MCLabelsMuonEnergyLossesMillipede(MCLabelsBase):
    def __init__(self, context):
        MCLabelsBase.__init__(self, context)
        self.AddParameter("BinWidth", "Bin width [in meters].", 15)
        self.AddParameter(
            "Boundary",
            "Half edge length of a cube [in meters]. "
            + "Will be used as a boundary."
            + "Millipede default are 600m.",
            600.0,
        )
        self.AddParameter(
            "WriteParticleVector",
            "Also writes the labels in form of "
            + "a particle vector to be visualized "
            + "via steamshovel",
            False,
        )
        self.AddParameter(
            "MaxNumBins",
            "If provided, exactly this number of bins is "
            + "added to the labels. Non existing bins are "
            + "padded with NaNs. Additional bins are cut off. "
            + "This can be useful when writing tabular data "
            + "that requires fixed sizes.",
            None,
        )

    def Configure(self):
        MCLabelsBase.Configure(self)
        self._bin_width = self.GetParameter("BinWidth")
        self._boundary = self.GetParameter("Boundary")
        self._write_vector = self.GetParameter("WriteParticleVector")
        self._max_num_bins = self.GetParameter("MaxNumBins")

    def add_labels(self, frame):

        # get track_cache
        track_cache, _ = mu_utils.get_muongun_track_cache(frame)

        # get muon
        muon = mu_utils.get_muon(
            frame=frame,
            primary=frame[self._primary_key],
            convex_hull=self._convex_hull,
            track_cache=track_cache,
        )

        labels = dataclasses.I3MapStringDouble()

        if self._write_vector:
            binned_energy_losses, bin_center_pos = (
                mu_utils.get_binned_energy_losses_in_cube(
                    frame=frame,
                    muon=muon,
                    bin_width=self._bin_width,
                    boundary=self._boundary,
                    return_bin_centers=self._write_vector,
                    track_cache=track_cache,
                )
            )
        else:
            binned_energy_losses = mu_utils.get_binned_energy_losses_in_cube(
                frame=frame,
                muon=muon,
                bin_width=self._bin_width,
                boundary=self._boundary,
                return_bin_centers=self._write_vector,
                track_cache=track_cache,
            )

        # write to frame
        labels["track_anchor_x"] = muon.pos.x
        labels["track_anchor_y"] = muon.pos.y
        labels["track_anchor_z"] = muon.pos.z
        labels["track_anchor_time"] = muon.time
        labels["azimuth"] = muon.dir.azimuth
        labels["zenith"] = muon.dir.zenith

        for i, energy_i in enumerate(binned_energy_losses):

            # stop adding energy losses if we reached the maximum
            if self._max_num_bins is not None:
                if i >= self._max_num_bins:
                    msg = "MaxNumBinsis set to {}. ".format(self._max_num_bins)
                    msg += "Cutting off an additional {} losses!".format(
                        len(binned_energy_losses) - self._max_num_bins
                    )
                    log_warn(msg)
                    break

            labels["EnergyLoss_{:05d}".format(i)] = energy_i

        # pad rest with NaNs
        if self._max_num_bins is not None:
            for i in range(len(binned_energy_losses), self._max_num_bins):
                labels["EnergyLoss_{:05d}".format(i)] = float("NaN")

        frame.Put(self._output_key, labels)

        if self._write_vector:
            part_vec = dataclasses.I3VectorI3Particle()
            for energy_i, pos_i in zip(binned_energy_losses, bin_center_pos):
                part = dataclasses.I3Particle()
                part.pos = dataclasses.I3Position(*pos_i)
                part.energy = energy_i
                part.dir = dataclasses.I3Direction(muon.dir)
                part.time = (
                    muon.pos - part.pos
                ).magnitude / dataclasses.I3Constants.c
                part_vec.append(part)
            frame.Put(self._output_key + "ParticleVector", part_vec)
