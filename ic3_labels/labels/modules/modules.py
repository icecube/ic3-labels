#!/usr/bin/env python
# -*- coding: utf-8 -*
''' I3Modules to add Labels for deep Learning
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray

from ic3_labels.labels.base_module import MCLabelsBase
from ic3_labels.labels.utils import high_level as hl
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils import general


class MCLabelsDeepLearning(MCLabelsBase):

    """Creates extensive Muon, primary and misc Labels.
    """

    def __init__(self, context):
        # super(MCLabelsDeepLearning, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter("IsMuonGun",
                          "Indicate whether this is a MuonGun dataset.", False)

    def Configure(self):
        # super(MCLabelsDeepLearning, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._is_muongun = self.GetParameter("IsMuonGun")

    def Physics(self, frame):
        labels = hl.get_labels(frame=frame,
                               convex_hull=self._convex_hull,
                               domPosDict=self._dom_pos_dict,
                               primary=frame[self._primary_key],
                               pulse_map_string=self._pulse_map_string,
                               mcpe_series_map_name=self._mcpe_series_map_name,
                               is_muongun=self._is_muongun)

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsTau(MCLabelsBase):
    def Physics(self, frame):
        labels = hl.get_tau_labels(
                    frame=frame,
                    convex_hull=self._convex_hull)

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsCascadeParameters(MCLabelsBase):
    def Physics(self, frame):
        labels = hl.get_cascade_parameters(frame=frame,
                                           primary=frame[self._primary_key],
                                           convex_hull=self._convex_hull,
                                           extend_boundary=500)

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsCascades(MCLabelsBase):

    def __init__(self, context):
        # super(MCLabelsCascades, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter("ExtendBoundary",
                          "Extend boundary of convex hull [in meters].",
                          0)

    def Configure(self):
        # super(MCLabelsCascades, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._extend_boundary = self.GetParameter("ExtendBoundary")

    def Physics(self, frame):
        labels = hl.get_cascade_labels(frame=frame,
                                       primary=frame[self._primary_key],
                                       convex_hull=self._convex_hull,
                                       extend_boundary=self._extend_boundary)

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsCorsikaMultiplicity(MCLabelsBase):
    def Physics(self, frame):
        labels = hl.get_muon_bundle_information(frame=frame,
                                                convex_hull=self._convex_hull)
        labels['num_coincident_events'] = \
            general.get_num_coincident_events(frame)

        primary = frame[self._primary_key]
        labels['PrimaryEnergy'] = primary.energy
        labels['PrimaryAzimuth'] = primary.dir.azimuth
        labels['PrimaryZenith'] = primary.dir.zenith
        labels['PrimaryDirectionX'] = primary.dir.x
        labels['PrimaryDirectionY'] = primary.dir.y
        labels['PrimaryDirectionZ'] = primary.dir.z

        label_names = ['num_coincident_events', 'num_muons',
                       'num_muons_at_cyl', 'num_muons_at_cyl_above_threshold',
                       'num_muons_at_entry',
                       'num_muons_at_entry_above_threshold']
        pid_names = ['p_is_coincident_event', 'p_is_muon_bundle',
                     'p_is_muon_bundle_at_cyl',
                     'p_is_muon_bundle_at_cyl_above_threshold',
                     'p_is_muon_bundle_at_entry',
                     'p_is_muon_bundle_at_entry_above_threshold']

        for label, p_name in zip(label_names, pid_names):
            labels[p_name] = labels[label] > 1

        # write to frame
        frame.Put(self._output_key, dataclasses.I3MapStringDouble(labels))

        self.PushFrame(frame)


class MCLabelsCorsikaAzimuthExcess(MCLabelsBase):
    def Physics(self, frame):
        # create empty labelDict
        labels = dataclasses.I3MapStringDouble()

        muons_inside = mu_utils.get_muons_inside(frame, self._convex_hull)
        labels['NoOfMuonsInside'] = len(muons_inside)

        # get muons
        mostEnergeticMuon = mu_utils.get_most_energetic_muon_inside(
                                                frame, self._convex_hull,
                                                muons_inside=muons_inside)
        if mostEnergeticMuon is None:
            labels['Muon_energy'] = np.nan
            labels['Muon_vertexX'] = np.nan
            labels['Muon_vertexY'] = np.nan
            labels['Muon_vertexZ'] = np.nan
            labels['Muon_vertexTime'] = np.nan
            labels['Muon_azimuth'] = np.nan
            labels['Muon_zenith'] = np.nan
        else:
            labels['Muon_energy'] = mostEnergeticMuon.energy
            labels['Muon_vertexX'] = mostEnergeticMuon.pos.x
            labels['Muon_vertexY'] = mostEnergeticMuon.pos.y
            labels['Muon_vertexZ'] = mostEnergeticMuon.pos.z
            labels['Muon_vertexTime'] = mostEnergeticMuon.time
            labels['Muon_azimuth'] = mostEnergeticMuon.dir.azimuth
            labels['Muon_zenith'] = mostEnergeticMuon.dir.zenith

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsMuonScattering(MCLabelsBase):

    """Creates labels to identify muon scattering.
    """

    def __init__(self, context):
        # super(MCLabelsDeepLearning, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter("MinLength",
                          "Minimum required track lenth inside detector to "
                          "qualify an event as a muon scattering event",
                          1000)
        self.AddParameter("MinLengthBefore",
                          "Minimum required track lenth inside detector "
                          "before the energy loss to qualify an event as a "
                          "muon scattering event",
                          400)
        self.AddParameter("MinLengthAfter",
                          "Minimum required track lenth inside detector "
                          "after the energy loss to qualify an event as a "
                          "muon scattering event",
                          400)
        self.AddParameter("MinMuonEntryEnergy",
                          "Minimum required muon energy at point of entry "
                          "to qualify an event as a muon scattering event",
                          10000)
        self.AddParameter("MinRelativeLossEnergy",
                          "Minimum required relative energy of the muon loss "
                          "to qualify an event as a muon scattering event. "
                          "The relative energy loss is calculated as the "
                          "loss energy / muon energy at entry",
                          0.5)

    def Configure(self):
        # super(MCLabelsDeepLearning, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._min_length = self.GetParameter("MinLength")
        self._min_length_before = self.GetParameter("MinLengthBefore")
        self._min_length_after = self.GetParameter("MinLengthAfter")
        self._min_muon_entry_energy = self.GetParameter("MinMuonEntryEnergy")
        self._min_rel_loss_energy = self.GetParameter("MinRelativeLossEnergy")

    def Physics(self, frame):
        labels = mu_utils.get_muon_scattering_info(
                            frame=frame,
                            primary=frame[self._primary_key],
                            convex_hull=self._convex_hull,
                            min_length=self._min_length,
                            min_length_before=self._min_length_before,
                            min_length_after=self._min_length_after,
                            min_muon_entry_energy=self._min_muon_entry_energy,
                            min_rel_loss_energy=self._min_rel_loss_energy,
                            )
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsMuonEnergyLosses(MCLabelsBase):
    def __init__(self, context):
        MCLabelsBase.__init__(self, context)
        self.AddParameter("MuonKey", "Name of the muon.", 'MCPrimary')
        self.AddParameter("BinWidth", "Bin width [in meters].", 10)
        self.AddParameter("ExtendBoundary",
                          "Extend boundary of convex hull [in meters].",
                          150)
        self.AddParameter("IncludeUnderOverFlow",
                          "Include over and under flow bins.",
                          False)
        self.AddParameter("ForceNumBins",
                          "Force number of bins to be this value."
                          "Will append zeros or remove last bins.",
                          None)

    def Configure(self):
        MCLabelsBase.Configure(self)
        self._muon_key = self.GetParameter("MuonKey")
        self._bin_width = self.GetParameter("BinWidth")
        self._extend_boundary = self.GetParameter("ExtendBoundary")
        self._force_num_bins = self.GetParameter("ForceNumBins")
        self._include_under_over_flow = \
            self.GetParameter("IncludeUnderOverFlow")

    def Physics(self, frame):

        labels = dataclasses.I3MapStringDouble()

        binnned_energy_losses = mu_utils.get_inf_muon_binned_energy_losses(
                        frame=frame,
                        convex_hull=self._convex_hull,
                        muon=frame[self._muon_key],
                        bin_width=self._bin_width,
                        extend_boundary=self._extend_boundary,
                        include_under_over_flow=self._include_under_over_flow,
                        )

        # force the number of bins to match ForceNumBins
        if self._force_num_bins is not None:

            num_bins = len(binnned_energy_losses)

            # too many bins: remove last bins
            if num_bins > self._force_num_bins:
                binnned_energy_losses = \
                    binnned_energy_losses[:self._force_num_bins]

            # too few bins: append zeros
            elif num_bins < self._force_num_bins:
                num_bins_to_add = self._force_num_bins - num_bins
                # print('Appending {} zeros'.format(num_bins_to_add))
                binnned_energy_losses = np.concatenate((
                            binnned_energy_losses, np.zeros(num_bins_to_add)))

        # write to frame
        for i, energy_i in enumerate(binnned_energy_losses):
            labels['EnergyLoss_{:04d}'.format(i)] = energy_i

        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsMuonEnergyLossesInCylinder(MCLabelsBase):
    def __init__(self, context):
        MCLabelsBase.__init__(self, context)
        self.AddParameter("BinWidth", "Bin width [in meters].", 15)
        self.AddParameter("NumBins", "Number of bins to create.", 100)
        self.AddParameter("CylinderHeight",
                          "The height (z) of the axial clinder [in meters].",
                          1000.)
        self.AddParameter("CylinderRadius",
                          "The radius (x-y) of the axial clinder [in meters].",
                          600.)

    def Configure(self):
        MCLabelsBase.Configure(self)
        self._bin_width = self.GetParameter("BinWidth")
        self._num_bins = self.GetParameter("NumBins")
        self._cylinder_height = self.GetParameter("CylinderHeight")
        self._cylinder_radius = self.GetParameter("CylinderRadius")

    def Physics(self, frame):

        # get muon
        muon = mu_utils.get_muon(
            frame=frame,
            primary=frame[self._primary_key],
            convex_hull=self._convex_hull,
        )

        labels = dataclasses.I3MapStringDouble()

        binnned_energy_losses = mu_utils.get_binned_energy_losses_in_cylinder(
            frame=frame,
            muon=muon,
            bin_width=self._bin_width,
            num_bins=self._num_bins,
            cylinder_height=self._cylinder_height,
            cylinder_radius=self._cylinder_radius,
          )

        # write to frame
        labels['track_anchor_x'] = muon.pos.x
        labels['track_anchor_y'] = muon.pos.y
        labels['track_anchor_z'] = muon.pos.z
        labels['track_anchor_time'] = muon.time
        labels['azimuth'] = muon.dir.azimuth
        labels['zenith'] = muon.dir.zenith

        for i, energy_i in enumerate(binnned_energy_losses):
            labels['EnergyLoss_{:05d}'.format(i)] = energy_i

        frame.Put(self._output_key, labels)

        self.PushFrame(frame)
