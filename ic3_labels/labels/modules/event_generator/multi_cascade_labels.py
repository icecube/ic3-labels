import numpy as np
from icecube import dataclasses, icetray

from ic3_labels.labels.base_module import MCLabelsBase
from ic3_labels.labels.utils import high_level as hl
from ic3_labels.labels.utils.cascade import get_cascade_em_equivalent


class EventGeneratorMultiCascadeLabels(MCLabelsBase):

    """Class to get multi cascade labels for Event-Generator

    Important: this class depends on a synthetic simulation where a
    hypothetical track is created, by injecting N neutrino-induced
    cascades along the track direction.
    """

    def __init__(self, context):
        # super(EventGeneratorMuonTrackLabels, self).__init__(self, context)
        MCLabelsBase.__init__(self, context)
        self.AddParameter("MCTreeName",
                          "Name of the I3MCTree to use.",
                          'I3MCTree')
        self.AddParameter("ExtendBoundary",
                          "Extend boundary of convex hull around IceCube "
                          "[in meters].",
                          500)
        self.AddParameter("RunOnDAQFrames",
                          "If True, the label module will run on DAQ frames "
                          "instead of Physics frames",
                          False)

    def Configure(self):
        # super(EventGeneratorMuonTrackLabels, self).Configure(self)
        MCLabelsBase.Configure(self)
        self._extend_boundary = self.GetParameter("ExtendBoundary")
        self._mc_tree_name = self.GetParameter("MCTreeName")
        self._run_on_daq = self.GetParameter("RunOnDAQFrames")

    def DAQ(self, frame):
        """Run on DAQ frames.

        Parameters
        ----------
        frame : I3Frame
            The current DAQ Frame
        """
        if self._run_on_daq:
            self.add_labels(frame)

        self.PushFrame(frame)

    def Physics(self, frame):
        """Run on Physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Physics Frame
        """
        if not self._run_on_daq:
            self.add_labels(frame)

        self.PushFrame(frame)

    def add_labels(self, frame):
        """Add labels to frame

        Parameters
        ----------
        frame : I3Frame
            The frame to which to add the labels.
        """
        mc_tree = frame[self._mc_tree_name]

        # get cascades: for this synthetic simulation it is assumed
        # that each cascade is injected as a daugther neutrino into
        # the primary neutrino
        primary = frame[self._primary_key]

        labels = hl.get_cascade_parameters(
            frame, primary,
            convex_hull=self._convex_hull,
            extend_boundary=self._extend_boundary,
            write_mc_cascade_to_frame=False,
        )
        primary_cascade = mc_tree.get_daughters(primary)[0]

        # Now extract cascade info from each daughter neutrino
        neutrinos = [p for p in mc_tree.get_daughters(primary)
                     if p.is_neutrino]
        for i, neutrino in enumerate(neutrinos):

            cascade_name = 'cascade_{:05d}_'.format(i + 1)

            daughters = mc_tree.get_daughters(neutrino)
            if daughters[0].time > primary_cascade.time:
                sign = 1
            else:
                sign = -1

            distance = (
                primary_cascade.pos - primary.daughters[0].pos
            ).magnitude * sign

            # sum up energies for daughters if not neutrinos
            # tau can immediately decay in neutrinos which carry away energy
            # that would not be visible, this is currently not accounted for
            e_total, e_em, e_hadron, e_track = get_cascade_em_equivalent(
                mctree, neutrino)

            labels[cascade_name + 'x'] = daughters[0].pos.x
            labels[cascade_name + 'y'] = daughters[0].pos.y
            labels[cascade_name + 'z'] = daughters[0].pos.z
            labels[cascade_name + 't'] = daughters[0].time
            labels[cascade_name + 'distance'] = distance
            labels[cascade_name + 'energy'] = e_total

            # compute (EM equivalen) fraction of energy for each component:
            # EM, hadronic, track
            labels[cascade_name+'energy_fraction_em'] = e_em / e_total
            labels[cascade_name+'energy_fraction_hadron'] = e_hadron / e_total
            labels[cascade_name+'energy_fraction_track'] = e_track / e_total

        # write to frame
        frame.Put(self._output_key, labels)
