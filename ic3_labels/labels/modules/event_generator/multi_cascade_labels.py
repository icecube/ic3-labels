import numpy as np
from icecube import dataclasses, icetray

from ic3_labels.labels.base_module import MCLabelsBase
from ic3_labels.labels.utils import high_level as hl


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
        # that each cascade is injected as a primary neutrino, where
        # the first one defines the primary cascade
        neutrinos = mc_tree.get_primaries()

        labels = hl.get_cascade_parameters(
            frame, neutrinos[0],
            convex_hull=self._convex_hull,
            extend_boundary=self._extend_boundary,
            write_mc_cascade_to_frame=False,
        )

        # Now extract cascade info from each neutrino
        for i, primary in enumerate(neutrinos[1:]):
            labels_i = hl.get_cascade_parameters(
                frame, primary,
                convex_hull=self._convex_hull,
                extend_boundary=self._extend_boundary,
                write_mc_cascade_to_frame=False,
            )
            for key, value in labels_i.items():
                labels[key + '_{:05d}'] = value

        # write to frame
        frame.Put(self._output_key, labels)
