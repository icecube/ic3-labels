''' I3Module Base Class
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray, phys_services

from ic3_labels.labels.utils import detector


class MCLabelsBase(icetray.I3ConditionalModule):

    """Base Module class for MC labels.
    """

    def __init__(self, context):
        # super(MCLabelsBase, self).__init__(self, context)
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("PulseMapString", "Name of pulse map to use.",
                          'InIcePulses')
        self.AddParameter("MCPESeriesMapName", "Name of MCPE Series map to use.",
                          'I3MCPESeriesMap')
        self.AddParameter("PrimaryKey", "Name of the primary.", 'MCPrimary')
        self.AddParameter(
            "ConvexHull",
            "The convex hull to use. Must either be an instance of "
            "icecube.phys_services.ExtrudedPolygon or a string defining a "
            "convex hull in ic3_labels.labels.utils.detector."
            "Or a string named icecube_hull_ext_{extension in meters}."
            "If None is provided, a "
            "convex hull around the outer IceCube strings will be constructed "
            "based on the given Geometry file.",
            None,
        )
        self.AddParameter("OutputKey", "Save labels to this frame key.",
                          'LabelsDeepLearning')

        self.AddParameter("RunOnDAQFrames",
                    "If True, the label module will run on DAQ frames "
                    "instead of Physics frames",
                    False)

    def Configure(self):
        self._pulse_map_string = self.GetParameter("PulseMapString")
        self._mcpe_series_map_name = self.GetParameter("MCPESeriesMapName")
        self._primary_key = self.GetParameter("PrimaryKey")
        self._convex_hull = self.GetParameter("ConvexHull")
        self._output_key = self.GetParameter("OutputKey")
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

        This method should be implemented by the derived class.
        Labels should be added to the frame using the key specified in
        self._output_key.

        Parameters
        ----------
        frame : I3Frame
            The frame to which to add the labels.
        """
        raise NotImplementedError

    def Geometry(self, frame):
        geoMap = frame['I3Geometry'].omgeo
        domPosDict = {(i[0][0], i[0][1]): (i[1].position.x,
                                           i[1].position.y,
                                           i[1].position.z)
                      for i in geoMap if i[1].omtype.name == 'IceCube'}

        if self._convex_hull is None:
            points = [
                domPosDict[(31, 1)], domPosDict[(1, 1)],
                domPosDict[(6, 1)], domPosDict[(50, 1)],
                domPosDict[(74, 1)], domPosDict[(72, 1)],
                domPosDict[(78, 1)], domPosDict[(75, 1)],

                domPosDict[(31, 60)], domPosDict[(1, 60)],
                domPosDict[(6, 60)], domPosDict[(50, 60)],
                domPosDict[(74, 60)], domPosDict[(72, 60)],
                domPosDict[(78, 60)], domPosDict[(75, 60)]
                ]
            self._convex_hull = phys_services.ExtrudedPolygon(
                detector.convert_pos_list(points)
            )

        elif isinstance(self._convex_hull, str):
            if self._convex_hull.startswith("icecube_hull_ext_"):
                extension = float(self._convex_hull.split("_")[-1])

                self._convex_hull = phys_services.ExtrudedPolygon(
                    detector.icecube_hull_points_i3, padding=extension,
                )
            else:
                self._convex_hull = getattr(detector, self._convex_hull)

        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)
