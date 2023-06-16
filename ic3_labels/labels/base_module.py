''' I3Module Base Class
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray

from scipy.spatial import ConvexHull

from ic3_labels.labels.utils import detector

import re

EXTENDED_HULL_RE = re.compile("icecube_hull_ext_([+-]?\d+)")

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
            "The the convex hull to use. Must either be an instance of "
            "scipy.spatial.ConvexHull or a string defining a convex hull "
            "in ic3_labels.labels.utils.detector."
            "Or a string named icecube_hull_ext_{extensin in meters}."
            "If None is provided, a "
            "convex hull around the outer IceCube strings will be constructed "
            "based on the given Geometry file.",
            None,
        )
        self.AddParameter("OutputKey", "Save labels to this frame key.",
                          'LabelsDeepLearning')

    def Configure(self):
        self._pulse_map_string = self.GetParameter("PulseMapString")
        self._mcpe_series_map_name = self.GetParameter("MCPESeriesMapName")
        self._primary_key = self.GetParameter("PrimaryKey")
        self._convex_hull = self.GetParameter("ConvexHull")
        self._output_key = self.GetParameter("OutputKey")

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
            self._convex_hull = ConvexHull(points)

        elif isinstance(self._convex_hull, str):
            if (match := EXTENDED_HULL_RE.match(self._convex_hull)) is not None:
                extension = int(match.group(1))
                self._convex_hull = detector.get_extended_convex_hull(extension)
            else:
                self._convex_hull = getattr(detector, self._convex_hull)

        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)
