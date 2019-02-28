#!/usr/bin/env python
# -*- coding: utf-8 -*
''' I3Module Base Class
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray

from scipy.spatial import ConvexHull


class MCLabelsBase(icetray.I3ConditionalModule):

    """Base Module class for MC labels.
    """

    def __init__(self, context):
        super(MCLabelsBase, self).__init__(self, context)
        self.AddParameter("PulseMapString", "Name of pulse map to use.",
                          'InIcePulses')
        self.AddParameter("PrimaryKey", "Name of the primary.", 'MCPrimary')
        self.AddParameter("OutputKey", "Save labels to this frame key.",
                          'LabelsDeepLearning')

    def Configure(self):
        self._pulse_map_string = self.GetParameter("PulseMapString")
        self._primary_key = self.GetParameter("PrimaryKey")
        self._output_key = self.GetParameter("OutputKey")

    def Geometry(self, frame):
        geoMap = frame['I3Geometry'].omgeo
        domPosDict = {(i[0][0], i[0][1]): (i[1].position.x,
                                           i[1].position.y,
                                           i[1].position.z)
                      for i in geoMap if i[1].omtype.name == 'IceCube'}
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
        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)