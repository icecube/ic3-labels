"""Helper functions for geometry calculations.
"""

from __future__ import print_function, division
import sys
import math
import numpy as np
import logging

from icecube import dataclasses
from icecube.phys_services import ExtrudedPolygon

from ic3_labels.labels.utils import detector
from ic3_labels.labels.utils.geometry_scipy import (
    get_axial_cylinder_intersections,
    get_cube_intersections,
    distance_to_axis_aligned_Volume,
    distance_to_icecube_hull,
    distance_to_deepcore_hull,
    get_sphere_intersection,
)


# create an arbitrary direction vector for point_is_inside
# This is defined as a global variable to reduce computational load
DEFAULT_DIRECTION = dataclasses.I3Direction(0, 0, 1)


def get_intersections(convex_hull, pos, direction):
    """Get intersections with convex hull

    Function to get the intersection points of an infinite line and the
    convex hull. The returned t's are the scaling factors for v_dir to
    get the intersection points. If t < 0 the intersection is 'behind'
    v_pos. This can be used decide whether a track is a starting track.

    Parameters
    ----------
    convex_hull : icecube.phys_services.ExtrudedPolygon
        Defines the desired convex volume.
    pos : I3Position
        A point of the line.
    direction : I3Direction
        Directional vector of the line.

    Returns
    -------
    t : array-like shape=(n_intersections)
        Scaling factors for v_dir to get the intersection points.
        Actual intersection points are v_pos + t * v_dir.
    """
    res = convex_hull.intersection(pos, direction)
    return tuple(t for t in (res.first, res.second) if math.isfinite(t))


def point_is_inside(convex_hull, pos):
    """Determine if a point is inside the convex hull.

    A default directional vector is asumend. If this track has an intersection
    in front and behind v_pos, then must v_pos be inside the hull.
    The rare case of a point inside the hull surface is treated as
    being inside the hull.

    Parameters
    ----------
    convex_hull : icecube.phys_services.ExtrudedPolygon
        Defines the convex hull.
    pos : I3Position
        The point for which to check if it's inside the convex hull.

    Returns
    -------
    is_inside : boolean
        True if the point is inside the detector.
        False if the point is outside the detector
    """

    # note: the distances are ordered and comparison against NaN results False
    res = convex_hull.intersection(pos, DEFAULT_DIRECTION)
    return res.first <= 0 and res.second >= 0


def distance_to_convex_hull(convex_hull, pos):
    """Determine closest distance of a point to the convex hull.

    Parameters
    ----------
    convex_hull : icecube.phys_services.ExtrudedPolygon
        Defines the convex hull.
    pos : I3Position
        The point for which to check the distance to the convex hull.

    Returns
    -------
    distance: float
        absolute value of closest distance from the point
        to the convex hull
        (maybe easier/better to have distance poositive
         or negative depending on whether the point is inside
         or outside. Alternatively check with point_is_inside)
    """
    raise NotImplementedError


def is_in_detector_bounds(pos, extend_boundary=60):
    """Determine whether a point is within detector bounds

    Parameters
    ----------
    pos : I3Position
        Position to be checked.
    extend_boundary : float
        Extend boundary of detector by extend_boundary

    Returns
    -------
    bool
        True if within detector bounds + extend_boundary
    """
    if not math.isfinite(extend_boundary):
        if math.isinf(extend_boundary):
            # Choose a finite value that is close to largest float
            # Note: choosing sys.float_info.max directly results in
            #       issues in ExtrudedPolygon, where the point is
            #       always evaluated as outside (even if it is
            #       inside the convex hull with the specified padding.)
            extend_boundary = sys.float_info.max / 10.0
        else:
            raise ValueError(
                f"extend_boundary must be finite, got {extend_boundary}"
            )
    convex_hull = ExtrudedPolygon(
        detector.icecube_hull_points_i3,
        padding=extend_boundary,
    )
    return point_is_inside(
        convex_hull=convex_hull,
        pos=pos,
    )
