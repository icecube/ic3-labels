#!/usr/bin/env python
# -*- coding: utf-8 -*
'''Helper functions for geometry calculations.
'''
from __future__ import print_function, division
import numpy as np


def ray_triangle_intersection(ray_near, ray_dir, triangle):
    """
    Möller–Trumbore intersection algorithm in pure python
    Based on http://en.wikipedia.org/wiki/
    M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    The returned t's are the scaling for ray_dir do get the triangle
    intersection point. The ray is starting from ray_near and goes to
    infinity. Only positive t's are returned.

    Parameters
    ----------
    ray_near : array-like shape=(3,)
        Starting point of the ray.

    ray_dir : array-like shape=(3,)
        Directional vector of the ray.

    triangle : array-like shape=(3, 3)
        Triangle for which the interaction point should be found.

    Returns
    -------
    t : float or np.nan
        Intersection is ray_near + t * ray_dir or np.nan for no
        intersection

    """
    (v1, v2, v3) = triangle
    eps = 0.000001
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_dir, edge2)
    det = edge1.dot(pvec)
    if abs(det) < eps:
        return np.nan
    inv_det = 1. / det
    tvec = ray_near - v1
    u = tvec.dot(pvec) * inv_det
    if u < 0. or u > 1.:
        return np.nan
    qvec = np.cross(tvec, edge1)
    v = ray_dir.dot(qvec) * inv_det
    if v < 0. or u + v > 1.:
        return np.nan
    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return np.nan
    return t


def get_intersections(convex_hull, v_pos, v_dir, eps=1e-4):
    '''Function to get the intersection points of an infinite line and the
    convex hull. The returned t's are the scaling factors for v_dir to
    get the intersection points. If t < 0 the intersection is 'behind'
    v_pos. This can be used decide whether a track is a starting track.

    Parameters
    ----------
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    v_pos : array-like shape=(3,)
        A point of the line.

    v_dir : array-like shape=(3,)
        Directional vector of the line.

    eps : float or None
        Min distance between intersection points to be treated as
        different points.

    Returns
    -------
    t : array-like shape=(n_intersections)
        Scaling factors for v_dir to get the intersection points.
        Actual intersection points are v_pos + t * v_dir.
    '''
    if not isinstance(v_pos, np.ndarray):
        v_pos = np.array(v_pos)
    if not isinstance(v_dir, np.ndarray):
        v_dir = np.array(v_dir)
    t_s = [ray_triangle_intersection(v_pos,
                                     v_dir,
                                     convex_hull.points[simp])
           for simp in convex_hull.simplices]
    t_s = np.array(t_s)
    t_s = t_s[np.isfinite(t_s)]
    if len(t_s) != 2:   # A line should have at max 2 intersection points
                        # with a convex hull
        t_s_back = [ray_triangle_intersection(v_pos,
                                              -v_dir,
                                              convex_hull.points[simp])
                    for simp in convex_hull.simplices]
        t_s_back = np.array(t_s_back)
        t_s_back = t_s_back[np.isfinite(t_s_back)]
        t_s = np.hstack((t_s, t_s_back * (-1.)))
    if isinstance(eps, float):  # Remove similar intersections
        if eps >= 0.:
            t_selected = []
            intersections = []
            for t_i in t_s:
                intersection_i = v_pos + t_i * v_dir
                distances = [np.linalg.norm(intersection_i - intersection_j)
                             for intersection_j in intersections]
                if not (np.array(distances) < eps).any():
                    t_selected.append(t_i)
                    intersections.append(intersection_i)
            t_s = np.array(t_selected)
    return t_s


def point_is_inside(convex_hull,
                    v_pos,
                    default_v_dir=np.array([0., 0., 1.]),
                    eps=1e-4):
    '''Function to determine if a point is inside the convex hull.
    A default directional vector is asumend. If this track has an intersection
    in front and behind v_pos, then must v_pos be inside the hull.
    The rare case of a point inside the hull surface is treated as
    being inside the hull.

    Parameters
    ----------
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    v_pos : array-like shape=(3,)
        Position.

    default_v_dir : array-like shape=(3,), optional (default=[0, 0, 1])
        See get_intersections()

    eps : float or None
        See get_intersections()

    Returns
    -------
    is_inside : boolean
        True if the point is inside the detector.
        False if the point is outside the detector
    '''
    t_s = get_intersections(convex_hull, v_pos, default_v_dir, eps)
    return len(t_s) == 2 and (t_s >= 0).any() and (t_s <= 0).any()


def distance_to_convex_hull(convex_hull, v_pos):
    '''Function to determine the closest distance of a point
            to the convex hull.

    Parameters
    ----------
    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    v_pos : array-like shape=(3,)
        Position.

    Returns
    -------
    distance: float
        absolute value of closest distance from the point
        to the convex hull
        (maybe easier/better to have distance poositive
         or negativ depending on wheter the point is inside
         or outside. Alernatively check with point_is_inside)
    '''
    raise NotImplementedError


def get_closest_point_on_edge(edge_point1, edge_point2, point):
    '''Function to determine the closest point
            on an edge defined by the two points
            edge_point1 and edge_point2

    Parameters
    ----------

    edge_point1 : array-like shape=(3,)
        First edge point .

    edge_point2 : array-like shape=(3,)
        Second edge point .

    point : array-like shape=(3,)
        point of which to find the distance
        to the edge

    Returns
    -------
    distance: array-like shape=(3,)
        closest point on the edge
    '''
    if edge_point1 == edge_point2:
        return ValueError('Points do not define line.')
    A = np.array(edge_point1)
    B = np.array(edge_point2)
    P = np.array(point)
    vec_edge = B - A
    vec_point = P - A
    norm_edge = np.linalg.norm(vec_edge)
    t_projection = np.dot(vec_edge, vec_point) / (norm_edge**2)

    t_clipped = min(1, max(t_projection, 0))
    closest_point = A + t_clipped*vec_edge

    return closest_point


def get_distance_to_edge(edge_point1, edge_point2, point):
    '''Function to determine the closest distance of a point
            to an edge defined by the two points
            edge_point1 and edge_point2

    Parameters
    ----------

    edge_point1 : array-like shape=(3,)
        First edge point .

    edge_point2 : array-like shape=(3,)
        Second edge point .

    point : array-like shape=(3,)
        point of which to find the distance
        to the edge

    Returns
    -------
    distance: float
    '''
    closest_point = get_closest_point_on_edge(edge_point1,
                                              edge_point2, point)
    distance = np.linalg.norm(closest_point - point)
    return distance


def get_edge_intersection(edge_point1, edge_point2, point):
    '''Returns t:
        edge_point1 + u*(edge_point2-edge_point1)
        =
        point + t * (0, 1, 0)
        if u is within [0,1].
        [Helper Function to find out if point
         is inside the icecube 2D Polygon]

    Parameters
    ----------

    edge_point1 : array-like shape=(3,)
        First edge point .

    edge_point2 : array-like shape=(3,)
        Second edge point .

    point : array-like shape=(3,)
        point of which to find the distance
        to the edge

    Returns
    -------
    t: float.
        If intersection is within edge
        othwise returns nan.
    '''
    if edge_point1 == edge_point2:
        return ValueError('Points do not define line.')
    A = np.array(edge_point1)
    B = np.array(edge_point2)
    P = np.array(point)
    vec_edge = B - A
    vec_point = P - A

    u = vec_point[0] / vec_edge[0]
    t = u * vec_edge[1] - vec_point[1]

    if u > -1e-8 and u < 1 + 1e-8:
        return t
    return float('nan')


def distance_to_axis_aligned_Volume(pos, points, z_min, z_max):
    '''Function to determine the closest distance of a point
       to the edge of a Volume defined by z_zmin,z_max and a
       2D-Polygon described through a List of counterclockwise
       points.

    Parameters
    ----------

    pos :I3Position
        Position.
    points : array-like shape=(?,3)
        List of counterclockwise points
        describing the polygon of the volume
        in the x-y-plane
    z_max : float
        Top layer of IceCube-Doms
    z_min : float
        Bottom layer of IceCube-Doms

    Returns
    -------
    distance: float
        closest distance from the point
        to the edge of the volume
        negativ if point is inside,
        positiv if point is outside
    '''
    no_of_points = len(points)
    edges = [(points[i], points[(i + 1) % (no_of_points)])
             for i in range(no_of_points)]
    xy_distance = float('inf')
    list_of_ts = []

    for edge in edges:
        x = (edge[0][0], edge[1][0])
        y = (edge[0][1], edge[1][1])
        distance = get_distance_to_edge(edge[0], edge[1],
                                        [pos[0], pos[1], 0])
        t = get_edge_intersection(edge[0], edge[1],
                                  [pos[0], pos[1], 0])
        if not np.isnan(t):
            list_of_ts.append(t)
        if distance < xy_distance:
            xy_distance = distance
    is_inside_xy = False
    if len(list_of_ts) == 2:
        # u's are pos and negativ
        if list_of_ts[0]*list_of_ts[1] < 0:
            is_inside_xy = True
        # point is exactly on border
        elif len([t for t in list_of_ts if t == 0]) == 1:
            is_inside_xy = True

    # ---- Calculate z_distance
    is_inside_z = False
    if pos[2] < z_min:
        # underneath detector
        z_distance = z_min - pos[2]
    elif pos[2] < z_max:
        # same height
        is_inside_z = True
        z_distance = min(pos[2] - z_min, z_max - pos[2])
    else:
        # above detector
        z_distance = pos[2] - z_max

    # ---- Combine distances
    if is_inside_z:
        if is_inside_xy:
            # inside detector
            distance = - min(xy_distance, z_distance)
        else:
            distance = xy_distance
    else:
        if is_inside_xy:
            distance = z_distance
        else:
            distance = np.sqrt(z_distance**2 + xy_distance**2)

    return distance


def distance_to_icecube_hull(pos, z_min=-502, z_max=501):
    '''Function to determine the closest distance of a point
            to the icecube hull. This is only
            an approximate distance.

    Parameters
    ----------

    pos :I3Position
        Position.
    z_max : float
        Top layer of IceCube-Doms
    z_min : float
        Bottom layer of IceCube-Doms

    Returns
    -------
    distance: float
        closest distance from the point
        to the icecube hull
        negativ if point is inside,
        positiv if point is outside
    '''
    points = [
           [-570.90002441, -125.13999939, 0],  # string 31
           [-256.14001465, -521.08001709, 0],  # string 1
           [ 361.        , -422.82998657, 0],  # string 6
           [ 576.36999512,  170.91999817, 0],  # string 50
           [ 338.44000244,  463.72000122, 0],  # string 74
           [ 101.04000092,  412.79000854, 0],  # string 72
           [  22.11000061,  509.5       , 0],  # string 78
           [-347.88000488,  451.51998901, 0],  # string 75
            ]
    return distance_to_axis_aligned_Volume(pos, points, z_min, z_max)


def distance_to_deepcore_hull(pos, z_min=-502, z_max=188):
    '''Function to determine the closest distance of a point
            to the deep core hull. This is only
            an approximate distance.

    Parameters
    ----------

    pos :I3Position
        Position.
    z_max : float
        Top layer of IceCube-Doms
    z_min : float
        Bottom layer of IceCube-Doms

    Returns
    -------
    distance: float
        closest distance from the point
        to the icecube hull
        negativ if point is inside,
        positiv if point is outside
    '''
    points = [
           [-77.80000305175781, -54.33000183105469, 0],  # string 35
           [1.7100000381469727, -150.6300048828125, 0],  # string 26
           [124.97000122070312, -131.25, 0],  # string 27
           [194.33999633789062, -30.920000076293945, 0],  # string 37
           [90.48999786376953, 82.3499984741211, 0],  # string 46
           [-32.959999084472656, 62.439998626708984, 0],  # string 45
            ]
    return distance_to_axis_aligned_Volume(pos, points, z_min, z_max)


def is_in_detector_bounds(pos, extend_boundary=60):
    '''Function to determine whether a point is still
        withtin detector bounds

    Parameters
    ----------
    pos : I3Position
        Position to be checked.

    extend_boundary : float
        Extend boundary of detector by extend_boundary

    Returns
    -------
    is_inside : bool
        True if within detector bounds + extend_boundary
    '''
    distance = distance_to_icecube_hull(pos)
    return distance - extend_boundary <= 0
