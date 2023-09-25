""" Convex Hulls for IceCube Detector
"""
from scipy.spatial import ConvexHull
import numpy as np


icecube_hull_points = [
    [-570.90002441, -125.13999939, -500],  # string 31
    [-256.14001465, -521.08001709, -500],  # string 1
    [ 361.        , -422.82998657, -500],  # string 6
    [ 576.36999512,  170.91999817, -500],  # string 50
    [ 338.44000244,  463.72000122, -500],  # string 74
    [ 101.04000092,  412.79000854, -500],  # string 72
    [  22.11000061,  509.5       , -500],  # string 78
    [-347.88000488,  451.51998901, -500],  # string 75
    [-570.90002441, -125.13999939, -500],  # string 31
    [-256.14001465, -521.08001709, 500],  # string 1
    [ 361.        , -422.82998657, 500],  # string 6
    [ 576.36999512,  170.91999817, 500],  # string 50
    [ 338.44000244,  463.72000122, 500],  # string 74
    [ 101.04000092,  412.79000854, 500],  # string 72
    [  22.11000061,  509.5       , 500],  # string 78
    [-347.88000488,  451.51998901, 500],  # string 75
    [-570.90002441, -125.13999939, 500],  # string 31
]
icecube_hull = ConvexHull(icecube_hull_points)

# add extensions around IceCube
def get_extended_convex_hull(extension: float) -> ConvexHull:
    """
    Returns the ConvexHull (scipy.spacial.ConvexHull) 
    of the detector extended by `extension` meters
    """ 
    icecube_hull_points_i = []
    for x, y, z in icecube_hull_points:

        # extend along z-axis
        if z < 0:
            z_ext = z - extension
        else:
            z_ext = z + extension

        # extend radially outward
        # Note: this is a decent approximation to a proper extended ConvexHull
        pos_vec = np.array([x, y])
        pos_vec_ext = pos_vec + extension * pos_vec / np.linalg.norm(pos_vec)

        icecube_hull_points_i.append([pos_vec_ext[0], pos_vec_ext[1], z_ext])

    return ConvexHull(icecube_hull_points_i)

# Assuming dust layer to be at -150m to -50m
icecube_hull_upper = ConvexHull([
    [-570.90002441, -125.13999939, -50],  # string 31
    [-256.14001465, -521.08001709, -50],  # string 1
    [ 361.        , -422.82998657, -50],  # string 6
    [ 576.36999512,  170.91999817, -50],  # string 50
    [ 338.44000244,  463.72000122, -50],  # string 74
    [ 101.04000092,  412.79000854, -50],  # string 72
    [  22.11000061,  509.5       , -50],  # string 78
    [-347.88000488,  451.51998901, -50],  # string 75
    [-570.90002441, -125.13999939, -50],  # string 31
    [-256.14001465, -521.08001709, 500],  # string 1
    [ 361.        , -422.82998657, 500],  # string 6
    [ 576.36999512,  170.91999817, 500],  # string 50
    [ 338.44000244,  463.72000122, 500],  # string 74
    [ 101.04000092,  412.79000854, 500],  # string 72
    [  22.11000061,  509.5       , 500],  # string 78
    [-347.88000488,  451.51998901, 500],  # string 75
    [-570.90002441, -125.13999939, 500],  # string 31
])

icecube_hull_lower = ConvexHull([
    [-570.90002441, -125.13999939, -500],  # string 31
    [-256.14001465, -521.08001709, -500],  # string 1
    [ 361.        , -422.82998657, -500],  # string 6
    [ 576.36999512,  170.91999817, -500],  # string 50
    [ 338.44000244,  463.72000122, -500],  # string 74
    [ 101.04000092,  412.79000854, -500],  # string 72
    [  22.11000061,  509.5       , -500],  # string 78
    [-347.88000488,  451.51998901, -500],  # string 75
    [-570.90002441, -125.13999939, -500],  # string 31
    [-256.14001465, -521.08001709, -150],  # string 1
    [ 361.        , -422.82998657, -150],  # string 6
    [ 576.36999512,  170.91999817, -150],  # string 50
    [ 338.44000244,  463.72000122, -150],  # string 74
    [ 101.04000092,  412.79000854, -150],  # string 72
    [  22.11000061,  509.5       , -150],  # string 78
    [-347.88000488,  451.51998901, -150],  # string 75
    [-570.90002441, -125.13999939, -150],  # string 31
])

# This is is convex hull around IceCube minus 1 outer layer and 3 in z
icecube_veto_hull_m1 = ConvexHull([
    [-447.74, -113.13, -450],  # string 32, DOM 57 (approx z)
    [-211.35, -404.48, -450],  # string 8, DOM 57 (approx z)
    [ 282.18, -325.74, -450],  # string 12, DOM 57 (approx z)
    [ 472.05,  127.9 , -450],  # string 49, DOM 57 (approx z)
    [ 303.41,  335.64, -450],  # string 66, DOM 57 (approx z)
    [ -21.97,  393.24, -450],  # string 71, DOM 57 (approx z)
    [-268.9 ,  354.24, -450],  # string 69, DOM 57 (approx z)
    [-447.74, -113.13, -450],  # string 32, DOM 57 (approx z)
    [-447.74, -113.13, 450],  # string 32, DOM 4 (approx z)
    [-211.35, -404.48, 450],  # string 8, DOM 4 (approx z)
    [ 282.18, -325.74, 450],  # string 12, DOM 4 (approx z)
    [ 472.05,  127.9 , 450],  # string 49, DOM 4 (approx z)
    [ 303.41,  335.64, 450],  # string 66, DOM 4 (approx z)
    [ -21.97,  393.24, 450],  # string 71, DOM 4 (approx z)
    [-268.9 ,  354.24, 450],  # string 69, DOM 4 (approx z)
    [-447.74, -113.13, 450],  # string 32, DOM 4 (approx z)
])

# This is is convex hull around IceCube minus 2 outer layers and 6 in z
icecube_veto_hull_m2 = ConvexHull([
    [-324.39,  -93.43, -400],  # string 33, DOM 54 (approx z)
    [-166.4 , -287.79, -400],  # string 16, DOM 54 (approx z)
    [ 210.47, -209.77, -400],  # string 19, DOM 54 (approx z)
    [ 330.03,  127.2 , -400],  # string 48, DOM 54 (approx z)
    [ 174.47,  315.54, -400],  # string 65, DOM 54 (approx z)
    [-189.98,  257.42, -400],  # string 62, DOM 54 (approx z)
    [-324.39,  -93.43, -400],  # string 33, DOM 54 (approx z)
    [-324.39,  -93.43, 400],  # string 33, DOM 7 (approx z)
    [-166.4 , -287.79, 400],  # string 16, DOM 7 (approx z)
    [ 210.47, -209.77, 400],  # string 19, DOM 7 (approx z)
    [ 330.03,  127.2 , 400],  # string 48, DOM 7 (approx z)
    [ 174.47,  315.54, 400],  # string 65, DOM 7 (approx z)
    [-189.98,  257.42, 400],  # string 62, DOM 7 (approx z)
    [-324.39,  -93.43, 400],  # string 33, DOM 7 (approx z)
])
