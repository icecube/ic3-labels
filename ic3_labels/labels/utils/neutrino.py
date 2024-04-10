"""Helper functions for icecube specific labels.
"""

from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from ic3_labels.labels.utils import geometry


def get_interaction_neutrino(
    frame, primary, convex_hull=None, extend_boundary=0, sanity_check=False
):
    """Get the first neutrino daughter of a primary neutrino, that interacted
    inside the convex hull.

    The I3MCTree is traversed to find the first interaction inside the convex
    hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        If None, the IceCube detector volume is assumed.
    extend_boundary : float, optional
        Extend boundary of IceCube detector by this distance [in meters].
        This option is only used if convex_hull is None, e.g. if the IceCube
        detector is used.
    sanity_check : bool, optional
        If true, the neutrino is obtained by two different methods and cross
        checked to see if results match.

    Returns
    -------
    I3Particle, None
        Returns None if no interaction exists inside the convex hull
        Returns the found neutrino as an I3Particle.

    Raises
    ------
    ValueError
        Description
    """

    mctree = frame["I3MCTree"]

    # get first in ice neutrino
    nu_in_ice = None
    for p in mctree:
        if p.is_neutrino and p.location_type_string == "InIce":
            nu_in_ice = p
            break

    if nu_in_ice is not None:

        # check if nu_in_ice has interaction inside convex hull
        daughters = mctree.get_daughters(nu_in_ice)
        assert len(daughters) > 0, "Expected at least one daughter!"

        # check if point is inside
        if convex_hull is None:
            point_inside = geometry.is_in_detector_bounds(
                daughters[0].pos, extend_boundary=extend_boundary
            )
        else:
            point_inside = geometry.point_is_inside(
                convex_hull,
                daughters[0].pos,
            )

        if not point_inside:
            nu_in_ice = None

    # ---------------
    # Sanity Check
    # ---------------
    if sanity_check:
        nu_in_ice_rec = get_interaction_neutrino_rec(
            frame=frame,
            primary=primary,
            convex_hull=convex_hull,
            extend_boundary=extend_boundary,
        )

        if nu_in_ice_rec != nu_in_ice:
            if (
                nu_in_ice_rec is None
                or nu_in_ice is None
                or nu_in_ice_rec.id != nu_in_ice.id
                or nu_in_ice_rec.minor_id != nu_in_ice.minor_id
            ):
                raise ValueError("{} != {}".format(nu_in_ice_rec, nu_in_ice))
    # ---------------

    return nu_in_ice


def get_interaction_neutrino_rec(
    frame, primary, convex_hull=None, extend_boundary=0
):
    """Get the first neutrino daughter of a primary neutrino, that interacted
    inside the convex hull.

    The I3MCTree is traversed to find the first interaction inside the convex
    hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        If None, the IceCube detector volume is assumed.
    extend_boundary : float, optional
        Extend boundary of IceCube detector by this distance [in meters].
        This option is only used if convex_hull is None, e.g. if the IceCube
        detector is used.

    Returns
    -------
    I3Particle, None
        Returns None if no interaction exists inside the convex hull
        Returns the found neutrino as an I3Particle.
    """
    if primary is None:
        return None

    mctree = frame["I3MCTree"]

    # traverse I3MCTree until first interaction inside the convex hull is found
    daughters = mctree.get_daughters(primary)

    # No daughters found, so no interaction
    if len(daughters) == 0:
        return None

    # check if interaction point is inside
    if convex_hull is None:
        point_inside = geometry.is_in_detector_bounds(
            daughters[0].pos, extend_boundary=extend_boundary
        )
    else:
        point_inside = geometry.point_is_inside(convex_hull, daughters[0].pos)

    if point_inside:
        # interaction is inside the convex hull: neutrino found!
        if primary.is_neutrino:
            return primary
        else:
            return None

    else:
        # daughters are not inside convex hull.
        # Either one of these daughters has secondary particles which has an
        # interaction inside, or there is no interaction within the convex hull

        interaction_neutrinos = []
        for n in daughters:
            # check if this neutrino has interaction inside the convex hull
            neutrino = get_interaction_neutrino_rec(
                frame, n, convex_hull, extend_boundary
            )
            if neutrino is not None:
                interaction_neutrinos.append(neutrino)

        if len(interaction_neutrinos) == 0:
            # No neutrinos interacting in the convex hull could be found.
            return None

        if len(interaction_neutrinos) > 1:
            print(interaction_neutrinos)
            raise ValueError("Expected only one neutrino to interact!")

        # Found a neutrino that had an interaction inside the convex hull
        return interaction_neutrinos[0]
