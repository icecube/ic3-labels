'''Helper functions for icecube specific labels.
'''
from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils.neutrino import get_interaction_neutrino


def get_cascade_energy_deposited(frame, convex_hull, cascade):
    '''Function to get the total energy a cascade deposited
        in the volume defined by the convex hull. Assumes
        that Cascades lose all of their energy in the convex
        hull if their vertex is in the hull. Otherwise the enrgy
        deposited by a cascade will be 0.
        (naive: There is possibly a better solution to this)

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    cascade : I3Particle
        Cascade.

    Returns
    -------
    energy : float
        Deposited Energy.
    '''
    if geometry.point_is_inside(
            convex_hull,
            (cascade.pos.x, cascade.pos.y, cascade.pos.z)
            ):
        # if inside convex hull: add all of the energy
        return cascade.energy
    else:
        return 0.0


def get_cascade_of_primary_nu(frame, primary,
                              convex_hull=None,
                              extend_boundary=200):
    """Get cascade of a primary particle.

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
        Returns None if no cascade interaction exists inside the convex hull
        Returns the found cascade as an I3Particle.
        The returned I3Particle will have the vertex, direction and total
        visible energy of the cascade and the type of the neutrino that
        interacted in the detector. The deoposited energy is defined here
        as the sum of the energies of the daugther particles, unless these are
        neutrinos.
        (Does not account for energy carried away by neutrinos of tau decay)
    """
    neutrino = get_interaction_neutrino(frame, primary,
                                        convex_hull=convex_hull,
                                        extend_boundary=extend_boundary,
                                        sanity_check=True)

    if neutrino is None or not neutrino.is_neutrino:
        return None

    mctree = frame['I3MCTree']

    # traverse I3MCTree until first interaction inside the convex hull is found
    daughters = mctree.get_daughters(neutrino)

    # -----------------------
    # Sanity Checks
    # -----------------------
    assert len(daughters) > 0, 'Expected at least one daughter!'

    # check if point is inside
    if convex_hull is None:
        point_inside = geometry.is_in_detector_bounds(
                            daughters[0].pos, extend_boundary=extend_boundary)
    else:
        point_inside = geometry.point_is_inside(convex_hull,
                                                (daughters[0].pos.x,
                                                 daughters[0].pos.y,
                                                 daughters[0].pos.z))
    assert point_inside, 'Expected interaction to be inside defined volume!'
    # -----------------------

    # interaction is inside the convex hull: cascade found!

    # get cascade
    cascade = dataclasses.I3Particle(neutrino)
    cascade.dir = dataclasses.I3Direction(primary.dir)
    cascade.pos = dataclasses.I3Position(daughters[0].pos)
    cascade.time = daughters[0].time

    # sum up energies for daughters if not neutrinos
    # tau can immediately decay in neutrinos which carry away energy
    # that would not be visible, this is currently not accounted for
    deposited_energy = 0.
    for d in daughters:
        if d.is_neutrino:
            # skip neutrino: the energy is not visible
            continue
        deposited_energy += d.energy

    cascade.energy = deposited_energy
    return cascade
