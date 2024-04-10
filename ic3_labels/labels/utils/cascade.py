"""Helper functions for icecube specific labels.
"""

from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, simclasses
from icecube.icetray.i3logging import log_warn

# Try to import ShowerParameters from I3SimConstants
try:
    from icecube.sim_services import I3SimConstants

    ShowerParameters = I3SimConstants.ShowerParameters

except (ImportError, AttributeError):
    try:
        from icecube.sim_services import ShowerParameters
    except ImportError:
        print("Can not include 'ShowerParameters' from icecube.sim_services")
        print("Using custom python module instead.")
        from ic3_labels.labels.utils.shower_parameters import ShowerParameters

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils.neutrino import get_interaction_neutrino


EMTypes = [
    dataclasses.I3Particle.ParticleType.EMinus,
    dataclasses.I3Particle.ParticleType.EPlus,
    dataclasses.I3Particle.ParticleType.Brems,
    dataclasses.I3Particle.ParticleType.DeltaE,
    dataclasses.I3Particle.ParticleType.PairProd,
    dataclasses.I3Particle.ParticleType.Gamma,
    # Pi0 decays to 2 gammas and produce EM showers
    dataclasses.I3Particle.ParticleType.Pi0,
    dataclasses.I3Particle.ParticleType.EMinus,
    dataclasses.I3Particle.ParticleType.EMinus,
]

HadronTypes = [
    dataclasses.I3Particle.ParticleType.Hadrons,
    dataclasses.I3Particle.ParticleType.Neutron,
    dataclasses.I3Particle.ParticleType.PiPlus,
    dataclasses.I3Particle.ParticleType.PiMinus,
    dataclasses.I3Particle.ParticleType.K0_Long,
    dataclasses.I3Particle.ParticleType.KPlus,
    dataclasses.I3Particle.ParticleType.KMinus,
    dataclasses.I3Particle.ParticleType.PPlus,
    dataclasses.I3Particle.ParticleType.PMinus,
    dataclasses.I3Particle.ParticleType.K0_Short,
    dataclasses.I3Particle.ParticleType.Eta,
    dataclasses.I3Particle.ParticleType.Lambda,
    dataclasses.I3Particle.ParticleType.SigmaPlus,
    dataclasses.I3Particle.ParticleType.Sigma0,
    dataclasses.I3Particle.ParticleType.SigmaMinus,
    dataclasses.I3Particle.ParticleType.Xi0,
    dataclasses.I3Particle.ParticleType.XiMinus,
    dataclasses.I3Particle.ParticleType.OmegaMinus,
    dataclasses.I3Particle.ParticleType.NeutronBar,
    dataclasses.I3Particle.ParticleType.LambdaBar,
    dataclasses.I3Particle.ParticleType.SigmaMinusBar,
    dataclasses.I3Particle.ParticleType.Sigma0Bar,
    dataclasses.I3Particle.ParticleType.SigmaPlusBar,
    dataclasses.I3Particle.ParticleType.Xi0Bar,
    dataclasses.I3Particle.ParticleType.XiPlusBar,
    dataclasses.I3Particle.ParticleType.OmegaPlusBar,
    dataclasses.I3Particle.ParticleType.DPlus,
    dataclasses.I3Particle.ParticleType.DMinus,
    dataclasses.I3Particle.ParticleType.D0,
    dataclasses.I3Particle.ParticleType.D0Bar,
    dataclasses.I3Particle.ParticleType.DsPlus,
    dataclasses.I3Particle.ParticleType.DsMinusBar,
    dataclasses.I3Particle.ParticleType.LambdacPlus,
    dataclasses.I3Particle.ParticleType.WPlus,
    dataclasses.I3Particle.ParticleType.WMinus,
    dataclasses.I3Particle.ParticleType.Z0,
    dataclasses.I3Particle.ParticleType.NuclInt,
]


def get_interaction_extension_length(frame, primary):
    """Get the extension length of the interaction/cascade of the first
    interaction of the primary particle.

    Parameters
    ----------
    frame : current frame
        Needed to retrieve I3MCTree
    primary : I3Particle
        The particle for which to calculate the extension length of the
        first interaction.

    Returns
    -------
    float
        The maximum extension length.
    """
    tree = frame["I3MCTree"]
    daughters = tree.get_daughters(primary)

    assert len(daughters) > 0, "Expected at least 1 daughter"

    vertex = daughters[0].pos

    extension = get_extension_from_vertex(
        frame, primary, vertex, consider_particle_length=False
    )
    return (vertex - extension).magnitude


def get_interaction_extension_pos(frame, primary):
    """Get the maximum extension position of the interaction/cascade of the
    first interaction of the primary particle.

    Parameters
    ----------
    frame : current frame
        Needed to retrieve I3MCTree
    primary : I3Particle
        The particle for which to calculate the extension length of the
        first interaction.

    Returns
    -------
    I3Position
        The position of the maximum extension.
    """
    tree = frame["I3MCTree"]
    daughters = tree.get_daughters(primary)

    assert len(daughters) >= 2, "Expected at least 2 daughters"

    vertex = daughters[0].pos

    return get_extension_from_vertex(
        frame, primary, vertex, consider_particle_length=False
    )


def get_extension_from_vertex(
    frame, particle, vertex, consider_particle_length=True
):
    """Get the maximum extension of a particle or any of its daughter particles
    in regard to the given vertex.

    Helper-function for 'get_interaction_extension'

    Parameters
    ----------
    frame : current frame
        Needed to retrieve I3MCTree
    particle : I3Particle
        The particle for which to calculate the maximum extension length.
        This particle should be one of the daughter particles after the
        interaction vertex.
    vertex : I3Position
        The vertex to which the maximum extension is to be calculated.
    consider_particle_length : bool, optional
        If True, consider the length of the particle itself.

    Returns
    -------
    I3Position
        The position of the maximum extension.
    """
    tree = frame["I3MCTree"]
    daughters = tree.get_daughters(particle)

    if consider_particle_length:
        if np.isfinite(particle.length) and particle.length > 0:
            particle_end_pos = particle.pos + particle.dir * particle.length
        else:
            particle_end_pos = particle.pos

        max_distance = (vertex - particle_end_pos).magnitude
        max_extension = particle_end_pos
    else:
        max_distance = 0.0
        max_extension = None

    for d in daughters:
        extension = get_extension_from_vertex(frame, d, vertex)

        # calculate distance to vertex
        dist = (vertex - extension).magnitude
        if dist >= max_distance:
            # found new furthest extension
            max_distance = dist
            max_extension = extension

    return max_extension


def convert_to_em_equivalent(cascade):
    """Get electro-magnetic (EM) equivalent energy of a given cascade.

    Note: this is only an average expected EM equivalent. Possible existing
    daughter particles in the I3MCTree are not taken into account!

    Parameters
    ----------
    cascade : I3Particle
        The cascade.

    Returns
    -------
    float
        The EM equivalent energy of the given cascade.
    """
    # scale energy of cascade to EM equivalent
    em_scale = ShowerParameters(cascade.type, cascade.energy).emScale
    return cascade.energy * em_scale


def get_cascade_em_equivalent(mctree, cascade_primary):
    """Get electro-magnetic (EM) equivalent energy of a given cascade.

    Recursively walks through daughters of a provided cascade primary and
    collects EM equivalent energy.
    Note: muons and taus are added completely as EM equivalent energy!
    This disregards the fact that a tau can for instance decay and the neutrino
    may carry away a big portion of energy

    Parameters
    ----------
    mctree : I3MCTree
        The current I3MCTree
    cascade_primary : I3Particle
        The cascade primary particle.

    Returns
    -------
    float
        The total EM equivalent energy of the given cascade.
    float
        The total EM equivalent energy of the EM cascade.
    float
        The total EM equivalent energy of the hadronic cascade.
    float
        The total EM equivalent energy in muons and taus (tracks).
    """

    daughters = mctree.get_daughters(cascade_primary)

    # ---------------------------------
    # stopping conditions for recursion
    # ---------------------------------
    if (
        cascade_primary.location_type
        != dataclasses.I3Particle.LocationType.InIce
    ):
        # skip particles that are way outside of the detector volume
        return 0.0, 0.0, 0.0, 0.0

    # check if we have a muon or tau
    if cascade_primary.type in [
        dataclasses.I3Particle.ParticleType.MuMinus,
        dataclasses.I3Particle.ParticleType.MuPlus,
        dataclasses.I3Particle.ParticleType.TauMinus,
        dataclasses.I3Particle.ParticleType.TauPlus,
    ]:
        # For simplicity we will assume that all energy is deposited.
        # Note: this is wrong for instance for taus that decay where the
        # neutrino will carry away a large fraction of the energy
        return cascade_primary.energy, 0.0, 0.0, cascade_primary.energy

    if len(daughters) == 0:

        if cascade_primary.is_neutrino:
            # skip neutrino: the energy is not visible
            return 0.0, 0.0, 0.0, 0.0

        else:

            # get EM equivalent energy
            energy = convert_to_em_equivalent(cascade_primary)

            # EM energy
            if cascade_primary.type in EMTypes:
                return energy, energy, 0.0, 0.0

            # Hadronic energy
            elif cascade_primary.type in HadronTypes:
                return energy, 0.0, energy, 0.0

            else:
                log_warn(
                    "Unknown particle type: {}. Assuming hadron!".format(
                        cascade_primary.type
                    )
                )
                return energy, 0.0, energy, 0.0

    # ---------------------------------

    # collect energy from hadronic, em, and tracks
    energy_total = 0.0
    energy_em = 0.0
    energy_hadron = 0.0
    energy_track = 0.0

    # recursively walk through daughters and accumulate energy
    for daughter in daughters:

        # get energy depositions of particle and its daughters
        e_total, e_em, e_hadron, e_track = get_cascade_em_equivalent(
            mctree, daughter
        )

        # CMC splits up hadronic cascades to segments of electrons
        # In other words: if the cascade primary is a hadron, the daughter
        # particles need to contribute to the hadronic component of the shower
        if cascade_primary.type in HadronTypes:
            e_hadron += e_em
            e_em = 0

        # accumulate energies
        energy_total += e_total
        energy_em += e_em
        energy_hadron += e_hadron
        energy_track += e_track

    return energy_total, energy_em, energy_hadron, energy_track


def get_cascade_energy_deposited(frame, convex_hull, cascade):
    """Function to get the total energy a cascade deposited
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
    """
    if geometry.point_is_inside(convex_hull, cascade.pos):
        # if inside convex hull: add all of the energy
        return convert_to_em_equivalent(cascade)
    else:
        return 0.0


def get_cascade_of_primary_nu(
    frame, primary, convex_hull=None, extend_boundary=200, sanity_check=False
):
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
    sanity_check : bool, optional
        If true, the neutrino is obtained by two different methods and cross
        checked to see if results match.

    Returns
    -------
    I3Particle, None
        Returns None if no cascade interaction exists inside the convex hull
        Returns the found cascade as an I3Particle.
        The returned I3Particle will have the vertex, direction and total
        visible energy (EM equivalent) of the cascade. In addition it will
        have the type of the interaction NEUTRINO. The visible energy is
        defined here as the sum of the EM equivalent energies of the  daughter
        particles, unless these are neutrinos.  Only energies of particles
        that have 'InIce' location_type are considered. This meas that
        energies from hadron daughter particles get converted to the EM
        equivalent energy.
        (Does not account for energy carried away by neutrinos of tau decay)
    float
        The total EM equivalent energy of the EM cascade.
    float
        The total EM equivalent energy of the hadronic cascade.
    float
        The total EM equivalent energy in muons and taus (tracks).
    """
    neutrino = get_interaction_neutrino(
        frame,
        primary,
        convex_hull=convex_hull,
        extend_boundary=extend_boundary,
        sanity_check=sanity_check,
    )

    if neutrino is None or not neutrino.is_neutrino:
        return None, None, None, None

    mctree = frame["I3MCTree"]

    # traverse I3MCTree until first interaction inside the convex hull is found
    daughters = mctree.get_daughters(neutrino)

    # -----------------------
    # Sanity Checks
    # -----------------------
    assert len(daughters) > 0, "Expected at least one daughter!"

    # check if point is inside
    if convex_hull is None:
        point_inside = geometry.is_in_detector_bounds(
            daughters[0].pos, extend_boundary=extend_boundary
        )
    else:
        point_inside = geometry.point_is_inside(convex_hull, daughters[0].pos)

    assert point_inside, "Expected interaction to be inside defined volume!"
    # -----------------------

    # interaction is inside the convex hull/extension boundary: cascade found!

    # get cascade
    cascade = dataclasses.I3Particle(neutrino)
    cascade.shape = dataclasses.I3Particle.ParticleShape.Cascade
    cascade.dir = dataclasses.I3Direction(primary.dir)
    cascade.pos = dataclasses.I3Position(daughters[0].pos)
    cascade.time = daughters[0].time
    cascade.length = get_interaction_extension_length(frame, neutrino)

    # sum up energies for daughters if not neutrinos
    # tau can immediately decay in neutrinos which carry away energy
    # that would not be visible, this is currently not accounted for
    e_total, e_em, e_hadron, e_track = get_cascade_em_equivalent(
        mctree, neutrino
    )

    cascade.energy = e_total
    return cascade, e_em, e_hadron, e_track
