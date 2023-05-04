import numpy as np


# some enums for CORSIKA->PDG compatibility
ParticleType = {
    'Gamma': 1,
    'PPlus': 14,
    'He4Nucleus': 402,
    'N14Nucleus': 1407,
    'O16Nucleus': 1608,
    'Al27Nucleus': 2713,
    'Fe56Nucleus': 5626,
    'NuE': 66,
    'NuEBar': 67,
    'NuMu': 68,
    'NuMuBar': 69,
    'NuTau': 133,
    'NuTauBar': 134,
    'MuMinus': 6,
    'MuPlus': 5,
}


def get_weighted_primary(frame, MCPrimary='MCPrimary'):
    """Add weighted primary to frame

    Weighted CORSIKA simulation (as well as some NuGen simulation) can have
    coincidences mixed in that should not be used to calculate weights, as they
    were chosen at "natural" frequency. Find the primary that was chosen from a
    biased spectrum, and put it in the frame.

    Parameters
    ----------
    frame : TYPE
        Description
    MCPrimary : str, optional
        Name of the primary particle to put into the frame.
    """

    MCTreeName = None
    for mctree in ['I3MCTree_preMuonProp', 'I3MCTree']:
        if (mctree in frame) and (len(frame[mctree].primaries) != 0):
            MCTreeName = mctree
            break

    if MCTreeName is None:
        return

    primaries = frame[MCTreeName].primaries

    if len(primaries) == 0:
        return

    if len(primaries) == 1:
        idx = 0

    elif 'I3MCWeightDict' in frame:
        idx = [i for i in range(len(primaries)) if primaries[i].is_neutrino]
        assert len(idx) == 0, (idx, primaries)
        idx = idx[0]

    elif 'CorsikaWeightMap' in frame:
        wmap = frame['CorsikaWeightMap']
        # Only filter by particle type if we're still using CORSIKA-style
        # codes. This is a horrendous hack that will have to be revisited
        # once PDG-coded simulation becomes more common.
        if dataclasses.I3Particle.PPlus == ParticleType['PPlus']:
            if 'PrimaryType' in wmap:
                primaries = [
                    p for p in primaries if p.type == wmap['PrimaryType']]

            elif 'ParticleType' in wmap:
                primaries = [
                    p for p in primaries if p.type == wmap['ParticleType']]

        if len(primaries) == 0:
            return

        elif len(primaries) == 1:
            idx = 0

        elif 'PrimaryEnergy' in wmap:
            prim_e = wmap['PrimaryEnergy']
            idx = int(np.nanargmin([abs(p.energy-prim_e) for p in primaries]))

        elif 'PrimarySpectralIndex' in wmap:
            prim_e = wmap['Weight']**(-1./wmap['PrimarySpectralIndex'])
            idx = int(np.nanargmin([abs(p.energy-prim_e) for p in primaries]))

        else:
            idx = 0

    frame[MCPrimary] = primaries[idx]
