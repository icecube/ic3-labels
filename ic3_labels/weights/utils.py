from ic3_labels.labels.utils import general


def get_weighted_primary(frame, MCPrimary="MCPrimary", mctree_name=None):
    """Add weighted primary to frame

    Weighted CORSIKA simulation (as well as some NuGen simulation) can have
    coincidences mixed in that should not be used to calculate weights, as they
    were chosen at "natural" frequency. Find the primary that was chosen from a
    biased spectrum, and put it in the frame.

    Parameters
    ----------
    frame : I3Frame
        The I3Frame from which to retrieve the weighted primary particle.
    MCPrimary : str, optional
        Name of the primary particle to put into the frame.
    mctree_name : str, optional
        The name of the I3MCTree to use.
        If None is provided, one of 'I3MCTree_preMuonProp', 'I3MCTree'
        will be used.
    """
    frame[MCPrimary] = general.get_weighted_primary(
        frame=frame, mctree_name=mctree_name
    )
