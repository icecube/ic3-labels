#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import numpy as np
from glob import glob
from icecube import dataio, icetray
from icecube import MuonGun
from icecube import dataclasses
from icecube.icetray.i3logging import log_info as log


def harvest_generators(infiles, n_files=-1, equal_generators=True):
    '''
    Harvest serialized generator configurations from a set of I3 files.

    Parameters
    ----------
    infiles : list of str
        List of input files
    n_files : int, optional
        If equal_generators is True, the n_files will be used to calculate
        normalization for the MuonGun weights.
    equal_generators : bool, optional
        If True, it is assumed that all files in the infiles list have the same
        S frame, e.g. the same MuonGun.GenerationProbability.

    Returns
    -------
    I3 generator
        The MuonGun flux generator
    int
        Number of found generators, e.g. the used n_files for weighting

    Raises
    ------
    ValueError
        If equal_generators is true, but no number of files is explicitly
        given. This is needed, since only the first file of the input files
        will be used.
    '''
    if not isinstance(infiles, list):
        infiles = [infiles]

    generator = None
    n_generators = 0
    if equal_generators:
        log('Assuming all generator objects of the given file list are equal!')
        if n_files == -1 or not isinstance(n_files, int):
            raise ValueError('For equal generators the number of files needs '
                             'to be given explicitly!')

    for fname in infiles:

        try:
            f = dataio.I3File(fname)
            frame = f.pop_frame(icetray.I3Frame.Stream('S'))
            f.close()
        except RuntimeError as e:
            log('WARNING: Could not retrieve S-Frame from {!r}'.format(fname))
            log(str(e))
        else:
            if frame is not None:
                for key in frame.keys():
                    try:
                        frame_obj = frame[key]
                    except KeyError:
                        log('WARNING: Could not retrieve {} from frame'.format(
                            key))
                        frame_obj = None
                    if isinstance(frame_obj, MuonGun.GenerationProbability):
                        log('{}: found "{}" ({})'.format(
                            fname,
                            key,
                            type(frame_obj).__name__), unit="MuonGun")
                        n_generators += 1
                        if generator is None:
                            generator = frame_obj
                        else:
                            generator += frame_obj

        # If we are using equal generators, we can stop once we found the
        # first file with a generator
        if equal_generators and generator:
            break

    if equal_generators:
        assert n_generators == 1
        generator = generator * n_files
        n_generators = n_files
    return generator, n_generators


def get_fluxes_and_names():
    """Get fluxes and names of all available models.

    Returns
    -------
    list of MuonGun models, list of str
        List of MuonGun models
        List of model names
    """
    table_path = os.path.expandvars('$I3_BUILD/MuonGun/resources/tables/*')
    table_files = glob(table_path)
    table_files = [os.path.basename(table_file) for table_file in table_files]

    flux_names = np.unique(
        [table_file.split('.')[0] for table_file in table_files])

    fluxes = []
    for flux_name in flux_names:
        fluxes.append(MuonGun.load_model(flux_name))

    return fluxes, flux_names
