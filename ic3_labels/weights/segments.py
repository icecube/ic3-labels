#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

from icecube.weighting.weighting import from_simprod
from icecube.weighting import get_weighted_primary
from icecube import icetray, dataclasses
from icecube import MuonGun
from icecube.icetray import I3Units

from collections import Iterable
from copy import deepcopy

import math
import numpy as np

import os
import cPickle as pickle

from ic3_labels.weights import fluxes_corsika, fluxes_muongun, fluxes_neutrino
from ic3_labels.weights.mese_weights import MESEWeights


def generate_generator(outpath, dataset_number, n_files):
    if isinstance(dataset_number, Iterable) and isinstance(n_files, Iterable):
        if len(dataset_number) != len(np.flatnonzero(
                np.asarray(dataset_number))):
            print('At least one of the present datasets of this type doesnt '
                  'have a generator. The weighting is done with OneWeight and '
                  'there is only the current dataset taken into account for '
                  'the weighting!')
            return None
        if len(dataset_number) != len(n_files):
            raise ValueError('Dataset_number and n_files have to be the same '
                             'length if both are supposed to be Iterables.')
        else:
            for i in range(len(dataset_number)):
                if i == 0:
                    generator = from_simprod(dataset_number[i]) * n_files[i]
                else:
                    generator += from_simprod(dataset_number[i]) * n_files[i]
    elif (isinstance(dataset_number, int) or
          isinstance(dataset_number, float)) and \
         (isinstance(n_files, int) or
          isinstance(n_files, float)):
        generator = from_simprod(dataset_number) * n_files
    else:
        raise ValueError('Dataset_number and n_files either have to be both '
                         'numbers (int or float) or be both Iterables of the '
                         'same length.')
    with open(outpath, 'w') as open_file:
        pickle.dump(generator, open_file)
    return outpath


def calc_weights(frame, fluxes, flux_names, n_files, generator, key):
    weight_dict = {}
    primary = frame['MCPrimary']
    energy = primary.energy
    ptype = primary.type
    costheta = math.cos(primary.dir.zenith)

    if generator is not None:
        if frame.Has('I3MCWeightDict'):
            mc_weight_dict = frame['I3MCWeightDict']
            p_int = mc_weight_dict['TotalInteractionProbabilityWeight']
            unit = I3Units.cm2 / I3Units.m2
        else:
            p_int = 1
            unit = 1

        for flux, flux_name in zip(fluxes, flux_names):
            try:
                flux_val = flux.getFlux(ptype, energy, costheta)
            except RuntimeError:
                fluxes.remove(flux)
                flux_names.remove(flux_name)
            else:
                # Type weight seems to be obsolete with generators
                # type_weight = .5
                weight = p_int * (flux_val / unit) / \
                    generator(energy, ptype, costheta)
                weight_dict[flux_name] = float(weight)
    else:
        if not frame.Has('I3MCWeightDict'):
            raise TypeError('For non neutrino simulation a generator '
                            'object should be present!')
        else:
            one_weight = frame['I3MCWeightDict']['OneWeight']
            n_events = frame['I3MCWeightDict']['NEvents']
            type_weight = .5

            for flux, flux_name in zip(fluxes, flux_names):
                try:
                    flux_val = flux.getFlux(ptype, energy, costheta)
                except RuntimeError:
                    fluxes.remove(flux)
                    flux_names.remove(flux_name)
                else:
                    weight = flux_val * one_weight / \
                        (type_weight * n_events * n_files)
                    weight_dict[flux_name] = float(weight)

    frame[key] = dataclasses.I3MapStringDouble(weight_dict)
    return True


@icetray.traysegment
def calc_weights_muongun(tray, name, fluxes, flux_names, generator, key):

    def update_weight_dict(frame, frame_key, flux_name):
        if not frame.Has(frame_key):
            I3_double_container = dataclasses.I3MapStringDouble()
            I3_double_container[flux_name] = deepcopy(frame[flux_name].value)
        else:
            I3_double_container = deepcopy(frame[frame_key])
            I3_double_container[flux_name] = deepcopy(frame[flux_name].value)
            frame.Delete(frame_key)

        frame.Put(frame_key, I3_double_container)
        return True

    for flux, flux_name in zip(fluxes, flux_names):
        flux_name = flux_name.replace('-', '_')
        tray.AddModule('I3MuonGun::WeightCalculatorModule',
                       flux_name,
                       Model=flux,
                       Generator=generator)
        tray.AddModule(update_weight_dict, 'update_wd_{}'.format(flux_name),
                       frame_key=key,
                       flux_name=flux_name)
        tray.AddModule('Delete',
                       'delete_{}'.format(flux_name),
                       keys=[flux_name])


@icetray.traysegment
def do_the_weighting(tray, name,
                     fluxes,
                     flux_names,
                     dataset_type,
                     dataset_n_files,
                     generator,
                     key):
    """Calculate weights and add to frame

    Parameters
    ----------
    tray : TYPE
        Description
    name : TYPE
        Description
    fluxes : TYPE
        Description
    flux_names : TYPE
        Description
    dataset_type : str
        Defines the kind of data: 'nugen', 'muongun', 'corsika'
    dataset_n_files : int
        Number of files in dataset. Not needed for MuonGun data.
    generator : I3 generator object
        The generator object
    key : str
        Defines the key to which the weight dictionary will be booked.

    Raises
    ------
    ValueError
        Description
    """
    if isinstance(generator, str):
        import cPickle as pickle
        if os.path.isfile(generator):
            with open(generator, 'r') as open_file:
                generator = pickle.load(open_file)
        else:
            raise ValueError('File {} not found!'.format(generator))

    tray.AddModule(get_weighted_primary, 'get dem primary',
                   If=lambda frame: not frame.Has('MCPrimary'))

    if dataset_type.lower() != 'muongun':
        # Corsika or NuGen
        tray.AddModule(calc_weights,
                       'WeightCalc',
                       fluxes=fluxes,
                       flux_names=flux_names,
                       n_files=dataset_n_files,
                       generator=generator,
                       key=key)
    else:
        # MuonGun
        tray.AddModule('Rename', 'renaming_mctree',
                       Keys=['I3MCTree_preMuonProp', 'I3MCTree'],
                       If=lambda frame: not frame.Has('I3MCTree'))
        tray.AddSegment(calc_weights_muongun,
                        'WeightCalc',
                        fluxes=fluxes,
                        flux_names=flux_names,
                        generator=generator,
                        key=key)
        tray.AddModule('Rename', 'revert_renaming_mctree',
                       Keys=['I3MCTree', 'I3MCTree_preMuonProp'],
                       If=lambda frame: not frame.Has('I3MCTree_preMuonProp'))


@icetray.traysegment
def WeightEvents(tray, name,
                 infiles,
                 dataset_type,
                 dataset_n_files,
                 dataset_n_events_per_run,
                 dataset_number,
                 key='weights',
                 add_mese_weights=False,
                 check_n_files=True):
    """Calculate weights and add to frame

    Parameters
    ----------
    tray : Icetray
        The IceCube Icetray
    name : str
        Name of I3Segement
    infiles : list of str
        A list of the input file paths.
    dataset_type : str
        Defines the kind of data: 'nugen', 'muongun', 'corsika'
    dataset_n_files : int
        Number of files in dataset. Not needed for MuonGun data.
    dataset_n_events_per_run : int
        Number of events per run. Needed for MESE weights.
    dataset_number : int
        Corsika dataset number.
    key : str
        Defines the key to which the weight dictionary will be booked.
    add_mese_weights : bool, optional
        If true, weights used for MESE 7yr cascade paper will be added.
        (As well as an additional filtering step)
    check_n_files : bool, optional
        If true, check if provided n_files argument seems reasonable.

    Raises
    ------
    ValueError
        Description
    """
    dataset_type = dataset_type.lower()

    if dataset_type == 'muongun':

        # get fluxes and generator
        fluxes, flux_names = fluxes_muongun.get_fluxes_and_names()
        generator, n_files = fluxes_muongun.harvest_generators(
                                                    infiles,
                                                    n_files=-1,
                                                    equal_generators=False)

    elif dataset_type == 'corsika':
        fluxes, flux_names = fluxes_corsika.get_fluxes_and_names()

        n_files = len([f for f in infiles if 'gcd' not in f.lower()])
        generator = from_simprod(dataset_number) * dataset_n_files

    elif dataset_type == 'nugen':
        fluxes, flux_names = fluxes_neutrino.get_fluxes_and_names()

        generator = None
        n_files = len([f for f in infiles if 'gcd' not in f.lower()])

    else:
        raise ValueError('Unkown dataset_type: {!r}'.format(dataset_type))

    if check_n_files:
        assert n_files == dataset_n_files, \
            'N_files do not match: {!r} != {!r}'.format(n_files,
                                                        dataset_n_files)

    def add_meta_data(frame):
        frame_key = '{}_meta_info'.format(key)
        meta_data = {
            'n_files': dataset_n_files,
            'n_events_per_run': dataset_n_events_per_run,
        }
        frame[frame_key] = dataclasses.I3MapStringDouble(meta_data)
        return True
    tray.AddModule(add_meta_data, 'add_meta_data')

    tray.AddSegment(do_the_weighting, 'do_the_weighting',
                    fluxes=fluxes,
                    flux_names=flux_names,
                    dataset_type=dataset_type,
                    dataset_n_files=dataset_n_files,
                    generator=generator,
                    key='{}_{}'.format(key, dataset_type),
                    )

    if add_mese_weights and dataset_type in ['muongun', 'nugen']:
        tray.AddModule(MESEWeights, 'MESEWeights',
                       DatasetType=dataset_type,
                       DatasetNFiles=dataset_n_files,
                       DatasetNEventsPerRun=dataset_n_events_per_run,
                       OutputKey='{}_mese'.format(key),
                       )
