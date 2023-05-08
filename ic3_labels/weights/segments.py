
try:
    from icecube.weighting.weighting import from_simprod
    from icecube.weighting import get_weighted_primary
except ModuleNotFoundError as e:
    from ic3_labels.weights.utils import get_weighted_primary

from icecube import icetray, dataclasses
from icecube import MuonGun
from icecube.icetray import I3Units

try:
    from collections import Iterable
except ImportError:
    # >= python 3.10
    from collections.abc import Iterable

from copy import deepcopy

import math
import numpy as np

import os
import pickle


from ic3_labels.weights import fluxes_corsika, fluxes_muongun, fluxes_neutrino


def generate_generator(dataset_number, n_files, outpath=None):
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
    if outpath is not None:
        with open(outpath, 'w') as open_file:
            pickle.dump(generator, open_file, protocol=2)
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
        if frame.Has('CorsikaWeightMap'):
            cwm = frame["CorsikaWeightMap"]

            # overwrite values obtained from MCPrimary
            energy = cwm["PrimaryEnergy"]
            ptype = cwm["PrimaryType"]
            n_events = cwm['NEvents']
            type_weight = 1.0

            spectral_index = cwm["PrimarySpectralIndex"]
            if spectral_index == -1:
                energy_integral = (
                    np.log(cwm['EnergyPrimaryMax'])
                    - np.log(cwm['EnergyPrimaryMin'])
                )
            else:
                energy_integral = (
                    cwm['EnergyPrimaryMax']**(spectral_index + 1)
                    - cwm['EnergyPrimaryMin']**(spectral_index + 1)
                    ) / (spectral_index + 1)
            energy_weight = cwm['PrimaryEnergy']**spectral_index
            one_weight = energy_integral / energy_weight * cwm["AreaSum"]

        elif frame.Has('I3MCWeightDict'):
            one_weight = frame['I3MCWeightDict']['OneWeight']
            n_events = frame['I3MCWeightDict']['NEvents']
            type_weight = .5

        else:
            raise TypeError('No I3MCWeightDict or CorsikaWeightMap found!')

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
        Defines the kind of data: 'nugen', 'genie', 'muongun', 'corsika'
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
        import pickle
        if os.path.isfile(generator):
            with open(generator, 'r') as open_file:
                generator = pickle.load(open_file)
        else:
            raise ValueError('File {} not found!'.format(generator))

    tray.AddModule(get_weighted_primary, 'get dem primary',
                   If=lambda frame: not frame.Has('MCPrimary'))

    if dataset_type.lower() != 'muongun':
        # Corsika, NuGen, or GENIE
        tray.AddModule(calc_weights,
                       'WeightCalc',
                       fluxes=fluxes,
                       flux_names=flux_names,
                       n_files=dataset_n_files,
                       generator=generator,
                       key=key)
    else:
        # MuonGun
        # tray.AddModule('Rename', 'renaming_mctree',
        #                Keys=['I3MCTree_preMuonProp', 'I3MCTree'],
        #                If=lambda frame: not frame.Has('I3MCTree'))
        tray.AddSegment(calc_weights_muongun,
                        'WeightCalc',
                        fluxes=fluxes,
                        flux_names=flux_names,
                        generator=generator,
                        key=key)
        # tray.AddModule('Rename', 'revert_renaming_mctree',
        #                Keys=['I3MCTree', 'I3MCTree_preMuonProp'],
        #                If=lambda f: not f.Has('I3MCTree_preMuonProp'))


class AddWeightMetaData(icetray.I3Module):
    """
    Add a "W" (weight) frame for the weight meta data.
    """
    def __init__(self, ctx):
        super(AddWeightMetaData, self).__init__(ctx)
        self.AddParameter("NFiles", "Number of files used for weighting", None)
        self.AddParameter("NEventsPerRun", "Number of events per run.", None)
        self.AddParameter("Key", "Output key of the weights.", 'weights')

    def Configure(self):
        self._n_files = self.GetParameter("NFiles")
        self._n_events_per_run = self.GetParameter("NEventsPerRun")
        self._key = self.GetParameter("Key")
        self._frame_has_been_pushed = False
        self._frame_key = '{}_meta_info'.format(self._key)

    def Process(self):

        # get next frame
        frame = self.PopFrame()

        if not self._frame_has_been_pushed:

            # create weight frame and push it
            weight_frame = icetray.I3Frame('W')

            meta_data = {
                'n_files': self._n_files,
                'n_events_per_run': self._n_events_per_run,
            }
            weight_frame[self._frame_key] = dataclasses.I3MapStringInt(
                meta_data)

            self.PushFrame(weight_frame)

            self._frame_has_been_pushed = True

        self.PushFrame(frame)


@icetray.traysegment
def WeightEvents(tray, name,
                 infiles,
                 dataset_type,
                 dataset_n_files,
                 dataset_n_events_per_run,
                 dataset_number,
                 muongun_equal_generator=False,
                 key='weights',
                 use_from_simprod=False,
                 add_mceq_weights=False,
                 mceq_kwargs={},
                 add_nuveto_pf=False,
                 nuveto_kwargs={},
                 add_mese_weights=False,
                 add_atmospheric_self_veto=False,
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
        Defines the kind of data: 'nugen', 'genie', 'muongun', 'corsika'
    dataset_n_files : int
        Number of files in dataset. For MuonGun this is overwritten by the
        number of found generators. In this case, this value is only used
        to check if it matches the found n_files (if check is performed).
    dataset_n_events_per_run : int
        Number of events per run. Needed for MESE weights.
    dataset_number : int
        Corsika dataset number.
    muongun_equal_generator : bool, optional
        If True, it is assumed that all MuonGun generator objects are the same.
        In this case, only the first found MuonGun generator will be used
        and multiplied by the provided 'dataset_n_files'.
    key : str
        Defines the key to which the weight dictionary will be booked.
    use_from_simprod : bool, optional
        If True, weights will be calculated by obtaining a generator via
        from_simprod. If False, weights will be calculated based on the
        I3MCWeightDict for NuGen or CorsikaWeightMap (Corsika).
    add_mceq_weights : bool, optional
        If True, MCEq weights will be added. Make sure to add an existing
        cache file, otherwise this may take very long!
    mceq_kwargs : dict, optional
        Keyword arguments passed on to MCEq.
    add_nuveto_pf : bool, optional
        If True, nuVeto passing fractions will be added. Make sure to add
        an existing cache file, otherwise this may take very long!
    nuveto_kwargs : dict, optional
        Keyword arguments passed on to nuVeto.
    add_mese_weights : bool, optional
        If true, weights used for MESE 7yr cascade paper will be added.
        (As well as an additional filtering step)
    add_atmospheric_self_veto : bool, optional
        If True, the atmospheric self-veto passing fractions will be calculated
        and written to the frame.
    check_n_files : bool or list of str, optional
        If true, check if provided n_files argument seems reasonable.
        If list of str and if dataset_type is in the defined list:
        check if provided n_files arguments seems reasonable.
        The list of str defines the datatypes (in lower case) for which the
        n_files will be checked.

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
                                    n_files=dataset_n_files,
                                    equal_generators=muongun_equal_generator)

    elif dataset_type == 'corsika':
        fluxes, flux_names = fluxes_corsika.get_fluxes_and_names()

        n_files = len([f for f in infiles if 'gcd' not in f.lower()])
        if use_from_simprod:
            generator = from_simprod(dataset_number) * dataset_n_files
        else:
            generator = None

    elif dataset_type in ['nugen', 'genie']:
        fluxes, flux_names = fluxes_neutrino.get_fluxes_and_names()

        generator = None
        n_files = len([f for f in infiles if 'gcd' not in f.lower()])

    else:
        raise ValueError('Unkown dataset_type: {!r}'.format(dataset_type))

    # check if found number of events seems reasonable
    perform_check = False
    if isinstance(check_n_files, bool):
        perform_check = check_n_files
    else:
        if dataset_type in check_n_files:
            perform_check = True
    if perform_check:
        assert n_files == dataset_n_files, \
            'N_files do not match: {!r} != {!r}'.format(n_files,
                                                        dataset_n_files)

    # Use the number of found generators for MuonGun
    if dataset_type == 'muongun':
        dataset_n_files = n_files

    tray.AddModule(
        AddWeightMetaData, 'AddWeightMetaData',
        NFiles=dataset_n_files,
        NEventsPerRun=dataset_n_events_per_run,
        Key=key,
    )

    tray.AddSegment(do_the_weighting, 'do_the_weighting',
                    fluxes=fluxes,
                    flux_names=flux_names,
                    dataset_type=dataset_type,
                    dataset_n_files=dataset_n_files,
                    generator=generator,
                    key=key,
                    )

    if add_mceq_weights and dataset_type in ['nugen']:
        from ic3_labels.weights.modules import AddMCEqWeights

        tray.AddModule(
            AddMCEqWeights, 'AddMCEqWeights',
            n_files=dataset_n_files,
            **mceq_kwargs
        )

    if add_nuveto_pf and dataset_type in ['nugen']:
        from ic3_labels.weights.modules import AddNuVetoPassingFraction

        tray.AddModule(
            AddNuVetoPassingFraction, 'AddNuVetoPassingFraction',
            **nuveto_kwargs
        )

    if add_mese_weights and dataset_type in ['muongun', 'nugen', 'genie']:
        from ic3_labels.weights.mese_weights import MESEWeights

        tray.AddModule(MESEWeights, 'MESEWeights',
                       DatasetType=dataset_type,
                       DatasetNFiles=dataset_n_files,
                       DatasetNEventsPerRun=dataset_n_events_per_run,
                       OutputKey='{}_mese'.format(key),
                       )

    if add_atmospheric_self_veto and dataset_type in ['nugen', 'genie']:
        from ic3_labels.weights import self_veto

        tray.AddModule(
            self_veto.AtmosphericSelfVetoModule, 'AtmosphericSelfVetoModule',
            DatasetType=dataset_type,
        )


class UpdateMergedWeights(icetray.I3Module):
    """
    This updates the weights and meta data when merging files
    """
    def __init__(self, ctx):
        super(UpdateMergedWeights, self).__init__(ctx)
        self.AddParameter("Key", "Output key of the weights.", 'weights')
        self.AddParameter(
            "TotalNFiles",
            "Total number of n files of all merged files.",
            None,
        )

    def Configure(self):
        self._total_n_files = self.GetParameter("TotalNFiles")
        self._key = self.GetParameter("Key")
        self._frame_key = '{}_meta_info'.format(self._key)
        self._last_n_files = None
        self._last_n_events_per_run = None
        self._already_pushed_w_frame = False

    def Process(self):
        frame = self.PopFrame()

        # update meta data if it exists in this frame
        if (self._frame_key in frame and
                frame.get_stop(self._frame_key) == frame.Stop):
            meta = frame[self._frame_key]

            if self._last_n_events_per_run is not None:
                if meta['n_events_per_run'] != self._last_n_events_per_run:
                    raise ValueError('N events per Run changes: {} {}'.format(
                        self._last_n_events_per_run,
                        meta['n_events_per_run'],
                    ))

            # keep track of the last n_files: N_i
            self._last_n_files = meta['n_files']
            self._last_n_events_per_run = meta['n_events_per_run']

            # update total n_files
            meta['n_files'] = self._total_n_files

            # replace meta data
            del frame[self._frame_key]
            frame[self._frame_key] = meta

        # update weight data if it exists in this frame
        if self._key in frame and frame.get_stop(self._key) == frame.Stop:
            weights = frame[self._key]

            if self._last_n_files is None:
                raise ValueError('Did not find meta data!', frame)

            updated_weights = dataclasses.I3MapStringDouble()
            for name, weight in weights.items():
                updated_weights[name] = (
                    float(weight) * self._last_n_files
                    / frame[self._frame_key]['n_files']
                )
            del frame[self._key]
            frame[self._key] = updated_weights

        # only write one W-frame per file
        if (frame.Stop.id == 'W' and self._frame_key in frame and
                frame.get_stop(self._frame_key) == frame.Stop):
            if self._already_pushed_w_frame:
                return
            else:
                self.PushFrame(frame)
                self._already_pushed_w_frame = True
        else:
            self.PushFrame(frame)
