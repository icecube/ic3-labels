#!/usr/bin/env python
# -*- coding: utf-8 -*-
from icecube import dataclasses, icetray
import numpy as np

from ic3_labels.weights.mceq_models import MCEQFlux
from ic3_labels.weights.nuveto_models import AtmosphericNuVeto


class AddNuVetoPassingFraction(icetray.I3ConditionalModule):
    '''Module to add MCEq weights via provided cache file.
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('cache_file', 'The path to the cache file.')
        self.AddParameter(
            'prpl', 'NuVeto prpl file.', 'dnn_cascade_selection',
        )
        self.AddParameter(
            'output_key',
            'The name of the key to which the results will be written to.',
            'nuveto',
        )
        self.AddParameter(
            'primary_models',
            'The list of primary models for which to compute weights.',
            ['H3a'],
        )
        self.AddParameter(
            'interaction_model',
            'The interaction model for MCEq to use..',
            'SIBYLL2.3c',
        )

    def Configure(self):
        """Configure PulseModification.
        """
        self.cache_file = self.GetParameter("cache_file")
        self.prpl = self.GetParameter("prpl")
        self.output_key = self.GetParameter("output_key")
        self.primary_models = self.GetParameter("primary_models")
        self.interaction_model = self.GetParameter("interaction_model")

        # initialize fluxes
        self.veto_objects = {}
        for primary_model in self.primary_models:
            veto_obj = AtmosphericNuVeto()
            veto_obj.initialize(
                prpl=self.prpl,
                interaction_model=self.interaction_model,
                primary_model=primary_model,
                cache_file=self.cache_file,
                cache_read_only=True,
            )
            self.veto_objects[primary_model] = veto_obj

    def Physics(self, frame):
        """Add MCEq weights to existing weights key in frame

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        veto_pf = dataclasses.I3MapStringDouble()

        ow = frame['I3MCWeightDict']['OneWeight']
        type_weight = frame['I3MCWeightDict']['TypeWeight']
        ptype = frame['I3MCWeightDict']['PrimaryNeutrinoType']
        energy = frame['I3MCWeightDict']['PrimaryNeutrinoEnergy']
        cos_zen = np.cos(frame['I3MCWeightDict']['PrimaryNeutrinoZenith'])
        n_events = frame['I3MCWeightDict']['NEvents']

        for primary_model in self.primary_models:

            veto_obj = self.veto_objects[primary_model]

            for flux_type in ['total', 'pr', 'conv']:
                pf = veto_obj.get_passing_fraction(
                    ptype=ptype,
                    energy=energy,
                    costheta=cos_zen,
                    flux_type=flux_type,
                )

                flux_name = 'pf_{}_{}_{}_{}'.format(
                    self.prpl,
                    primary_model,
                    self.interaction_model,
                    flux_type,
                ).replace('.', '_')
                veto_pf[flux_name] = float(pf)

        frame[self.output_key] = veto_pf

        self.PushFrame(frame)


class AddMCEqWeights(icetray.I3ConditionalModule):
    '''Module to add MCEq weights via provided cache file.
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('n_files', 'The number of runs')
        self.AddParameter('cache_file', 'The path to the cache file.')
        self.AddParameter(
            'weight_key',
            'The name of the weight key in the I3Frame.',
            'weights',
        )
        self.AddParameter(
            'primary_models',
            'The list of primary models for which to compute weights.',
            ['H3a'],
        )
        self.AddParameter(
            'interaction_model',
            'The interaction model for MCEq to use..',
            'SIBYLL2.3c',
        )

    def Configure(self):
        """Configure PulseModification.
        """
        self.n_files = self.GetParameter("n_files")
        self.cache_file = self.GetParameter("cache_file")
        self.weight_key = self.GetParameter("weight_key")
        self.primary_models = self.GetParameter("primary_models")
        self.interaction_model = self.GetParameter("interaction_model")

        # initialize fluxes
        self.fluxes = {}
        for primary_model in self.primary_models:
            flux = MCEQFlux()
            flux.initialize(
                interaction_model=self.interaction_model,
                primary_model=primary_model,
                cache_file=self.cache_file,
                cache_read_only=True,
            )
            self.fluxes[primary_model] = flux

    def Physics(self, frame):
        """Add MCEq weights to existing weights key in frame

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        weights = dataclasses.I3MapStringDouble(frame[self.weight_key])

        ow = frame['I3MCWeightDict']['OneWeight']
        type_weight = frame['I3MCWeightDict']['TypeWeight']
        ptype = frame['I3MCWeightDict']['PrimaryNeutrinoType']
        energy = frame['I3MCWeightDict']['PrimaryNeutrinoEnergy']
        cos_zen = np.cos(frame['I3MCWeightDict']['PrimaryNeutrinoZenith'])
        n_events = frame['I3MCWeightDict']['NEvents']

        for primary_model, flux in self.primary_models.items():

            flux = self.fluxes[primary_model]

            for flux_type in ['total', 'pr', 'conv']:
                flux_val = flux.getFlux(
                    ptype=ptype,
                    energy=energy,
                    costheta=cos_zen,
                    flux_type=flux_type,
                )

                weight = flux_val * ow / (
                        type_weight * n_events * self.n_files)
                flux_name = 'MCEq_{}_{}_{}'.format(
                    primary_model, self.interaction_model.lower(), flux_type,
                ).replace('.', '_')
                weights[flux_name] = float(weight)

        del frame[self.weight_key]
        frame[self.weight_key] = weights

        self.PushFrame(frame)
