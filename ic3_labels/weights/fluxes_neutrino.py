#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

'''
Add NewNuFlux Models as well as the possibility to select NuMu
and NuE fluxes depending on the datatype.
Add more measured models.
'''


class UnbrokenPowerLaw(object):
    '''
    Unbroken power law, defined by normalisation, pivot point and
    gamma (negative).
    '''
    def __init__(self, gamma, norm, e_pivot=1e5):
        self.gamma = gamma
        self.norm = norm
        self.e_pivot = e_pivot

    def getFlux(self, ptype, energy, costheta):
        '''
        Allows a call similar to the neutrinoflux module. There is no
        dependency on ptype or costheta since an isotropical flux with
        a flavor ratio of 1:1:1 is assumed. Use the flux per flavor!
        '''
        n_types = 2  # Dividing by n_types gives flux per flavor and per type
        return self.norm * np.power(energy/self.e_pivot, self.gamma) / n_types


hese_flux = UnbrokenPowerLaw(-2., .95e-18)

# Global Fit Paper: https://arxiv.org/abs/1507.03991
# Divide the global_fit flux by 3 to obtain the flux per flavor
global_fit = UnbrokenPowerLaw(-2.5, 6.7e-18 / 3.)

# Aachen Paper: https://arxiv.org/abs/1607.08006
aachen_flux = UnbrokenPowerLaw(-2.13, .9e-18)

# ICRC2017 Collection: https://arxiv.org/abs/1710.01191
aachen_flux_8yr = UnbrokenPowerLaw(-2.19, 1.01e-18)

# MESE Paper: https://arxiv.org/abs/1410.1749
mese_flux = UnbrokenPowerLaw(-2.46, 2.06e-18)

# Cscd ICRC15: https://arxiv.org/abs/1510.05223
cscd_icrc15 = UnbrokenPowerLaw(-2.67, 2.3e-18)

# Current (2019) Cscd bestfit (Ana call paper outline presentation)
cscd_hans = UnbrokenPowerLaw(-2.53, 1.66e-18)

# Most recent HESE: https://arxiv.org/pdf/1510.05223
hese4_fixed = UnbrokenPowerLaw(-2., .84e-18)
hese4_bestfit = UnbrokenPowerLaw(-2.58, 2.2e-18)

# See ICRC2017 Collection
hese6 = UnbrokenPowerLaw(-2.92, 2.46e-18)


MEASUREMENTS = {'global_fit': global_fit,
                'hese_flux': hese_flux,
                'aachen_flux': aachen_flux,
                'mese_flux': mese_flux,
                'cscd_icrc15': cscd_icrc15,
                'hese4_fixed': hese4_fixed,
                'hese4_bestfit': hese4_bestfit,
                'aachen_flux_8yr': aachen_flux_8yr,
                'hese6': hese6,
                'cscd_hans': cscd_hans}


def makeFlux(measurement_name):
    return MEASUREMENTS[measurement_name]


MEASURED_MODELS = MEASUREMENTS.keys()


try:
    from icecube import neutrinoflux
except ImportError:
    NEUTRINOFLUX_MODELS = []
else:
    NEUTRINOFLUX_MODELS = [
        ['honda2006_gaisserH3a_elbert_v2_numu', 'conv'],
        ['honda2006_gaisserH4a_elbert_v2_numu', 'conv'],
        ['sarcevic_std_gaisserH3a_elbert_numu', 'prompt'],
        ['sarcevic_std_gaisserH4a_elbert_numu', 'prompt']]

try:
    from icecube import NewNuFlux as nn
except ImportError:
    NNFLUX_MODELS = []
else:
    NNFLUX_MODELS = [
        ['BERSS_H3a_central', ['none'], 'prompt'],
        ['BERSS_H3p_central', ['none'], 'prompt'],
        ['BERSS_H3p_lower', ['none'], 'prompt'],
        ['BERSS_H3p_upper', ['none'], 'prompt'],
        ['CORSIKA_GaisserH3a_QGSJET-II', ['none'], 'conv'],
        ['CORSIKA_GaisserH3a_SIBYLL-2.1', ['none'], 'conv'],
        ['CORSIKA_GaisserH3a_average', ['none'], 'conv'],
        ['bartol', ['none'], 'conv'],
        ['honda2006', ['none',
                       'gaisserH3a_elbert',
                       'gaisserH4a_elbert',
                       'gst13_elbert',
                       'gst13star_elbert',
                       'polygonato_mod_elbert'], 'conv'],
        ['sarcevic_max', ['none',
                          'gaisserH3a_elbert',
                          'gaisserH4a_elbert',
                          'polygonato_mod_elbert'], 'prompt'],
        ['sarcevic_min', ['none',
                          'gaisserH3a_elbert',
                          'gaisserH4a_elbert',
                          'polygonato_mod_elbert'], 'prompt'],
        ['sarcevic_std', ['none',
                          'gaisserH3a_elbert',
                          'gaisserH4a_elbert',
                          'polygonato_mod_elbert'], 'prompt']]


def get_fluxes_and_names(neutrinoflux_models='all',
                         nnflux_models='all',
                         measured_models='all'):
    if neutrinoflux_models == 'all':
        neutrinoflux_models = NEUTRINOFLUX_MODELS
    elif neutrinoflux_models is None:
        neutrinoflux_models = []

    if nnflux_models == 'all':
        nnflux_models = NNFLUX_MODELS
    elif nnflux_models is None:
        nnflux_models = []

    if measured_models == 'all':
        measured_models = MEASURED_MODELS
    elif measured_models is None:
        measured_models = []

    fluxes = []
    flux_names = []
    for model in neutrinoflux_models:
        model_type = model[1]
        if model_type == 'conv':
            fluxes.append(
                neutrinoflux.ConventionalNeutrinoFluxWithKnee(model[0]))
        if model_type == 'prompt':
            fluxes.append(
                neutrinoflux.PromptNeutrinoFluxWithKnee(model[0]))
        if model_type == 'astro':
            fluxes.append(
                neutrinoflux.AstroNeutrinoFlux(model[0]))
        flux_name = '{}_{}_nflux'.format(model[0], model_type)
        flux_name = flux_name.replace('-', '_')
        flux_names.append(flux_name)

    for model in nnflux_models:
        model_type = model[2]
        flux = nn.makeFlux(model[0])
        for knee_rew in model[1]:
            flux.knee_reweighting_model = knee_rew
            flux_name = '{}_{}_{}_NNFlux'.format(
                model[0],
                knee_rew,
                model_type)
            flux_name.replace('_none', '')

            fluxes.append(flux)
            flux_names.append(flux_name)

    for model in measured_models:
        fluxes.append(makeFlux(model))
        flux_names.append(model)

    return fluxes, flux_names
