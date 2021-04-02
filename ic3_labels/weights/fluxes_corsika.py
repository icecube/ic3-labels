#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from inspect import isclass

from icecube.weighting import fluxes
from icecube.icetray.i3logging import log_error, log_warn
from ic3_labels.weights.resources import fluxes as _fluxes


class MIMIC_NEUTRINOFLUX():
    def __init__(self, weighting_flux, name):
        allowed_base_classes = (fluxes.CompiledFlux, _fluxes.CosmicRayFlux)
        if not isinstance(weighting_flux, allowed_base_classes):
            raise TypeError('Weighting Flux has to be an instance '
                            'of CompiledFlux or CosmicRayFlux!')
        else:
            self.weighting_flux = weighting_flux
            self.name = name

    def getFlux(self, ptype, energy, costheta):
        return self.weighting_flux(energy, ptype)

    def __str__(self):
        return self.name


def get_fluxes_and_names(use_fallback_fluxes=False):
    flux_models = []
    for obj in dir(fluxes):
        cls = getattr(fluxes, obj)
        if isclass(cls):
            if issubclass(cls, fluxes.CompiledFlux) and \
               cls != fluxes.CompiledFlux:
                try:
                    flux_model = MIMIC_NEUTRINOFLUX(cls(), obj)
                    flux_model(1e4, 14)
                except Exception as e:
                    if use_fallback_fluxes:
                        log_warn(e)
                        log_warn('Falling back to ic3_labels flux {}'.format(
                            obj))
                        cls = getattr(_fluxes, obj)
                        flux_model = MIMIC_NEUTRINOFLUX(cls(), obj)
                    else:
                        raise e

                flux_models.append(flux_model)

    return flux_models, \
        [str(flux_model_i) + 'Weight' for flux_model_i in flux_models]
