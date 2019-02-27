#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from inspect import isclass

from icecube.weighting import fluxes


class MIMIC_NEUTRINOFLUX():
    def __init__(self, weighting_flux, name):
        if not isinstance(weighting_flux, fluxes.CompiledFlux):
            raise TypeError('Weighting Flux has to be an instance '
                            'of CompiledFlux!')
        else:
            self.weighting_flux = weighting_flux
            self.name = name

    def getFlux(self, ptype, energy, costheta):
        return self.weighting_flux(energy, ptype)

    def __str__(self):
        return self.name


def get_fluxes_and_names():
    flux_models = []
    for obj in dir(fluxes):
        cls = getattr(fluxes, obj)
        if isclass(cls):
            if issubclass(cls, fluxes.CompiledFlux) and \
               cls != fluxes.CompiledFlux:
                flux_models.append(MIMIC_NEUTRINOFLUX(cls(), obj))

    return flux_models, \
        [str(flux_model_i) + 'Weight' for flux_model_i in flux_models]
