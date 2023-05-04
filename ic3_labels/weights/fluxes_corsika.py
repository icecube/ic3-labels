import numpy as np
from inspect import isclass

try:
    from icecube.weighting import fluxes
except ModuleNotFoundError as e:
    import simweights as fluxes

from icecube.icetray.i3logging import log_error, log_warn
from ic3_labels.weights.resources import fluxes as _fluxes


class MIMIC_NEUTRINOFLUX():
    def __init__(self, weighting_flux, name, unit_conversion=1.):
        allowed_base_classes = []
        if hasattr(fluxes, 'CompiledFlux'):
            allowed_base_classes.append(fluxes.CompiledFlux)
        if hasattr(fluxes, 'CosmicRayFlux'):
            allowed_base_classes.append(fluxes.CosmicRayFlux)
        if hasattr(fluxes, '_fluxes'):
            allowed_base_classes.append(fluxes._fluxes.CosmicRayFlux)
        allowed_base_classes = tuple(allowed_base_classes)

        if not isinstance(weighting_flux, allowed_base_classes):
            raise TypeError('Weighting Flux has to be an instance '
                            'of CompiledFlux or CosmicRayFlux!',
                            weighting_flux)
        else:
            self.weighting_flux = weighting_flux
            self.name = name
            self.unit_conversion = unit_conversion

    def getFlux(self, ptype, energy, costheta):
        return self.weighting_flux(energy, ptype) * self.unit_conversion

    def __str__(self):
        return self.name


def get_fluxes_and_names(fallback_to_ic3_labels_flux=False):

    # get parent class
    try:
        ParentClass = fluxes.CompiledFlux
        unit_conversion = 1.
    except Exception as e:
        ParentClass = fluxes._fluxes.CosmicRayFlux
        unit_conversion = 0.0001  # cm^2 to m^2

    # cross-check if used fluxes have correct units
    # note: simweights changed output from m^2 to cm^2 at some point
    flux = fluxes.Hoerandel()(1e5, 14)
    flux_ic3labels = _fluxes.Hoerandel()(1e5, 14)
    assert np.allclose(flux * unit_conversion, flux_ic3labels), (
        flux, flux_ic3labels, unit_conversion)

    flux_models = []
    for obj in dir(fluxes):
        cls = getattr(fluxes, obj)
        if isclass(cls):
            if issubclass(cls, ParentClass) and cls != ParentClass:

                # Try to use flux from icecube.weighting.fluxes / simweights
                try:
                    flux_model = MIMIC_NEUTRINOFLUX(
                        cls(), obj, unit_conversion=unit_conversion)
                    flux_model.getFlux(1e4, 14, 0.)

                except TypeError as e:

                    # skip over FixedFractionFlux, which requires
                    # additional fractions set
                    if issubclass(cls, fluxes.FixedFractionFlux):
                        flux_model = None
                    else:
                        raise e

                # Fall back to ic3_labels version
                # (currently necessary for python >=3.8)
                except Exception as e:

                    # try to obtain flux from ic3_labels
                    if fallback_to_ic3_labels_flux:
                        log_warn('Caught error:' + str(e))
                        log_warn('Falling back to ic3_labels flux {}'.format(
                            obj))
                        cls = getattr(_fluxes, obj)
                        flux_model = MIMIC_NEUTRINOFLUX(
                            cls(), obj, unit_conversion=1.)

                    # if not falling back on ic3_labels fluxed: raise error
                    else:
                        raise e

                if flux_model is not None:
                    flux_models.append(flux_model)

    return flux_models, \
        [str(flux_model_i) + 'Weight' for flux_model_i in flux_models]
