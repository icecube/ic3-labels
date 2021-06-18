"""nuVeto Atmospheric Self-Veto Models

This file implements a wrapper class around nuVeto
(https://github.com/tianluyuan/nuVeto) which builds splines in energy and
zenith.
These can then be used to calculate the self-veto effect for atmospheric
neutrinos. See also the paper to nuVeto:
https://arxiv.org/abs/1805.11003

It is recommended to cache the results of nuVeto because these take a while
to produce. By default, the cache file is chosen
to be located in the 'resources' directory relative to the location of this
file. You may also set the environment variable 'NUVETO_CACHE_DIR' in order
to choose a different location for the cache file, or pass in an explicit
cache file when initializing the AtmosphericNuVeto object.

Environment Variables:

    'NUVETO_CACHE_DIR':
        If provided, the MCEq cache file will be written to this directory.
"""
import os
import logging
import pkg_resources
from copy import deepcopy
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from multiprocessing import Pool

import ic3_labels

log = logging.getLogger('AtmosphericNuVeto')


# If cashier is available, set up directory for caching of nuVeto results
try:
    from ic3_labels.weights.resources.cashier import cache
    got_cashier = True

    if 'NUVETO_CACHE_DIR' in os.environ:
        cache_dir = os.environ['NUVETO_CACHE_DIR']
        log.info(
            "Found 'NUVETO_CACHE_DIR' in environment variables: {}".format(
                cache_dir))

        if not os.path.exists(cache_dir):
            log.info('Creating cache directory: {}'.format(cache_dir))
            os.makedirs(cache_dir)

        CACHE_FILE = os.path.join(cache_dir, 'nuVeto.cache')

    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CACHE_FILE = os.path.join(script_dir, 'resources', 'nuVeto.cache')

    log.info('Using nuVeto cache file: {}'.format(CACHE_FILE))

except ImportError:
    got_cashier = False
    CACHE_FILE = None
    log.info("Could not import 'cashier'. NuVeto results will not be cached!")


# Dictionary that converts ptype -> nuVeto type string
PTYPE_CONVERTER = {
    12: 'nu_e',
    -12: 'nu_ebar',
    14: 'nu_mu',
    -14: 'nu_mubar',
    16: 'nu_tau',
    -16: 'nu_taubar',
}


def __solve_one_cos_theta__(settings):
    """Helper Function for Multiprocessing

    Solves for one cos(theta) grid point for the specified `settings`.

    Parameters
    ----------
    settings : dict
        A dictionary with the settings to run nuVeto with.

    Returns
    -------
    dict
        The passing fraction result for the total flux.
    dict
        The passing fraction result for the conv flux.
    dict
        The passing fraction result for the prompt flux.
    """
    from nuVeto import nuveto
    from nuVeto.utils import Units

    nuveto_obj = nuveto.builder(
        cos_theta=settings['cos_theta'],
        pmodel=settings['pmodel'],
        hadr=settings['hadr'],
        barr_mods=settings['barr_mods'],
        depth=settings['depth'],
        density=settings['density'],
    )

    total_pf_dict_i = {}
    conv_pf_dict_i = {}
    pr_pf_dict_i = {}
    for key in settings['ptype_converter'].keys():
        shape = (len(settings['energy_grid']),)
        total_pf_dict_i[key] = np.ones(shape)
        conv_pf_dict_i[key] = np.ones(shape)
        pr_pf_dict_i[key] = np.ones(shape)

    # fill in passing fractions
    for key, value in settings['ptype_converter'].items():
        print('At particle type:', value)

        for index_energy, energy_i in enumerate(settings['energy_grid']):

            # total
            num, den = nuveto_obj.get_fluxes(
                enu=energy_i*Units.GeV,
                kind='total {}'.format(value),
                accuracy=3.5,
                prpl=settings['prpl'],
                corr_only=False,
            )
            if num == den == 0:
                # If both are zero, we will just set the passing
                # fraction to 1. This is the conservative choice,
                # since it does not change anython. Passing fractions
                # are uusall muliplied to weights.
                fraction = 1.
            else:
                fraction = num/den

            total_pf_dict_i[key][index_energy] = fraction

            # conv
            num, den = nuveto_obj.get_fluxes(
                enu=energy_i*Units.GeV,
                kind='conv {}'.format(value),
                accuracy=3.5,
                prpl=settings['prpl'],
                corr_only=False,
            )
            if num == den == 0:
                # If both are zero, we will just set the passing
                # fraction to 1. This is the conservative choice,
                # since it does not change anython. Passing fractions
                # are uusall muliplied to weights.
                fraction = 1.
            else:
                fraction = num/den
            conv_pf_dict_i[key][index_energy] = fraction

            # prompt
            num, den = nuveto_obj.get_fluxes(
                enu=energy_i*Units.GeV,
                kind='pr {}'.format(value),
                accuracy=3.5,
                prpl=settings['prpl'],
                corr_only=False,
            )
            if num == den == 0:
                # If both are zero, we will just set the passing
                # fraction to 1. This is the conservative choice,
                # since it does not change anython. Passing fractions
                # are uusall muliplied to weights.
                fraction = 1.
            else:
                fraction = num/den
            pr_pf_dict_i[key][index_energy] = fraction

    return total_pf_dict_i, conv_pf_dict_i, pr_pf_dict_i


def get_spline(
        interaction_model,
        primary_model,
        prpl,
        months,
        theta_grid,
        theta_grid_cos,
        energy_grid,
        n_jobs=1,
        cached=True,
        cache_file=CACHE_FILE):
    """Get nuVeto spline

    Caculates nuVeto results for the given parameters. The results
    are obtained for the provided grid and interpolated.

    Parameters
    ----------
    interaction_model : str
        The interaction model. This is passed on to `MCEq` (via nuVeto).
    primary_model : str
        The primary model to use. Must be one of:
            GST_3-gen, GST_4-gen, H3a, H4a, poly-gonato, TIG, ZS, ZSP, GH
    prpl : str
        The detector veto probability PDF to use. This must be a valid
        prpl PDF created and available in nuVeto. This option is passed
        on to nuVeto.
    months : list of str
        The months for which to solve the cascade equations. These must be
        provided as a list of month names, e.g. ['January', 'August']. A list
        of splines will be returned of the same length as `months`.
    theta_grid : array_like
        The grid points in theta to evaluate on in degrees. If `theta_grid_cos`
        is True, this is instead cos(theta).
    theta_grid_cos : bool
        If True, `theta_grid` is interpreted as cos(theta), i.e. arccos() is
        applied first.
    energy_grid : array_like
        The energy grid points [in GeV] to evaluate on.
    n_jobs : int, optional
        Number of jobs to compute the splines. The grid evaluation points
        along zenith are distributed over the specified `n_jobs`.
    cached : bool, optional
        If True, the result will be cached, or taken from cache if previously
        already computed. This is recommended, as MCEq takes a while to run.
    cache_file : str, optional
        The path to the cache file to use.

    Returns
    -------
    dict
        The result of nuVeto together with the fitted splines. The structure is
        as follows:
        {
            # first month provided via `months`
            0: {
                'total_spline_dict': dict of RectBivariateSpline
                    A dictionary with the fitted splines for each particle
                    type for the 'total' passing fraction.
                    The dictionary keys are the PDG particle encodings.
                'conv_spline_dict': dict of RectBivariateSpline
                    A dictionary with the fitted splines for each particle
                    type for the 'conv' passing fraction.
                    The dictionary keys are the PDG particle encodings.
                'pr_spline_dict': dict of RectBivariateSpline
                    A dictionary with the fitted splines for each particle
                    type for the 'pr' passing fraction.
                    The dictionary keys are the PDG particle encodings.
                'total_pf_dict': dict of array_like
                    A dictionary with the total passing fraction for each
                    grid point. This is the result obtained from nuVeto
                    for the 'total' flux.
                'conv_pf_dict': dict of array_like
                    A dictionary with the conv passing fraction for each
                    grid point. This is the result obtained from nuVeto
                    for the 'conv' flux.
                'pr_pf_dict': dict of array_like
                    A dictionary with the prompt passing fraction for each
                    grid point. This is the result obtained from nuVeto
                    for the 'pr' flux.
                'nuveto_version' : str
                    The nuVeto version that was used to create the splines.
                'ic3_labels_version' : str
                    The version of the ic3-labels package that was used to
                    create the splines.
                'log10_e_grid' : array_like
                    The grid of energy points in log10.
                'theta_grid' : array_like
                    The grid of thetas.
            }

            # second month provided via `months`
            1: {
                ...
            }

            ...
        }
    """
    log.info('Getting Spline for {}; {} (cached={})'.format(
        interaction_model,
        primary_model,
        cached))

    def __solve_month__(
            interaction_model,
            pmodel,
            density_model,
            prpl,
            theta_grid,
            theta_grid_cos,
            energy_grid,
            n_jobs=1,
            ptype_converter=PTYPE_CONVERTER,
            eps=1e-128,
            ):
        """Compute passing fractions via nuVeto for the provided parameters

        Parameters
        ----------
        interaction_model : str
            The interaction model. This is passed on to `MCEq` (via nuVeto).
        pmodel : tuple of crf.PrimaryModel, 'str'
            The primary model to use. This is passed on to `MCEq` (via nuVeto).
        density_model : (str, (str, str))
            The density model to use. This is passed on to `MCEq` (via nuVeto).
        prpl : str
            The detector veto probability PDF to use. This must be a valid
            prpl PDF created and available in nuVeto. This option is passed
            on to nuVeto.
        theta_grid : array_like
            The grid points in theta to evaluate on in degrees.
            If `theta_grid_cos` is True, this is instead cos(theta).
        theta_grid_cos : bool
            If True, `theta_grid` is interpreted as cos(theta),
            i.e. arccos() is applied first.
        energy_grid : array_like
            The energy grid points [in GeV] to evaluate on.
        n_jobs : int, optional
            Number of jobs to compute the splines. The grid evaluation points
            along zenith are distributed over the specified `n_jobs`.
        ptype_converter : dict, optional
            A dictionary that converts PDG encoding to nuVeto type string.
        eps : float, optional
            A small float value > 0 that is used to clip the passing fraction
            prior to applying log10 for the spline fitting.

        Returns
        -------
        list of dict of RectBivariateSpline
            A list of dictionaries with the fitted splines for each particle
            type. The dictionary keys are the PDG particle encodings.
            The order of the dictionaries are: 'total', 'conv', 'pr'
        list of dict of array_like
            A list of dictionaries with the total flux for each grid point.
            This is the result obtained from MCEq.
            The order of the dictionaries are: 'total', 'conv', 'pr'
        array_like
            The grid of energy points in log10.
        """
        from nuVeto.utils import Units

        total_pf_dict = {}
        conv_pf_dict = {}
        pr_pf_dict = {}
        total_spline_dict = {}
        conv_spline_dict = {}
        pr_spline_dict = {}
        for key in ptype_converter.keys():
            shape = (len(energy_grid), len(theta_grid))
            total_pf_dict[key] = np.ones(shape)
            conv_pf_dict[key] = np.ones(shape)
            pr_pf_dict[key] = np.ones(shape)

        # transform theta to cos(theta)
        if theta_grid_cos:
            cos_theta_grid = theta_grid
        else:
            cos_theta_grid = np.cos(np.deg2rad(theta_grid))

        settings_base = {
            'pmodel': pmodel,
            'hadr': interaction_model,
            'barr_mods': (),
            'depth': 1950*Units.m,
            'density': density_model,
            'energy_grid': energy_grid,
            'prpl': prpl,
            'ptype_converter': ptype_converter,
        }

        settings_list = []
        results = []
        for i, cos_theta in enumerate(cos_theta_grid):
            settings = deepcopy(settings_base)
            settings['cos_theta'] = cos_theta
            settings_list.append(settings)

        if n_jobs == 1:
            for settings in settings_list:
                results.append(__solve_one_cos_theta__(settings))
        else:
            p = Pool(n_jobs)
            results = p.map(__solve_one_cos_theta__, settings_list)
            p.close()
            p.join()

        for i, result_i in enumerate(results):
            total_pf_dict_i, conv_pf_dict_i, pr_pf_dict_i = result_i
            for key, value in ptype_converter.items():
                total_pf_dict[key][:, i] = total_pf_dict_i[key]
                conv_pf_dict[key][:, i] = conv_pf_dict_i[key]
                pr_pf_dict[key][:, i] = pr_pf_dict_i[key]

        # create splines
        log10_e_grid = np.log10(energy_grid)
        for key, value in ptype_converter.items():
            total_spline_dict[key] = RectBivariateSpline(
                log10_e_grid,
                theta_grid,
                np.log10(np.clip(total_pf_dict[key], eps, float('inf'))),
                s=0,
            )
            conv_spline_dict[key] = RectBivariateSpline(
                log10_e_grid,
                theta_grid,
                np.log10(np.clip(conv_pf_dict[key], eps, float('inf'))),
                s=0,
            )
            pr_spline_dict[key] = RectBivariateSpline(
                log10_e_grid,
                theta_grid,
                np.log10(np.clip(pr_pf_dict[key], eps, float('inf'))),
                s=0,
            )

        spline_dicts = [
            total_spline_dict, conv_spline_dict, pr_spline_dict
        ]

        flux_dicts = [
            total_pf_dict, conv_pf_dict, pr_pf_dict
        ]

        return spline_dicts, flux_dicts, log10_e_grid

    def __get_spline__(
            interaction_model,
            primary_model,
            prpl,
            months,
            theta_grid,
            theta_grid_cos,
            energy_grid,
            ):
        """Get MCEq spline for the provided settings

        Parameters
        ----------
        interaction_model : str
            The interaction model. This is passed on to `MCEqRun`.
        primary_model : str
            The primary model to use. Must be one of:
                GST_3-gen, GST_4-gen, H3a, H4a, poly-gonato, TIG, ZS, ZSP, GH
        prpl : str
            The detector veto probability PDF to use. This must be a valid
            prpl PDF created and available in nuVeto. This option is passed
            on to nuVeto.
        months : list of str
            The months for which to solve the cascade equations. These must be
            provided as a list of month names, e.g. ['January', 'August']. A
            list of splines will be returned of the same length as `months`.
        theta_grid : array_like
            The grid points in theta to evaluate on in degrees.
            If `theta_grid_cos` is True, this is instead cos(theta).
        theta_grid_cos : bool
            If True, `theta_grid` is interpreted as cos(theta),
            i.e. arccos() is applied first.
        energy_grid : array_like
            The energy grid points [in GeV] to evaluate on.

        Returns
        -------
        dict
            The result of MCEq together with the fitted splines.
            See documentation of `get_spline()` for more details.

        Raises
        ------
        AttributeError
            If the provided `primary_model` is unknown.
        """
        log.info('\tCalculating \'{}\' \'{}\''.format(
            interaction_model, primary_model))

        import crflux.models as pm

        splines = {}
        pmodels = {
            "GST_3-gen": (pm.GaisserStanevTilav, "3-gen"),
            "GST_4-gen": (pm.GaisserStanevTilav, "4-gen"),
            "H3a": (pm.HillasGaisser2012, "H3a"),
            "H4a": (pm.HillasGaisser2012, "H4a"),
            "poly-gonato": (pm.PolyGonato, False),
            "TIG": (pm.Thunman, None),
            "ZS": (pm.ZatsepinSokolskaya, 'default'),
            "ZSP": (pm.ZatsepinSokolskaya, 'pamela'),
            "GH": (pm.GaisserHonda, None),
        }

        for i, month in enumerate(months):
            density_model = ('MSIS00', ('SouthPole', month))

            try:
                pmodel = pmodels[primary_model]
            except KeyError:
                raise AttributeError(
                    'primary_model {} unknown. options: {}'.format(
                        primary_model, pmodels.keys()))

            spline_dicts, flux_dicts, log10_e_grid = __solve_month__(
                interaction_model=interaction_model,
                pmodel=pmodel,
                density_model=density_model,
                prpl=prpl,
                theta_grid=theta_grid,
                theta_grid_cos=theta_grid_cos,
                energy_grid=energy_grid,
                n_jobs=n_jobs,
            )

            nuveto_version = pkg_resources.get_distribution('nuVeto').version

            splines[i] = {}
            splines[i]['total_spline_dict'] = spline_dicts[0]
            splines[i]['conv_spline_dict'] = spline_dicts[1]
            splines[i]['pr_spline_dict'] = spline_dicts[2]
            splines[i]['total_pf_dict'] = flux_dicts[0]
            splines[i]['conv_pf_dict'] = flux_dicts[1]
            splines[i]['pr_pf_dict'] = flux_dicts[2]
            splines[i]['nuveto_version'] = nuveto_version
            splines[i]['ic3_labels_version'] = ic3_labels.__version__
            splines[i]['log10_e_grid'] = log10_e_grid
            splines[i]['theta_grid'] = theta_grid
        return splines

    if got_cashier and cached:
        if cache_file is None:
            cache_f = 'nuVeto.cache'
        else:
            cache_f = cache_file
        log.info('\tUsing cache \'{}\''.format(cache_f))

        @cache(cache_file=cache_f, cache_time=np.inf)
        def wrapped_get_spline(
                interaction_model,
                primary_model,
                prpl,
                months,
                theta_grid,
                theta_grid_cos,
                energy_grid,
                ):
            return __get_spline__(
                interaction_model=interaction_model,
                primary_model=primary_model,
                prpl=prpl,
                months=months,
                theta_grid=theta_grid,
                theta_grid_cos=theta_grid_cos,
                energy_grid=energy_grid,
            )

        return wrapped_get_spline(
            interaction_model=interaction_model,
            primary_model=primary_model,
            prpl=prpl,
            months=months,
            theta_grid=theta_grid,
            theta_grid_cos=theta_grid_cos,
            energy_grid=energy_grid,
        )
    else:
        return __get_spline__(
            interaction_model=interaction_model,
            primary_model=primary_model,
            prpl=prpl,
            months=months,
            theta_grid=theta_grid,
            theta_grid_cos=theta_grid_cos,
            energy_grid=energy_grid,
        )


class AtmosphericNuVeto(object):

    """Atmospheric nuVeto Wrapper

    Attributes
    ----------
    min_theta_deg : float, optional
        The minimum value of the theta grid in degrees.
        If `theta_grid_cos` is True, this is instead cos(theta).
    max_theta_deg : float, optional
        The maximum value of the theta grid in degrees.
        If `theta_grid_cos` is True, this is instead cos(theta).
    month_weights : array_like
        A list of probabilities for each month. These are used as weights
        to sample the corresponding month for each MC events.
        These weights can, for instance, be set to the relative livetime in
        each month of the year. This will then account for seasonal variations.
    months : list of str
        A list of months for which the interpolation splines are created.
    random_state : np.random.RandomState
        The random state that is used to draw the month for each MC event.
    splines : dict
        A dictionary containing the MCEq result and fitted splines.
        See documentation of `get_splines()` for more details.
    theta_grid : array_like
        The grid points in theta to evaluate on in degrees.
        If `theta_grid_cos` is True, this is instead cos(theta).
    theta_grid_cos : bool
        If True, `min_theta_deg` and `max_theta_deg` are interpreted as
        cos(theta), i.e. arccos() is applied first.
    """

    def __init__(
            self,
            min_theta_deg=0.,
            max_theta_deg=180.,
            theta_grid_cos=False,
            theta_steps=61,
            min_energy_gev=1e1,
            max_energy_gev=1e8,
            energy_steps=36,
            season='full_year',
            flux_type='total',
            random_state=None,
            **kwargs):
        """Initialize AtmosphericNuVeto Instance

        Parameters
        ----------
        min_theta_deg : float, optional
            The minimum value of the theta grid in degrees.
            If `theta_grid_cos` is True, this is instead cos(theta).
        max_theta_deg : float, optional
            The maximum value of the theta grid in degrees.
            If `theta_grid_cos` is True, this is instead cos(theta).
        theta_grid_cos : bool
            If True, `min_theta_deg` and `max_theta_deg` are interpreted as
            cos(theta), i.e. arccos() is applied first.
        theta_steps : int, optional
            The number of grid points between the specified min and max values.
        min_energy_gev : float, optional
            The minimum value of the energy grid in GeV.
        max_energy_gev : float, optional
            The maximum value of the energy grid in GeV.
        energy_steps : int, optional
            The number of grid points between the specified min and max values.
        season : str, optional
            What season to use. This may either be a single month ,
            for example 'January', or 'full_year' may be used to run MCEq
            for every month of the year.
        flux_type : str, optional
            The flux type to compute. This must be one of
                'total': combined prompt and conv flux
                'pr': prompt neutrino flux
                'conv': conventional neutrino flux
            This will set the default flux type when calling `getFlux()`.
            You may, however, overwrite this defaul by passing an alternative
            flux type to `getFlux()`. Setting this default value allows for
            drop in replacement of other flux implementations in IceCube.
        random_state : np.random.Randomstate or int, optional
            An int or random state to set the seed.
        **kwargs
            Additional keyword arguments. (Not used!)

        Raises
        ------
        ValueError
            Description
        """
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        if season.lower() == 'full_year':
            self.months = [
                'January',
                'February',
                'March',
                'April',
                'May',
                'June',
                'July',
                'August',
                'September',
                'October',
                'November',
                'December',
            ]
        else:
            self.months = [season]

        if flux_type.lower() not in ['total', 'conv', 'pr']:
            raise ValueError('Flux type: {} must be on of {}'.format(
                flux_type.lower(), ['total', 'conv', 'pr']))

        if theta_steps < 4 or energy_steps < 4:
            raise ValueError('Steps must be >= 4, but are {} and {}'.format(
                theta_steps, energy_steps))

        self.flux_type = flux_type
        self.set_month_weights(np.ones_like(self.months, dtype=float))
        self.min_theta = min_theta_deg
        self.max_theta = max_theta_deg
        if theta_grid_cos:
            self.min_theta = np.cos(np.deg2rad(min_theta_deg))
            self.max_theta = np.cos(np.deg2rad(max_theta_deg))
        self.theta_grid_cos = theta_grid_cos
        self.theta_grid = np.linspace(
            self.min_theta, self.max_theta, theta_steps)
        self.min_energy = min_energy_gev
        self.max_energy = max_energy_gev
        self.energy_grid = np.logspace(
            np.log10(self.min_energy), np.log10(self.max_energy), energy_steps)

        self.splines = None

    def initialize(
            self,
            prpl,
            interaction_model='SIBYLL2.3c',
            primary_model='H3a',
            n_jobs=1,
            cached=True,
            cache_file=CACHE_FILE,
            ):
        """Initialize AtmosphericNuVeto instance

        This will compute the splines or retrieve these from the cache
        if `cached` is True and if these have been previously computed.
        This method must be called prior to calls to `getFlux()`.

        Parameters
        ----------
        prpl : str
            The detector veto probability PDF to use. This must be a valid
            prpl PDF created and available in nuVeto. This option is passed
            on to nuVeto.
        interaction_model : str
            The interaction model. This is passed on to `MCEqRun`.
        primary_model : str
            The primary model to use. Must be one of:
                GST_3-gen, GST_4-gen, H3a, H4a, poly-gonato, TIG, ZS, ZSP, GH
        n_jobs : int, optional
            Number of jobs to compute the splines. The grid evaluation points
            along zenith are distributed over the specified `n_jobs`.
        cached : bool, optional
            If True, the result will be cached and if already computed, it will
            be retrieved from cache. This avoids recomputation of MCEq, which
            is recommended in order to reduce computation time.
        cache_file : str, optional
            The path to the cache file to use.
        """
        self.splines = get_spline(
            interaction_model=interaction_model,
            primary_model=primary_model,
            prpl=prpl,
            months=self.months,
            theta_grid=self.theta_grid,
            theta_grid_cos=self.theta_grid_cos,
            energy_grid=self.energy_grid,
            n_jobs=n_jobs,
            cached=cached,
            cache_file=cache_file,
        )

        nuveto_version = pkg_resources.get_distribution('nuVeto').version

        # throw warning if there is a version mis-match.
        for key, spline in self.splines.items():
            msg = (
                'Cached file was created with {} version {}, '
                'but this is version {}!'
            )
            if nuveto_version != spline['nuveto_version']:
                log.warning(msg.format(
                    'nuVeto', spline['nuveto_version'], nuveto_version))

            if ic3_labels.__version__ != spline['ic3_labels_version']:
                log.warning(msg.format(
                    'ic3_labels',
                    spline['ic3_labels_version'],
                    ic3_labels.__version__,
                ))

    def set_month_weights(self, month_weights):
        """Summary

        Parameters
        ----------
        month_weights : array_like
            A list of probabilities for each month (these will be normalized
            internally). These are used as weights to sample the corresponding
            month for each MC events. These weights can, for instance, be set
            to the relative livetime in each month of the year.
            This will then account for seasonal variations.

        Raises
        ------
        AttributeError
            If the length of the provided `muon_weights` does not match the
            length of the specified months.
        """
        if len(month_weights) != len(self.months):
            raise AttributeError(
                'month_weights needs to be of the same '
                'length like self.months.'
            )
        self.month_weights = month_weights / np.sum(month_weights)

    def get_passing_fraction(
            self,
            ptype,
            energy,
            costheta,
            selected_month=None,
            random_state=None,
            flux_type=None,
            ):
        """Get atmospheric neutrino passing fraction for provided particle

        Parameters
        ----------
        ptype : array_like or int
            The PDG encoding.
            For instance: I3MCWeightDict -> PrimaryNeutrinoType
        energy : array_like or float
            The energy of the primary particle.
            For instance: I3MCWeightDict -> PrimaryNeutrinoEnergy
        costheta : array_like or float
            The cos(zenith) angle of the primary particle.
            For instance: cos(I3MCWeightDict -> PrimaryNeutrinoZenith)
        selected_month : array_like, optional
            The month in which each event occurred. This must be given as
            an array of integer values between [0, 11] if `season` is
            'full_year'. If the `MCEQFlux` instance is initialized with only
            one month as the season, then `selected_month` must not be set.
            If None provided, the corresponding month of each event will be
            sampled via the defined `month_weights`.
        random_state : np.random.Randomstate or int, optional
            An int or random state to set the seed.
            If None provided, the random state will be used that was
            reated during initialization.
        flux_type : str, optional
            The flux type to compute. This must be one of
                'total': combined prompt and conv flux
                'pr': prompt neutrino flux
                'conv': conventional neutrino flux
            If None is provided, the specified default value at
            object instantiation time (__init__()) will be used.

        Returns
        -------
        array_like
            The atmospheric neutrino passing fraction for the given particle.

        Raises
        ------
        RuntimeError
            If AtmosphericNuVeto has not been initialized yet.
        ValueError
            If wrong `flux_type` is provided.
        """
        if self.splines is None:
            raise RuntimeError(
                'No splines calculated! Run \'initialize\' first')

        if len(self.months) == 1 and selected_month is not None:
            raise ValueError(
                'The months may not be set, since the AtmosphericNuVeto '
                + 'instance is initialized with only one month: {}'.format(
                    self.months)
            )

        if flux_type is None:
            flux_type = self.flux_type
        elif flux_type.lower() not in ['total', 'conv', 'pr']:
            raise ValueError('Flux type: {} must be on of {}'.format(
                flux_type.lower(), ['total', 'conv', 'pr']))
            flux_type = flux_type.lower()

        if random_state is None:
            random_state = self.random_state
        elif not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        # convert to numpy arrays and make sure these are at least 1D
        ptype = np.atleast_1d(ptype)
        energy = np.atleast_1d(energy)
        costheta = np.atleast_1d(costheta)

        if len(self.splines) > 1:
            if selected_month is None:
                int_months = np.arange(len(self.splines), dtype=int)
                selected_month = random_state.choice(
                    int_months,
                    replace=True,
                    size=len(energy),
                    p=self.month_weights,
                )
            else:
                selected_month = np.asarray(selected_month, dtype=int)
        else:
            selected_month = list(self.splines.keys())[0]
        passing_fraction = np.ones_like(energy)
        passing_fraction[:] = float('NaN')
        log10_energy = np.log10(energy)
        theta = np.rad2deg(np.arccos(costheta))

        for ptype_i in np.unique(ptype):
            mask_ptype = ptype == ptype_i
            for i in self.splines.keys():
                if isinstance(selected_month, int):
                    idx_ptype = mask_ptype
                else:
                    is_in_month = selected_month == i
                    idx_ptype = np.logical_and(mask_ptype, is_in_month)
                passing_fraction[idx_ptype] = self.splines[i][
                        flux_type + '_spline_dict'][ptype_i](
                    log10_energy[idx_ptype],
                    theta[idx_ptype],
                    grid=False)
        return np.power(10., passing_fraction)
