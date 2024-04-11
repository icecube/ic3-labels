"""MCEq Flux Models

This script implements the use of MCEq flux models via the IceCube standard
method of getFlux(ptype, energy, costheta). As such it may be used as a
drop-in replacement for other fluxes. Weighting in IceCube is performed
by multiplying the flux by the normalized one weight:

NuGen:
    (with generator)
        weight = p_int * (flux_val / unit) * generator(energy, ptype, costheta)
    (without generator)
        weight = flux_val * one_weight / (type_weight * n_events * n_files)

    with flux_val = flux_object.getFlux(ptype, energy, costheta)

It is recommended to cache the results of MCEq because these take a while
to produce. By default, the cache file is chosen
to be located in the 'resources' directory relative to the location of this
script. You may also set the environment variable 'MCEQ_CACHE_DIR' in order
to choose a different location for the cache file, or pass in an explicit
cache file when initializing the MCEQFlux object.

Environment Variables:

    'MCEQ_CACHE_DIR':
        If provided, the MCEq cache file will be written to this directory.

    'MKL_PATH':
        Path to the MKL libraries. If provided, these are passed on to MCEq.
        Note: the python package can be installed directly with MKL support
        via 'pip install MCEq[MKL]'.

Credit for the vast majority of code in this file goes to Mathis Boerner.
"""

import os
import logging
from copy import deepcopy
import numpy as np
from scipy.interpolate import RectBivariateSpline

import ic3_labels

log = logging.getLogger("MCEqFlux")


# If cashier is available, set up directory for caching of MCEq results
try:
    from ic3_labels.weights.resources.cashier import cache

    got_cashier = True

    if "MCEQ_CACHE_DIR" in os.environ:
        cache_dir = os.environ["MCEQ_CACHE_DIR"]
        log.info(
            "Found 'MCEQ_CACHE_DIR' in environment variables: {}".format(
                cache_dir
            )
        )

        if not os.path.exists(cache_dir):
            log.info("Creating cache directory: {}".format(cache_dir))
            os.makedirs(cache_dir)

        CACHE_FILE = os.path.join(cache_dir, "mceq.cache")

    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CACHE_FILE = os.path.join(script_dir, "resources", "mceq.cache")

    log.info("Using MCEq cache file: {}".format(CACHE_FILE))

except ImportError:
    got_cashier = False
    CACHE_FILE = None
    log.info("Could not import 'cashier'. MCEq results will not be cached!")


# Dictionary that converts ptype -> MCEq type string
PTYPE_CONVERTER = {
    12: "nue",
    -12: "antinue",
    14: "numu",
    -14: "antinumu",
    16: "nutau",
    -16: "antinutau",
    13: "mu+",
    -13: "mu-",
}


def get_spline(
    interaction_model,
    primary_model,
    months,
    theta_grid,
    theta_grid_cos,
    cached=True,
    cache_file=CACHE_FILE,
    cache_read_only=False,
):
    """Get MCEq spline

    Solves the MCEq cascade equations for the given parameters. The equations
    are solved on the provided grid and interpolated.

    Parameters
    ----------
    interaction_model : str
        The interaction model. This is passed on to `MCEqRun`.
    primary_model : str
        The primary model to use. Must be one of:
            GST_3-gen, GST_4-gen, H3a, H4a, poly-gonato, TIG, ZS, ZSP, GH
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
    cached : bool, optional
        If True, the result will be cached, or taken from cache if previously
        already computed. This is recommended, as MCEq takes a while to run.
    cache_file : str, optional
        The path to the cache file to use.
    cache_read_only : bool, optional
        If True, the cache is read only.

    Returns
    -------
    dict
        The result of MCEq together with the fitted splines. The structure is
        as follows:
        {
            # first month provided via `months`
            0: {
                'total_spline_dict': dict of RectBivariateSpline
                    A dictionary with the fitted splines for each particle
                    type for the 'total' flux. The dictionary keys are the
                    PDG particle encodings.
                'conv_spline_dict': dict of RectBivariateSpline
                    A dictionary with the fitted splines for each particle
                    type for the 'conv' flux. The dictionary keys are the
                    PDG particle encodings.
                'pr_spline_dict': dict of RectBivariateSpline
                    A dictionary with the fitted splines for each particle
                    type for the 'pr' flux. The dictionary keys are the
                    PDG particle encodings.
                'total_flux_dict': dict of array_like
                    A dictionary with the total flux for each grid point.
                    This is the result obtained from MCEq for the 'total' flux.
                'conv_flux_dict': dict of array_like
                    A dictionary with the conv flux for each grid point.
                    This is the result obtained from MCEq for the 'conv' flux.
                'pr_flux_dict': dict of array_like
                    A dictionary with the prompt flux for each grid point.
                    This is the result obtained from MCEq for the 'pr' flux.
                'config_updates':   dict
                    A dictionary of config updates that were applied to
                    mceq_config prior to solving the equations.
                'mceq_version' : str
                    The MCEq version that was used to create the splines.
                'ic3_labels_version' : str
                    The version of the ic3-labels package that was used to
                    create the splines.
                'e_grid' : array_like
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
    log.info(
        "Getting Spline for {}; {} (cached={})".format(
            interaction_model, primary_model, cached
        )
    )

    def __solve_month__(
        mceq_run,
        e_grid,
        theta_grid,
        theta_grid_cos,
        ptype_converter=PTYPE_CONVERTER,
        eps=1e-128,
    ):
        """Solve MCEq equations for the provided mceq_run instance.

        Parameters
        ----------
        mceq_run : MCEqRun instance
            The MCEqRun instance. This instance must be configured to use
            the desired geometry and season.
        e_grid : array_like
            The grid of energy points in log10.
        theta_grid : array_like
            The grid points in theta to evaluate on in degrees.
            If `theta_grid_cos` is True, this is instead cos(theta).
        theta_grid_cos : bool
            If True, `theta_grid` is interpreted as cos(theta),
            i.e. arccos() is applied first.
        ptype_converter : dict, optional
            A dictionary that converts PDG encoding to MCEq type string.
        eps : float, optional
            A small float value > 0 that is used to clip the total flux
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
        """
        total_flux_dict = {}
        conv_flux_dict = {}
        pr_flux_dict = {}
        total_spline_dict = {}
        conv_spline_dict = {}
        pr_spline_dict = {}
        for key, value in ptype_converter.items():
            total_flux_dict[key] = np.ones((len(e_grid), len(theta_grid)))
            conv_flux_dict[key] = np.ones((len(e_grid), len(theta_grid)))
            pr_flux_dict[key] = np.ones((len(e_grid), len(theta_grid)))

        for i, theta_i in enumerate(theta_grid):
            if theta_grid_cos:
                theta_i = np.rad2deg(np.arccos(theta_i))
            mceq_run.set_theta_deg(theta_i)
            mceq_run.solve()

            # fill in flux totals
            for key, value in ptype_converter.items():
                total_flux_dict[key][:, i] = mceq_run.get_solution(
                    "total_{}".format(value)
                )
                conv_flux_dict[key][:, i] = mceq_run.get_solution(
                    "conv_{}".format(value)
                )
                pr_flux_dict[key][:, i] = mceq_run.get_solution(
                    "pr_{}".format(value)
                )

        # create splines
        for key, value in ptype_converter.items():
            total_spline_dict[key] = RectBivariateSpline(
                e_grid,
                theta_grid,
                np.log10(np.clip(total_flux_dict[key], eps, float("inf"))),
                s=0,
            )
            conv_spline_dict[key] = RectBivariateSpline(
                e_grid,
                theta_grid,
                np.log10(np.clip(conv_flux_dict[key], eps, float("inf"))),
                s=0,
            )
            pr_spline_dict[key] = RectBivariateSpline(
                e_grid,
                theta_grid,
                np.log10(np.clip(pr_flux_dict[key], eps, float("inf"))),
                s=0,
            )

        spline_dicts = [total_spline_dict, conv_spline_dict, pr_spline_dict]

        flux_dicts = [total_flux_dict, conv_flux_dict, pr_flux_dict]

        return spline_dicts, flux_dicts

    def __get_spline__(
        interaction_model,
        primary_model,
        months,
        theta_grid,
        theta_grid_cos,
    ):
        """Get MCEq spline for the provided settings

        Parameters
        ----------
        interaction_model : str
            The interaction model. This is passed on to `MCEqRun`.
        primary_model : str
            The primary model to use. Must be one of:
                GST_3-gen, GST_4-gen, H3a, H4a, poly-gonato, TIG, ZS, ZSP, GH, GSF
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
        log.info(
            "\tCalculating '{}' '{}'".format(interaction_model, primary_model)
        )

        import mceq_config
        from MCEq import version
        from MCEq.core import MCEqRun
        import crflux.models as pm

        config_updates = {
            "h_obs": 1000.0,
            "debug_level": 1,
        }
        if "MKL_PATH" in os.environ:
            config_updates["MKL_path"] = os.environ["MKL_PATH"]

        splines = {}
        pmodels = {
            "GST_3-gen": (pm.GaisserStanevTilav, "3-gen"),
            "GST_4-gen": (pm.GaisserStanevTilav, "4-gen"),
            "H3a": (pm.HillasGaisser2012, "H3a"),
            "H4a": (pm.HillasGaisser2012, "H4a"),
            "poly-gonato": (pm.PolyGonato, False),
            "TIG": (pm.Thunman, None),
            "ZS": (pm.ZatsepinSokolskaya, "default"),
            "ZSP": (pm.ZatsepinSokolskaya, "pamela"),
            "GH": (pm.GaisserHonda, None),
            "GSF": (pm.GlobalSplineFitBeta, None),
        }

        for i, month in enumerate(months):
            config_updates["density_model"] = (
                "MSIS00_IC",
                ("SouthPole", month),
            )

            # update settings in mceq_config
            # Previous method mceq_config.config is deprecated and resulted
            # in pickle errors for deepcopy.
            for name, value in config_updates.items():
                setattr(mceq_config, name, value)

            try:
                pmodel = pmodels[primary_model]
            except KeyError:
                raise AttributeError(
                    "primary_model {} unknown. options: {}".format(
                        primary_model, pmodels.keys()
                    )
                )
            mceq_run = MCEqRun(
                interaction_model=interaction_model,
                primary_model=pmodel,
                theta_deg=0.0,
                **config_updates
            )
            e_grid = np.log10(deepcopy(mceq_run.e_grid))
            spline_dicts, flux_dicts = __solve_month__(
                mceq_run, e_grid, theta_grid, theta_grid_cos
            )

            splines[i] = {}
            splines[i]["total_spline_dict"] = spline_dicts[0]
            splines[i]["conv_spline_dict"] = spline_dicts[1]
            splines[i]["pr_spline_dict"] = spline_dicts[2]
            splines[i]["total_flux_dict"] = flux_dicts[0]
            splines[i]["conv_flux_dict"] = flux_dicts[1]
            splines[i]["pr_flux_dict"] = flux_dicts[2]
            splines[i]["config_updates"] = deepcopy(config_updates)
            splines[i]["mceq_version"] = version.__version__
            splines[i]["ic3_labels_version"] = ic3_labels.__version__
            splines[i]["e_grid"] = e_grid
            splines[i]["theta_grid"] = theta_grid
        return splines

    if got_cashier and cached:
        if cache_file is None:
            cache_f = "mceq.cache"
        else:
            cache_f = cache_file
        log.info("\tUsing cache '{}'".format(cache_f))

        @cache(cache_file=cache_f, read_only=cache_read_only)
        def wrapped_get_spline(
            interaction_model,
            primary_model,
            months,
            theta_grid,
            theta_grid_cos,
        ):
            return __get_spline__(
                interaction_model=interaction_model,
                primary_model=primary_model,
                months=months,
                theta_grid=theta_grid,
                theta_grid_cos=theta_grid_cos,
            )

        return wrapped_get_spline(
            interaction_model,
            primary_model,
            months,
            theta_grid,
            theta_grid_cos,
        )
    else:
        return __get_spline__(
            interaction_model=interaction_model,
            primary_model=primary_model,
            months=months,
            theta_grid=theta_grid,
            theta_grid_cos=theta_grid_cos,
        )


class MCEQFlux(object):
    """MCQe Flux Wrapper

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
        min_theta_deg=0.0,
        max_theta_deg=180.0,
        theta_grid_cos=False,
        theta_steps=181,
        season="full_year",
        flux_type="total",
        random_state=None,
        **kwargs
    ):
        """Initialize MCEQFlux Instance

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
            You may, however, overwrite this default by passing an alternative
            flux type to `getFlux()`. Setting this default value allows for
            drop in replacement of other flux implementations in IceCube.
        random_state : np.random.Randomstate or int, optional
            An int or random state to set the seed.
        **kwargs
            Additional keyword arguments. (Not used!)
        """
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        if season.lower() == "full_year":
            self.months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            self.months = [season]

        if flux_type.lower() not in ["total", "conv", "pr"]:
            raise ValueError(
                "Flux type: {} must be on of {}".format(
                    flux_type.lower(), ["total", "conv", "pr"]
                )
            )

        self.flux_type = flux_type
        self.set_month_weights(np.ones_like(self.months, dtype=float))
        self.min_theta = min_theta_deg
        self.max_theta = max_theta_deg
        if theta_grid_cos:
            self.min_theta = np.cos(np.deg2rad(min_theta_deg))
            self.max_theta = np.cos(np.deg2rad(max_theta_deg))
        self.theta_grid_cos = theta_grid_cos
        self.theta_grid = np.linspace(
            self.min_theta, self.max_theta, theta_steps
        )

        self.splines = None

    def initialize(
        self,
        interaction_model="SIBYLL2.3c",
        primary_model="H3a",
        cached=True,
        cache_file=CACHE_FILE,
        cache_read_only=False,
    ):
        """Initialize MCEQFlux instance

        This will compute the splines or retrieve these from the cache
        if `cached` is True and if these have been previously computed.
        This method must be called prior to calls to `getFlux()`.

        Parameters
        ----------
        interaction_model : str
            The interaction model. This is passed on to `MCEqRun`.
        primary_model : str
            The primary model to use. Must be one of:
                GST_3-gen, GST_4-gen, H3a, H4a, poly-gonato, TIG, ZS, ZSP, GH
        cached : bool, optional
            If True, the result will be cached and if already computed, it will
            be retrieved from cache. This avoids recomputation of MCEq, which
            is recommended in order to reduce computation time.
        cache_file : str, optional
            The path to the cache file to use.
        cache_read_only : bool, optional
            If True, the cache is read only.
        """
        if cache_file is None:
            cache_file = CACHE_FILE

        self.splines = get_spline(
            interaction_model,
            primary_model,
            self.months,
            self.theta_grid,
            self.theta_grid_cos,
            cached=cached,
            cache_file=cache_file,
            cache_read_only=cache_read_only,
        )

        from MCEq import version

        # throw warning if there is a version mismatch.
        for key, spline in self.splines.items():
            msg = (
                "Cached file was created with {} version {}, "
                "but this is version {}!"
            )
            if version.__version__ != spline["mceq_version"]:
                log.warning(
                    msg.format(
                        "MCEq", spline["mceq_version"], version.__version__
                    )
                )

            if ic3_labels.__version__ != spline["ic3_labels_version"]:
                log.warning(
                    msg.format(
                        "ic3_labels",
                        spline["ic3_labels_version"],
                        ic3_labels.__version__,
                    )
                )

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
                "month_weights needs to be of the same "
                "length like self.months."
            )
        self.month_weights = month_weights / np.sum(month_weights)

    def getFlux(
        self,
        ptype,
        energy,
        costheta,
        selected_month=None,
        random_state=None,
        flux_type=None,
    ):
        """Get flux for provided particle

        The flux is given in GeV^-1 cm^-2 s^-1 sr^-1 and may be used to
        weight NuGen events via the normalized `one_weight`:
            weight = flux * one_weight / (type_weight * n_events * n_files)

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
            The flux for the given particle in GeV^-1 cm^-2 s^-1 sr^-1.

        Raises
        ------
        RuntimeError
            If MCEQFlux has not been initialized yet.
        ValueError
            If wrong `flux_type` is provided.
        """
        if self.splines is None:
            raise RuntimeError("No splines calculated! Run 'initialize' first")

        if len(self.months) == 1 and selected_month is not None:
            raise ValueError(
                "The months may not be set, since the MCEQFlux instance is "
                + "initialized with only one month: {}".format(self.months)
            )

        if flux_type is None:
            flux_type = self.flux_type
        elif flux_type.lower() not in ["total", "conv", "pr"]:
            raise ValueError(
                'Flux type: "{}" must be on of {}'.format(
                    flux_type.lower(), ["total", "conv", "pr"]
                )
            )
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
        flux = np.ones_like(energy)
        flux[:] = float("NaN")
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
                flux[idx_ptype] = self.splines[i][flux_type + "_spline_dict"][
                    ptype_i
                ](log10_energy[idx_ptype], theta[idx_ptype], grid=False)
        return np.power(10.0, flux)
