# No linting is done here, as this is a dirty hack to cache simprod data for
# weighting for offline usage.
# ruff: noqa: F821
"""Dirty Hack to cache simprod data for weighting for offline usage

Code is adopted from `from_simprod` function:
https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/
combo/trunk/weighting/python/weighting.py#L823

Line 823 is where to get the database PW from.
"""
from __future__ import print_function, division
import os
import importlib
from functools import partial

import click
import yaml
import warnings
import numpy as np

from icecube.weighting import weighting
from icecube.icetray import I3Units


# GLOBALS
_sql_types = dict(string=str, int=int, double=float, float=float, bool=bool)
NOTHING = object()


def get(collection, key, default=NOTHING, type=NOTHING):
    """
    Get with optional type coercion
    """
    if default is NOTHING:
        value = collection[key]
    else:
        value = collection.get(key, default)
    if type is NOTHING:
        return value
    else:
        return type(value)


def get_steering(cursor, dataset_id):
    cursor.execute(
        "SELECT name, type, value FROM steering_parameter WHERE dataset_id=%s",
        (dataset_id,),
    )
    steering = {}
    for name, typus, value in cursor.fetchall():
        try:
            steering[name] = _sql_types[typus](value)
        except ValueError:
            steering[name] = value
            pass
    return steering


def _import_mysql():
    "Import the flavor of the month"
    import importlib

    for impl in "MySQLdb", "mysql.connector", "pymysql":
        try:
            mysql = importlib.import_module(impl)
            return mysql
        except ImportError:
            pass
    raise ImportError("No MySQL bindings found!")


def get_generator_settings(
    dataset_id,
    database_pwd,
    use_muongun=False,
    database="vm-i3simprod.icecube.wisc.edu",
):
    """Get Settings to create generator

    Parameters
    ----------
    dataset_id : int
        The database id.
    database_pwd : str
        The database password. This is provided in the `from_simprod` script
        at combo/trunk/weighting/python/weighting.py#L823.
    database : str, optional
        The database url.

    Returns
    -------
    dict
        A dict with the generator settings.
            class: str
                Name of the Generator class.
            multiplier: float
                The multiplier to apply on the generator.
            kwargs: dict
                Keyword arguments that get passed to generator class
    """
    generator_data = {}

    import re

    mysql = _import_mysql()

    try:
        db = mysql.connect(
            host=database,
            user="i3simprod-ro",
            passwd=database_pwd,
            db="i3simprod",
        )
    except mysql.OperationalError as e:
        reason = e.args[1]
        reason += " This might happen if you tried to connect to the simprod database from many cluster jobs in parallel. Don't do that. Instead, query the generator for your dataset once, and pass it to your jobs in a file."
        raise mysql.OperationalError(e.args[0], reason)
    cursor = db.cursor()

    if isinstance(dataset_id, str):
        raise UnboundLocalError
    cursor.execute(
        "SELECT COUNT(*) FROM dataset WHERE dataset_id=%s", (dataset_id,)
    )
    if cursor.fetchone()[0] == 0:
        raise ValueError(
            "Dataset %s does not exist in the simprod database"
            % repr(dataset_id)
        )

    # In case this is a post-processed set, chase the chain back until we hit the real generated set
    while True:
        cursor.execute(
            "SELECT description FROM dataset WHERE dataset_id=%s",
            (dataset_id,),
        )
        description = cursor.fetchone()[0]
        match = (
            re.match(
                r".*(from|of) dataset (\d{4,5})", description, re.IGNORECASE
            )
            if description
            else None
        )
        if match:
            dataset_id = int(match.group(2))
        else:
            try:
                try:
                    parent_id = get_steering(cursor, dataset_id)[
                        "inputdataset"
                    ]
                except KeyError:
                    parent_id = get_steering(cursor, dataset_id)[
                        "MCPE_dataset"
                    ]
                # check if this is an IceTop dataset, in which case we should
                # stop before we get to generation level
                parent = get_steering(cursor, parent_id)
                if "CORSIKA::platform" in parent:
                    break
                dataset_id = parent_id
            except KeyError:
                break

    # query category and number of completed files
    cursor.execute(
        "SELECT category FROM dataset JOIN simcat ON dataset.simcat_id=simcat.simcat_id and dataset.dataset_id=%s",
        (dataset_id,),
    )
    row = cursor.fetchone()
    category = row[0]
    steering = get_steering(cursor, dataset_id)
    get_steering_param = partial(get, steering)

    if category == "Test":
        if steering["mctype"] == "corsika":
            category = "CORSIKA-in-ice"
        elif steering["mctype"].startswith("nugen"):
            category = "neutrino-generator"

    def _coerce_tray_parameter(row):
        if not row:
            return None
        if row[1] in _sql_types:
            try:
                return _sql_types[row[1]](row[2])
            except ValueError:
                # not a literal, must be a function
                return SimprodFunction(
                    row[2], get_steering(cursor, dataset_id)
                )
        else:
            cursor.execute(
                "SELECT value FROM carray_element WHERE cparameter_id=%s",
                (row[0],),
            )
            return [float(v[0]) for v in cursor.fetchall()]

    def get_tray_parameter(dataset_id, key, klass=None):
        if klass is None:
            cursor.execute(
                "SELECT cparameter_id, type, value FROM cparameter WHERE dataset_id=%s AND name=%s ORDER BY tray_index ASC",
                (dataset_id, key),
            )
        else:
            cursor.execute(
                "SELECT cparameter_id, type, value FROM cparameter INNER JOIN (module_pivot, module) ON (module_pivot.module_id=module.module_id AND cparameter.module_pivot_id=module_pivot.module_pivot_id) WHERE module_pivot.dataset_id=%s AND cparameter.name=%s AND module.class=%s ORDER BY cparameter.tray_index ASC",
                (dataset_id, key, klass),
            )
        values = list(map(_coerce_tray_parameter, cursor.fetchall()))
        if len(values) == 0:
            return None
        elif len(values) == 1:
            return values[0]
        else:
            return values

    def get_metaproject(dataset_id, tray_name, tray_index=None):
        """
        Get metaproject version for a tray by name, or if that fails, by index
        """
        cursor.execute(
            "SELECT metaproject.name, metaproject.major_version, metaproject.minor_version, metaproject.patch_version FROM tray JOIN metaproject_pivot ON tray.tray_index=metaproject_pivot.tray_index AND tray.dataset_id=metaproject_pivot.dataset_id JOIN metaproject ON metaproject_pivot.metaproject_id=metaproject.metaproject_id WHERE tray.dataset_id=%s AND tray.name=%s",
            (dataset_id, tray_name),
        )
        row = cursor.fetchone()
        if row is None and tray_index is not None:
            cursor.execute(
                "SELECT metaproject.name, metaproject.major_version, metaproject.minor_version, metaproject.patch_version FROM tray JOIN metaproject_pivot ON tray.tray_index=metaproject_pivot.tray_index AND tray.dataset_id=metaproject_pivot.dataset_id JOIN metaproject ON metaproject_pivot.metaproject_id=metaproject.metaproject_id WHERE tray.dataset_id=%s AND tray.tray_index=%s",
                (dataset_id, tray_index),
            )
            row = cursor.fetchone()
        metaproject, major, minor, patch = row
        prerelease = None
        if "-" in patch:
            patch, prerelease = patch.split("-")
        return (metaproject, int(major), int(minor), int(patch), prerelease)

    if category == "neutrino-generator":
        if "NUGEN::elogmin" in steering:
            emin, emax = 10 ** get_steering_param(
                "NUGEN::elogmin", type=float
            ), 10 ** get_steering_param("NUGEN::elogmax", type=float)
        elif "NUGEN::from_energy" in steering:
            emin, emax = get_steering_param(
                "NUGEN::from_energy", type=float
            ), get_steering_param("NUGEN::to_energy", type=float)
        else:
            emin, emax = get_steering_param(
                "NUGEN::emin", type=float
            ), get_steering_param("NUGEN::emax", type=float)
        nugen_kwargs = {}
        if "NUGEN::injectionradius" in steering:
            nugen_kwargs["InjectionRadius"] = get_steering_param(
                "NUGEN::injectionradius", type=float
            )
        elif "NUGEN::cylinder_length" in steering:
            nugen_kwargs["CylinderHeight"] = get_steering_param(
                "NUGEN::cylinder_length", type=float
            )
            nugen_kwargs["CylinderRadius"] = get_steering_param(
                "NUGEN::cylinder_radius", type=float
            )
        if get_metaproject(dataset_id, "nugen", 0)[1:] >= (4, 1, 6):
            nugen_kwargs["InjectionMode"] = "Cylinder"
        # generator = NeutrinoGenerator(
        #     NEvents=steering['nevents'],
        #     FromEnergy     =emin,
        #     ToEnergy       =emax,
        #     GammaIndex     =get_steering_param('NUGEN::gamma', type=float),
        #     NeutrinoFlavor =get_steering_param('NUGEN::flavor'),
        #     ZenithMin      =get_steering_param('NUGEN::zenithmin', type=float)*I3Units.deg,
        #     ZenithMax      =get_steering_param('NUGEN::zenithmax', type=float)*I3Units.deg,
        #     **nugen_kwargs)

        # write generator data
        generator_data["class"] = (
            "icecube.weighting.weighting.NeutrinoGenerator"
        )
        generator_data["multiplier"] = None
        generator_data["kwargs"] = dict(
            NEvents=steering["nevents"],
            FromEnergy=emin,
            ToEnergy=emax,
            GammaIndex=get_steering_param("NUGEN::gamma", type=float),
            NeutrinoFlavor=get_steering_param("NUGEN::flavor"),
            ZenithMin=get_steering_param("NUGEN::zenithmin", type=float)
            * I3Units.deg,
            ZenithMax=get_steering_param("NUGEN::zenithmax", type=float)
            * I3Units.deg,
            **nugen_kwargs,
        )

    elif category == "CORSIKA-in-ice":
        composition = steering.get("composition", "5-component")
        if composition.startswith("5-component") or composition == "jcorsika":
            gamma = get_tray_parameter(dataset_id, "pgam")
            if gamma is None:
                gamma = [-2] * 5
            else:
                gamma = [-abs(v) for v in gamma]
            norm = get_tray_parameter(dataset_id, "pnorm")
            if norm is None:
                norm = [10.0, 5.0, 3.0, 2.0, 1.0]
            if (
                get_tray_parameter(dataset_id, "CutoffType")
                == "EnergyPerNucleon"
            ):
                LowerCutoffType = "EnergyPerNucleon"
            else:
                LowerCutoffType = "EnergyPerParticle"
            UpperCutoffType = get_tray_parameter(dataset_id, "UpperCutoffType")
            if UpperCutoffType is None:
                corsika_version = get_tray_parameter(
                    dataset_id, "CorsikaVersion"
                )
                if isinstance(corsika_version, list):
                    corsika_version = corsika_version[-1]
                if corsika_version is None or "5comp" in corsika_version:
                    # 5-component dCORSIKA only supports a lower cutoff
                    UpperCutoffType = "EnergyPerParticle"
                elif get_metaproject(dataset_id, "generate", 0)[1] >= 4:
                    #  Upper cutoff type appeared in IceSim 4, and defaults to the lower cutoff type
                    UpperCutoffType = LowerCutoffType
                else:
                    UpperCutoffType = "EnergyPerParticle"
            length = get_tray_parameter(
                dataset_id,
                "length",
                "icecube.simprod.generators.CorsikaGenerator",
            )
            if length is None:
                if "CORSIKA::length" in steering:
                    length = (
                        get_steering_param("CORSIKA::length", type=float)
                        * I3Units.m
                    )
                else:
                    length = 1600 * I3Units.m
                    warnings.warn(
                        "No target cylinder length for dataset {dataset_id}! Assuming {length:.0f} m".format(
                            **locals()
                        )
                    )
            radius = get_tray_parameter(
                dataset_id,
                "radius",
                "icecube.simprod.generators.CorsikaGenerator",
            )
            if radius is None:
                if "CORSIKA::radius" in steering:
                    radius = (
                        get_steering_param("CORSIKA::radius", type=float)
                        * I3Units.m
                    )
                else:
                    radius = 800 * I3Units.m
                    warnings.warn(
                        "No target cylinder length for dataset {dataset_id}! Assuming {radius:.0f} m".format(
                            **locals()
                        )
                    )
            if use_muongun:
                from icecube import MuonGun

                nevents = get_steering_param("CORSIKA::showers", type=int)
                if gamma == [-2.0] * 5 and norm == [10.0, 5.0, 3.0, 2.0, 1.0]:
                    model = "Standard5Comp"
                elif gamma == [-2.6] * 5 and norm == [3.0, 2.0, 1.0, 1.0, 1.0]:
                    model = "CascadeOptimized5Comp"
                else:
                    raise ValueError("Unknown CORSIKA configuration!")
                # generator = nevents*MuonGun.corsika_genprob(model)

                # write generator data
                generator_data["class"] = "icecube.MuonGun.corsika_genprob"
                generator_data["kwargs"] = dict(config=model)
                generator_data["multiplier"] = nevents

            else:
                oversampling = get_steering_param("oversampling", 1, int)
                # generator = FiveComponent(oversampling*get_steering_param('CORSIKA::showers', type=int),
                #     emin=get_steering_param('CORSIKA::eprimarymin', type=float)*I3Units.GeV,
                #     emax=get_steering_param('CORSIKA::eprimarymax', type=float)*I3Units.GeV,
                #     normalization=norm, gamma=gamma,
                #     LowerCutoffType=LowerCutoffType, UpperCutoffType=UpperCutoffType,
                #     height=length, radius=radius)

                # write generator data
                generator_data["class"] = (
                    "icecube.weighting.weighting.FiveComponent"
                )
                generator_data["kwargs"] = dict(
                    nevents=oversampling
                    * get_steering_param("CORSIKA::showers", type=int),
                    emin=get_steering_param("CORSIKA::eprimarymin", type=float)
                    * I3Units.GeV,
                    emax=get_steering_param("CORSIKA::eprimarymax", type=float)
                    * I3Units.GeV,
                    normalization=norm,
                    gamma=gamma,
                    LowerCutoffType=LowerCutoffType,
                    UpperCutoffType=UpperCutoffType,
                    height=length,
                    radius=radius,
                )
                generator_data["multiplier"] = None

        elif composition.startswith("polygonato") or composition.startswith(
            "Hoerandel"
        ):
            if use_muongun:
                from icecube import MuonGun

                length = (
                    get_steering_param("CORSIKA::length", type=float)
                    * I3Units.m
                )
                radius = (
                    get_steering_param("CORSIKA::radius", type=float)
                    * I3Units.m
                )
                area = np.pi**2 * radius * (radius + length)
                areanorm = 0.131475115 * area
                # generator = (steering['CORSIKA::showers']/areanorm)*MuonGun.corsika_genprob('Hoerandel5')

                # write generator data
                generator_data["class"] = "icecube.MuonGun.corsika_genprob"
                generator_data["kwargs"] = dict(config="Hoerandel5")
                generator_data["multiplier"] = (
                    steering["CORSIKA::showers"] / areanorm
                )
            else:
                # generator = Hoerandel(steering['CORSIKA::showers'],
                #     emin=get_steering_param('CORSIKA::eprimarymin', type=float)*I3Units.GeV,
                #     emax=get_steering_param('CORSIKA::eprimarymax', type=float)*I3Units.GeV,
                #     dslope=get_steering_param('CORSIKA::dslope', type=float),
                #     height=get_steering_param('CORSIKA::length', type=float)*I3Units.m,
                #     radius=get_steering_param('CORSIKA::radius', type=float)*I3Units.m)

                # write generator data
                generator_data["class"] = (
                    "icecube.weighting.weighting.Hoerandel"
                )
                generator_data["kwargs"] = dict(
                    nevents=steering["CORSIKA::showers"],
                    emin=get_steering_param("CORSIKA::eprimarymin", type=float)
                    * I3Units.GeV,
                    emax=get_steering_param("CORSIKA::eprimarymax", type=float)
                    * I3Units.GeV,
                    dslope=get_steering_param("CORSIKA::dslope", type=float),
                    height=get_steering_param("CORSIKA::length", type=float)
                    * I3Units.m,
                    radius=get_steering_param("CORSIKA::radius", type=float)
                    * I3Units.m,
                )
                generator_data["multiplier"] = None

    elif category == "CORSIKA-ice-top":

        # get the parent (generator) dataset, as the generator parameters may
        # be buried several generations back
        substeering = steering
        while not (
            "CORSIKA::ebin" in substeering and "CORSIKA::radius" in substeering
        ):
            try:
                substeering = get_steering(cursor, substeering["inputdataset"])
            except KeyError:
                # sampling radius is in the topsimulator config
                radius = get_tray_parameter(
                    dataset_id,
                    "r",
                    "icecube.simprod.modules.IceTopShowerGenerator",
                )
                break
        else:
            # sampling radius is a steering parameter
            if isinstance(substeering["CORSIKA::radius"], str):
                radius = SimprodFunction(
                    substeering["CORSIKA::radius"], substeering
                )
            else:

                def radius(CORSIKA_ebin):
                    return substeering["CORSIKA::radius"]

        get_substeering_param = partial(get, substeering)

        # logarithmic energy bin is a function of the procnum
        ebin = SimprodFunction(substeering["CORSIKA::ebin"], substeering)

        # check that the energy steps are spaced like we expect
        dlogE = ebin(procnum=1) - ebin(procnum=0)
        assert dlogE > 0, "Subsequent procnums end up in different energy bins"
        eslope = get_substeering_param("CORSIKA::eslope", type=float)
        assert (
            eslope == -1
        ), "Weighting scheme only makes sense for E^-1 generation"

        try:
            oversampling = get_substeering_param(
                "CORSIKA::oversampling", type=int
            )
        except KeyError:
            oversampling = get_tray_parameter(
                dataset_id,
                "samples",
                "icecube.simprod.modules.IceTopShowerGenerator",
            )

        ctmin = np.cos(
            np.radians(get_substeering_param("CORSIKA::cthmax", type=float))
        )
        ctmax = np.cos(
            np.radians(get_substeering_param("CORSIKA::cthmin", type=float))
        )
        # projected area x solid angle: pi^2 r^2 (ctmax^2 - ctmin^2)

        emin = get_substeering_param("CORSIKA::ebin_first", type=float)
        emax = get_substeering_param("CORSIKA::ebin_last", type=float)
        num_ebins = int((emax - emin) / dlogE) + 1
        ebins = np.linspace(emin, emax, num_ebins)

        # go up further levels if necessary
        while "CORSIKA::primary" not in substeering:
            substeering = get_steering(cursor, substeering["inputdataset"])
        try:
            primary = substeering[
                "PRIMARY::%s" % substeering["CORSIKA::primary"]
            ]
        except KeyError:
            primary = getattr(ParticleType, substeering["CORSIKA::primary"])

        # number of showers in bin
        if isinstance(substeering["CORSIKA::showers"], str):
            nshowers = SimprodFunction(
                substeering["CORSIKA::showers"], substeering
            )
        elif "CORSIKA::showers" in substeering:

            def nshowers(CORSIKA_ebin):
                return int(substeering["CORSIKA::showers"])

        else:

            def nshowers(CORSIKA_ebin):
                return 1.0

        bin_r_n = np.array(
            [
                (eb, radius(CORSIKA_ebin=eb), nshowers(CORSIKA_ebin=eb))
                for eb in ebins
            ]
        )
        probs = []
        for (r, n), ebins in itertools.groupby(
            bin_r_n, lambda pair: (pair[1], pair[2])
        ):
            ebins = [pair[0] for pair in ebins]
            probs.append(
                PowerLaw(
                    eslope,
                    10 ** ebins[0],
                    10 ** (ebins[-1] + dlogE),
                    n * len(ebins),
                    area=AngularGenerationDistribution(
                        ctmin, ctmax, Circle(r)
                    ),
                    particle_type=ParticleType.values[primary],
                )
            )

        # turn into a collection
        generator = GenerationProbabilityCollection(probs).to_PDG()
        # normalize to relative proportion in each bin
        generator /= sum([prob.nevents for prob in probs])
        # and scale to total number of showers with over-sampling
        generator *= oversampling

        # write generator data
        raise NotImplementedError("CORSIKA-ice-top not yet supported")
        # generator_data['class'] = 'icecube.weighting.weighting.GenerationProbabilityCollection'
        # generator_data['kwargs'] = dict(
        #     probs=probs,
        #     probs=probs,
        # )
        # generator_data['multiplier'] = None

    else:
        raise ValueError(
            "No weighting scheme implemented for %s simulations" % (category)
        )
    cursor.close()
    db.close()
    return generator_data


def load_class(full_class_string):
    """
    dynamically load a class from a string

    Parameters
    ----------
    full_class_string : str
        The full class string to the given python clas.
        Example:
            my_project.my_module.my_class

    Returns
    -------
    python class
        PYthon class defined by the 'full_class_string'
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def get_generator(cache_file, dataset_id):
    """Get

    Parameters
    ----------
    cache_file : str
        The local yaml cache file. This must contain the `dataset_id`.
    dataset_id : int
        The dataset id.
    """
    with open(cache_file, "r") as stream:
        cfg = yaml.full_load(stream)

    generator_class = load_class(cfg[dataset_id]["class"])
    generator = generator_class(**cfg[dataset_id]["kwargs"])
    if cfg[dataset_id]["multiplier"] is not None:
        generator *= generator
    return generator


@click.command()
@click.argument("dataset_ids", type=int, nargs=-1)
@click.option("-p", "--password", type=str)
@click.option(
    "-o", "--outfile", default="simprod_data.yaml", help="Name of output file."
)
@click.option("--use_muongun", type=bool, default=False)
def main(dataset_ids, password, outfile, use_muongun):
    """Create a local cache file of Simprod Datasets

    Parameters
    ----------
    dataset_ids : list of int
        The dataset ids.
    password : str
        The database password. This is provided in the `from_simprod` script
        at combo/trunk/weighting/python/weighting.py#L823.
    outfile : str
        The path to where the local cache yaml file will be written to.
    """
    # load yaml file if it exists
    if os.path.exists(outfile):
        print(
            "Found existing file at {}, will append/overwrite entries.".format(
                outfile
            )
        )
        with open(outfile, "r") as stream:
            cfg = yaml.full_load(stream)
    else:
        cfg = {}

    print("Now retrieving data:")
    for dataset_id in dataset_ids:
        print("\t{}...".format(dataset_id))
        cfg[dataset_id] = get_generator_settings(
            dataset_id, database_pwd=password, use_muongun=use_muongun
        )
    print("Done!")

    # save yaml file
    print("Now saving to file: {}".format(outfile))
    with open(outfile, "w") as output:
        yaml.safe_dump(cfg, output, default_flow_style=False)

    print("Now testing if we can obtain generator:")
    for dataset_id in dataset_ids:
        print("\t{}...".format(dataset_id))
        get_generator(outfile, dataset_id)
    print("Done!")


if __name__ == "__main__":
    main()
