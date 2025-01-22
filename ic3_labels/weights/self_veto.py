from __future__ import print_function, division

import numpy as np
from scipy.spatial import ConvexHull

from icecube import dataclasses, icetray
from icecube import AtmosphericSelfVeto

from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils import tau as tau_utils
from ic3_labels.labels.utils.cascade import get_cascade_of_primary_nu


class AtmosphericSelfVetoModule(icetray.I3ConditionalModule):
    """Compute atmosperic self veto passing fraction"""

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "DatasetType",
            "Type of dataset. Must be one of: "
            "'muongun', 'nugen', 'genie', 'corsika', 'exp'",
        )
        self.AddParameter(
            "OutputKey",
            "Save weights to this frame key.",
            "AtmosphericSelfVetoFactors",
        )
        self.AddParameter(
            "VetoThresholds",
            "Veto thresholds to use.",
            [1e2, 5e2, 1e3, 1.25e3],
        )

    def Configure(self):
        self._dataset_type = self.GetParameter("DatasetType")
        self._output_key = self.GetParameter("OutputKey")
        self.veto_thresholds = self.GetParameter("VetoThresholds")

        self._dataset_type = self._dataset_type.lower()

        if self._dataset_type not in [
            "muongun",
            "nugen",
            "genie",
            "corsika",
            "exp",
        ]:
            raise ValueError(
                "Unknown dataset_type: {!r}".format(self._ataset_type)
            )

        # get self-veto
        af = AtmosphericSelfVeto.AnalyticPassingFraction
        self.conv_vetos = [
            af("conventional", veto_threshold=veto_threshold)
            for veto_threshold in self.veto_thresholds
        ]
        self.prompt_vetos = [
            af("charm", veto_threshold=veto_threshold)
            for veto_threshold in self.veto_thresholds
        ]

    def Geometry(self, frame):
        geoMap = frame["I3Geometry"].omgeo

        domPosDict = {
            (omkey[0], omkey[1]): (
                om.position.x,
                om.position.y,
                om.position.z,
            )
            for omkey, om in geoMap.items()
            if om.omtype.name == "IceCube"
        }
        points = [
            domPosDict[(31, 1)],
            domPosDict[(1, 1)],
            domPosDict[(6, 1)],
            domPosDict[(50, 1)],
            domPosDict[(74, 1)],
            domPosDict[(72, 1)],
            domPosDict[(78, 1)],
            domPosDict[(75, 1)],
            domPosDict[(31, 60)],
            domPosDict[(1, 60)],
            domPosDict[(6, 60)],
            domPosDict[(50, 60)],
            domPosDict[(74, 60)],
            domPosDict[(72, 60)],
            domPosDict[(78, 60)],
            domPosDict[(75, 60)],
        ]
        self._convex_hull = ConvexHull(points)
        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)

    def Physics(self, frame):

        veto_dict = {}

        # -------
        # NuGen
        # -------
        if self._dataset_type in ["nugen", "genie"]:

            # get MC info
            energy_true = frame["MCPrimary"].energy
            zenith_true = frame["MCPrimary"].dir.zenith

            if self._dataset_type == "nugen":
                true_type = frame["I3MCWeightDict"]["PrimaryNeutrinoType"]
            elif self._dataset_type == "genie":
                true_type = frame["MCPrimary"].type

            # --------------------
            # Get muon entry depth
            # --------------------

            # Be more lenient with GENIE sets and catch assert
            if self._dataset_type == "genie":
                try:
                    muon = mu_utils.get_muon_of_inice_neutrino(frame)
                except AssertionError as e:
                    print(e)
                    muon = None
                try:
                    tau = tau_utils.get_tau_of_inice_neutrino(frame)
                except AssertionError as e:
                    print(e)
                    tau = None

            # NuGen sets should not throw an assert
            else:
                tau = tau_utils.get_tau_of_inice_neutrino(frame)
                muon = mu_utils.get_muon_of_inice_neutrino(frame)

            # found a muon
            if muon is not None:
                entry = self._get_particle_entry(muon)
                entry_z = entry.z

            # found a tau
            elif tau is not None:
                entry = self._get_particle_entry(tau)
                entry_z = entry.z

            # no muon or tau exists: cascade
            else:
                cascade = get_cascade_of_primary_nu(
                    frame,
                    frame["MCPrimary"],
                    convex_hull=None,
                    extend_boundary=800,
                )[0]

                if cascade is not None:
                    entry_z = cascade.pos.z
                else:
                    cascade = get_cascade_of_primary_nu(
                        frame,
                        frame["MCPrimary"],
                        convex_hull=None,
                        extend_boundary=float("inf"),
                    )[0]

                    # Muon coming out of hadronic shower?
                    daughters = frame["I3MCTree"].get_daughters(cascade)

                    # collect possible muons from daughters of daughters
                    # e.g. Nu -> Nu + Hadrons -> Mu
                    muons = []
                    for d in daughters:
                        muons.extend(
                            [
                                m
                                for m in frame["I3MCTree"].get_daughters(d)
                                if mu_utils.is_muon(m)
                            ]
                        )
                    if muons:
                        # pick highest energy muon
                        indices = np.argsort([m.energy for m in muons])
                        muon = muons[indices[-1]]
                        entry = self._get_particle_entry(muon)
                        entry_z = entry.z
                    else:
                        entry_z = cascade.pos.z

            # ---------------
            # apply self veto
            # ---------------
            veto_args = (
                true_type,
                energy_true,
                np.cos(zenith_true),
                1950.0 - entry_z,
            )

            veto_dict = {"depth": 1950.0 - entry_z}

            for i, threshold in enumerate(self.veto_thresholds):
                add = "_{:3.0f}GeV".format(threshold)
                veto_dict["conv" + add] = float(self.conv_vetos[i](*veto_args))
                veto_dict["prompt" + add] = float(
                    self.prompt_vetos[i](*veto_args)
                )
            # ---------------

            frame[self._output_key] = dataclasses.I3MapStringDouble(veto_dict)

        self.PushFrame(frame)

    def _get_particle_entry(self, particle):

        entry = mu_utils.get_muon_initial_point_inside(
            particle, self._convex_hull
        )
        if entry is None:
            # get closest approach point as entry approximation
            entry = mu_utils.get_particle_closest_approach_to_position(
                particle, dataclasses.I3Position(0, 0, 0)
            )
        return entry
