#!/usr/bin/env python
# -*- coding: utf-8 -*
''' Functions to compute MESE weights
'''
from __future__ import print_function, division

import numpy as np
from scipy.spatial import ConvexHull

from icecube import dataclasses
from icecube import icetray
from icecube import NewNuFlux
from icecube import AtmosphericSelfVeto
from icecube.icetray.i3logging import log_info, log_warn

from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.utils.cascade import get_cascade_of_primary_nu


def atmosphericFlux(
        neutrinoEnergy,
        neutrinoZenith,
        neutrinoType,
        atmFluxConv,
        atmFluxPrompt):
    """
    Excessively complicated flux_atm calculation.

    This bundle is adapted from Nancy's adaptation-for-mrichman of her dataset
    wrangling scripts.  Each part is more flexible than it really needs to be
    for this application, but I considered it safer to keep it all rather than
    go through the potentially error-prone process of streamlining the code.
    """
    if isinstance(neutrinoEnergy, float):
        neutrinoEnergy = [neutrinoEnergy]
        neutrinoZenith = [neutrinoZenith]
        neutrinoType = [neutrinoType]

    p_types = [dataclasses.I3Particle.ParticleType(t) for t in neutrinoType]
    atmflux = np.zeros(len(neutrinoEnergy))

    badmask = neutrinoEnergy < 10.
    if atmFluxConv is not None:
        conv = atmFluxConv.getFlux(
            p_types,
            neutrinoEnergy, np.cos(neutrinoZenith))
        conv[badmask] = np.nan
        atmflux += conv
    if atmFluxPrompt is not None:
        prompt = atmFluxPrompt.getFlux(
            p_types,
            neutrinoEnergy, np.cos(neutrinoZenith))
        prompt[badmask] = np.nan
        atmflux += prompt
    # return atmflux

    if ((isinstance(neutrinoZenith, float))
            and (isinstance(neutrinoType, str))):
        for i in range(len(neutrinoEnergy)):
            if neutrinoEnergy[i] < 10.:
                atmflux[i] = np.nan
                continue

            conv = 0.
            prompt = 0.
            if atmFluxConv is not None:
                conv = atmFluxConv.getFlux(
                    dataclasses.I3Particle.ParticleType(neutrinoType),
                    neutrinoEnergy[i],
                    np.cos(neutrinoZenith))
            if atmFluxPrompt is not None:
                prompt = atmFluxPrompt.getFlux(
                    dataclasses.I3Particle.ParticleType(neutrinoType),
                    neutrinoEnergy[i],
                    np.cos(neutrinoZenith))
            atmflux[i] = conv+prompt
    else:
        for i in range(len(neutrinoEnergy)):
            if neutrinoEnergy[i] < 10.:
                atmflux[i] = np.nan
                continue
            conv = 0.
            prompt = 0.
            if atmFluxConv is not None:
                conv = atmFluxConv.getFlux(
                    dataclasses.I3Particle.ParticleType(neutrinoType[i]),
                    neutrinoEnergy[i],
                    np.cos(neutrinoZenith[i]))
            if atmFluxPrompt is not None:
                prompt = atmFluxPrompt.getFlux(
                    dataclasses.I3Particle.ParticleType(neutrinoType[i]),
                    neutrinoEnergy[i],
                    np.cos(neutrinoZenith[i]))
            atmflux[i] = conv+prompt

    if len(atmflux) == 1:
        atmflux = atmflux[0]

    return atmflux


class MESEWeights(icetray.I3ConditionalModule):

    """Calculate weights for MESE 7yr cascade ps paper.
    The returned weights are rates in Hz. To obtain number of events, this
    still has to be multiplied by livetime.
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("DatasetType",
                          "Type of dataset. Must be one of: "
                          "'muongun', 'nugen'")
        self.AddParameter("DatasetNFiles", "Number of files")
        self.AddParameter("DatasetNEventsPerRun",
                          "Number of generated events per file")
        self.AddParameter("OutputKey", "Save weights to this frame key.",
                          'MESE_weights')

    def Configure(self):
        self._dataset_type = self.GetParameter("DatasetType")
        self._n_files = self.GetParameter("DatasetNFiles")
        self._n_events_per_run = self.GetParameter("DatasetNEventsPerRun")
        self._ngen = self._n_events_per_run * self._n_files
        self._output_key = self.GetParameter("OutputKey")

        self._dataset_type = self._dataset_type.lower()

        if self._dataset_type not in ['muongun', 'nugen']:
            raise ValueError('Unkown dataset_type: {!r}'.format(dataset_type))

        # get Honda2006
        self.honda = NewNuFlux.makeFlux("honda2006")
        self.honda.knee_reweighting_model = 'gaisserH3a_elbert'
        self.honda.relative_kaon_contribution = .91
        # get self-veto
        self.af = AtmosphericSelfVeto.AnalyticPassingFraction
        self.honda_veto_hese = self.af('conventional', veto_threshold=1.25e3)
        self.honda_veto_mese = self.af('conventional', veto_threshold=1e2)
        # get the sarcevic model for prompt neutrinos
        self.enberg = NewNuFlux.makeFlux("sarcevic_std")
        self.enberg_veto_hese = self.af('charm', veto_threshold=1.25e3)
        self.enberg_veto_mese = self.af('charm', veto_threshold=1e2)
        self.conv_flux_multiplier = 1.07
        self.prompt_flux_multiplier = .2

    def Geometry(self, frame):
        geoMap = frame['I3Geometry'].omgeo
        domPosDict = {(i[0][0], i[0][1]): (i[1].position.x,
                                           i[1].position.y,
                                           i[1].position.z)
                      for i in geoMap if i[1].omtype.name == 'IceCube'}
        points = [
            domPosDict[(31, 1)], domPosDict[(1, 1)],
            domPosDict[(6, 1)], domPosDict[(50, 1)],
            domPosDict[(74, 1)], domPosDict[(72, 1)],
            domPosDict[(78, 1)], domPosDict[(75, 1)],

            domPosDict[(31, 60)], domPosDict[(1, 60)],
            domPosDict[(6, 60)], domPosDict[(50, 60)],
            domPosDict[(74, 60)], domPosDict[(72, 60)],
            domPosDict[(78, 60)], domPosDict[(75, 60)]
            ]
        self._convex_hull = ConvexHull(points)
        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)

    def Physics(self, frame):

        mese_dict = {
            'n_files': self._n_files,
            'n_events_per_run': self._n_events_per_run,
        }

        # get MC info
        energy_true = frame['MCPrimary'].energy
        zenith_true = frame['MCPrimary'].dir.zenith
        azimuth_true = frame['MCPrimary'].dir.azimuth

        # -------
        # NuGen
        # -------
        if self._dataset_type == 'nugen':
            # get oneweight / n_gen
            oneweight = frame['I3MCWeightDict']['OneWeight'] / self._ngen
            true_type = frame['I3MCWeightDict']['PrimaryNeutrinoType']
            is_tau = (np.abs(true_type) == 16).all()

            # calculate astrophysical weights
            mese_dict['weight_E269'] = 2.09e-18 * oneweight * (
                                                    energy_true / 1e5)**-2.69
            mese_dict['weight_E250'] = 2.23e-18 * oneweight * (
                                                    energy_true / 1e5)**-2.5

            # calculate atmospheric weights
            if is_tau:
                mese_dict['weight_conv'] = oneweight * atmosphericFlux(
                        neutrinoEnergy=energy_true,
                        neutrinoZenith=zenith_true,
                        neutrinoType=true_type,
                        atmFluxConv=None,
                        atmFluxPrompt=None,) * 2. * self.conv_flux_multiplier

                mese_dict['weight_prompt'] = oneweight * atmosphericFlux(
                        neutrinoEnergy=energy_true,
                        neutrinoZenith=zenith_true,
                        neutrinoType=true_type,
                        atmFluxConv=None,
                        atmFluxPrompt=None,) * 2. * self.prompt_flux_multiplier
            else:
                mese_dict['weight_conv'] = oneweight * atmosphericFlux(
                        neutrinoEnergy=energy_true,
                        neutrinoZenith=zenith_true,
                        neutrinoType=true_type,
                        atmFluxConv=self.honda,
                        atmFluxPrompt=None,) * 2. * self.conv_flux_multiplier
                mese_dict['weight_prompt'] = oneweight * atmosphericFlux(
                    neutrinoEnergy=energy_true,
                    neutrinoZenith=zenith_true,
                    neutrinoType=true_type,
                    atmFluxConv=None,
                    atmFluxPrompt=self.enberg,)*2.*self.prompt_flux_multiplier

            # ---------------------
            # Atmospheric Self Veto
            # ---------------------
            # get true_depth
            if 'IntersectionPoint' in frame:
                true_depth = frame['IntersectionPoint'].z
            else:
                muon = mu_utils.get_next_muon_daughter_of_nu(
                                                frame, frame['MCPrimary'])
                if muon is None:
                    # no muon exists: cascade
                    cascade = get_cascade_of_primary_nu(frame,
                                                        frame['MCPrimary'],
                                                        convex_hull=None,
                                                        extend_boundary=800)
                    true_depth = cascade.pos.z
                else:
                    entry = mu_utils.get_muon_initial_point_inside(
                                                muon, self._convex_hull)
                    if entry is None:
                        # get closest approach point as entry approximation
                        entry = mu_utils.get_muon_closest_approach_to_center(
                                                            frame, muon)
                    true_depth = entry.z

            # apply self veto
            veto_args = (true_type, energy_true,
                         np.cos(zenith_true),
                         1950. - true_depth
                         )

            if 'IsHese' in frame:
                if frame['IsHese'].value:
                    mese_dict['veto_conv'] = self.honda_veto_hese(*veto_args)
                    mese_dict['veto_prompt'] = self.enberg_veto_hese(
                                                                    *veto_args)
                else:
                    mese_dict['veto_conv'] = self.honda_veto_mese(*veto_args)
                    mese_dict['veto_prompt'] = self.enberg_veto_mese(
                                                                    *veto_args)

            else:
                log_warn('WARNING: IsHese does not exist. Using MESE veto')
                mese_dict['veto_conv'] = self.honda_veto_mese(*veto_args)
                mese_dict['veto_prompt'] = self.honda_veto_mese(*veto_args)

            mese_dict['weight_conv'] *= mese_dict['veto_conv']
            mese_dict['weight_prompt'] *= mese_dict['veto_prompt']
            # ---------------------

        # -------
        # MuonGun
        # -------
        elif self._dataset_type == 'muongun':
            if 'MuonWeight_GaisserH4a' in frame:
                # --- Where does magic number of 1.6 come from? MuonMultiplier
                mese_dict['muon_weight'] = \
                    frame['MuonWeight_GaisserH4a'].value * 1.6 / self._ngen

        # -----------------
        # Experimental Data
        # -----------------
        elif self._dataset_type == 'data':
            mjd = frame['I3EventHeader'].start_time.mod_julian_day_double

        # -----------------------------------------------------
        # final track cut:
        # drop low energy downgoing tracks and duplicate events
        # -----------------------------------------------------
        try:
            # get TrackFit_zenith
            TrackFit_zenith = frame['TrackFit'].dir.zenith

            # get energy_millipede
            energy_millipede = frame['MillipedeDepositedEnergy'].value

            # mask events
            track_mask = data_dict['is_cascade_reco'] | \
                ~((np.cos(TrackFit_zenith) > 0.3) & (energy_millipede < 10e3))

            if self._dataset_type in ['muongun', 'nugen']:
                uniq_mask = np.r_[True, np.diff(energy_true) != 0]
            else:
                uniq_mask = np.r_[True, np.diff(mjd) != 0]
            mese_dict['passed_final_track_cut'] = track_mask & uniq_mask
        except Exception as e:
            # log_warn(e)
            pass
        # -----------------------------------------------------
        for k, item in mese_dict.items():
            mese_dict[k] = float(item)
        frame[self._output_key] = dataclasses.I3MapStringDouble(mese_dict)

        self.PushFrame(frame)
