'''
Pybindings for I3SimConstants.ShowerParameters, e.g.:
    from icecube.sim_services import I3SimConstants
    ShowerParameters = I3SimConstants.ShowerParameters

Are not available in older icecube meta projects.
This is a python implementation and copy of:

    https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/
    sim-services/trunk/private/sim-services/I3SimConstants.cxx

ToDo:
    Once pybindings are available in icecube metaprojects, deprecate this
    pyhton copy to ensure that code is not being duplicated in different
    areas.
'''
import numpy as np
from icecube.icetray import I3Units
from icecube import dataclasses
from icecube.icetray.i3logging import log_warn


class ShowerParameters(object):

    """ShowerParameters copy of sim_services.I3SimConstants.ShowerParameters

    Attributes
    ----------
    arg : TYPE
        Description
    """

    def __init__(self, particle_type, energy,
                 density=0.9216*(I3Units.g/I3Units.cm3)):

        # initalize variables
        self.a = 0.
        self.b = 0.
        self.emScale = 1.
        self.emScaleSigma = 0.

        # protect against extremely low energies
        # NB: while Equation 4.11 of Leif's Masters' thesis is written in terms
        # of log10, we use the natural log here and divide the energy-scaling
        # coefficients (beta) below by ln(10) to compensate
        self.logE = max(0., np.log(energy))
        self.Lrad = 0.358*(I3Units.g/I3Units.cm3)/density

        self.isElectron = particle_type in [
            dataclasses.I3Particle.ParticleType.EMinus,
            dataclasses.I3Particle.ParticleType.EPlus,
            dataclasses.I3Particle.ParticleType.Brems,
            dataclasses.I3Particle.ParticleType.DeltaE,
            dataclasses.I3Particle.ParticleType.PairProd,
            dataclasses.I3Particle.ParticleType.Gamma,
            # Pi0 decays to 2 gammas and produce EM showers
            dataclasses.I3Particle.ParticleType.Pi0,
            dataclasses.I3Particle.ParticleType.EMinus,
            dataclasses.I3Particle.ParticleType.EMinus,
        ]

        self.isHadron = particle_type in [
            dataclasses.I3Particle.ParticleType.Hadrons,
            dataclasses.I3Particle.ParticleType.Neutron,
            dataclasses.I3Particle.ParticleType.PiPlus,
            dataclasses.I3Particle.ParticleType.PiMinus,
            dataclasses.I3Particle.ParticleType.K0_Long,
            dataclasses.I3Particle.ParticleType.KPlus,
            dataclasses.I3Particle.ParticleType.KMinus,
            dataclasses.I3Particle.ParticleType.PPlus,
            dataclasses.I3Particle.ParticleType.PMinus,
            dataclasses.I3Particle.ParticleType.K0_Short,

            dataclasses.I3Particle.ParticleType.Eta,
            dataclasses.I3Particle.ParticleType.Lambda,
            dataclasses.I3Particle.ParticleType.SigmaPlus,
            dataclasses.I3Particle.ParticleType.Sigma0,
            dataclasses.I3Particle.ParticleType.SigmaMinus,
            dataclasses.I3Particle.ParticleType.Xi0,
            dataclasses.I3Particle.ParticleType.XiMinus,
            dataclasses.I3Particle.ParticleType.OmegaMinus,
            dataclasses.I3Particle.ParticleType.NeutronBar,
            dataclasses.I3Particle.ParticleType.LambdaBar,
            dataclasses.I3Particle.ParticleType.SigmaMinusBar,
            dataclasses.I3Particle.ParticleType.Sigma0Bar,
            dataclasses.I3Particle.ParticleType.SigmaPlusBar,
            dataclasses.I3Particle.ParticleType.Xi0Bar,
            dataclasses.I3Particle.ParticleType.XiPlusBar,
            dataclasses.I3Particle.ParticleType.OmegaPlusBar,
            dataclasses.I3Particle.ParticleType.DPlus,
            dataclasses.I3Particle.ParticleType.DMinus,
            dataclasses.I3Particle.ParticleType.D0,
            dataclasses.I3Particle.ParticleType.D0Bar,
            dataclasses.I3Particle.ParticleType.DsPlus,
            dataclasses.I3Particle.ParticleType.DsMinusBar,
            dataclasses.I3Particle.ParticleType.LambdacPlus,
            dataclasses.I3Particle.ParticleType.WPlus,
            dataclasses.I3Particle.ParticleType.WMinus,
            dataclasses.I3Particle.ParticleType.Z0,
            dataclasses.I3Particle.ParticleType.NuclInt,
        ]

        self.isMuon = particle_type in [
            dataclasses.I3Particle.ParticleType.MuMinus,
            dataclasses.I3Particle.ParticleType.MuPlus,
        ]

        self.isTau = particle_type in [
            dataclasses.I3Particle.ParticleType.TauMinus,
            dataclasses.I3Particle.ParticleType.TauPlus,
        ]

        if ((not self.isHadron) and (not self.isElectron) and (not self.isMuon)
                and (not self.isTau)):
            # if we don't know it but it has a pdg code,
            # it is probably a hadron..
            self.isHadron = True

            # Added safety check: throw error in this case to make sure nothing
            # weird is happenning unkowingly
            # raise ValueError('Unkown particle type {!r}'.format(particle_type))
            log_warn(
                'Unkown particle type {!r}. Assuming this is a hadron!'.format(
                    particle_type)
            )

        if self.isElectron:

            if particle_type == dataclasses.I3Particle.ParticleType.EPlus:
                self.a = 2.00035+0.63190*self.logE
                self.b = self.Lrad/0.63008

            elif particle_type in [
                    dataclasses.I3Particle.ParticleType.Gamma,
                    dataclasses.I3Particle.ParticleType.Pi0,  # gamma, pi0
                    ]:

                self.a = 2.83923+0.58209*self.logE
                self.b = self.Lrad/0.64526

            else:
                self.a = 2.01849+0.63176*self.logE
                self.b = self.Lrad/0.63207

        elif self.isHadron:

            self.E0 = 0.
            self.m = 0.
            self.f0 = 1.
            self.rms0 = 0.
            self.gamma = 0.

            if particle_type == dataclasses.I3Particle.ParticleType.PiMinus:
                self.a = 1.69176636+0.40803489 * self.logE
                self.b = self.Lrad / 0.34108075
                self.E0 = 0.19826506
                self.m = 0.16218006
                self.f0 = 0.31859323
                self.rms0 = 0.94033488
                self.gamma = 1.35070162

            elif particle_type == dataclasses.I3Particle.ParticleType.K0_Long:
                self.a = 1.95948974+0.34934666 * self.logE
                self.b = self.Lrad / 0.34535151
                self.E0 = 0.21687243
                self.m = 0.16861530
                self.f0 = 0.27724987
                self.rms0 = 1.00318874
                self.gamma = 1.37528605

            elif particle_type == dataclasses.I3Particle.ParticleType.PPlus:
                self.a = 1.47495778+0.40450398 * self.logE
                self.b = self.Lrad / 0.35226706
                self.E0 = 0.29579368
                self.m = 0.19373018
                self.f0 = 0.02455403
                self.rms0 = 1.01619344
                self.gamma = 1.45477346

            elif particle_type == dataclasses.I3Particle.ParticleType.Neutron:
                self.a = 1.57739060+0.40631102 * self.logE
                self.b = self.Lrad / 0.35269455
                self.E0 = 0.66725124
                self.m = 0.19263595
                self.f0 = 0.17559033
                self.rms0 = 1.01414337
                self.gamma = 1.45086895

            elif particle_type == dataclasses.I3Particle.ParticleType.PMinus:
                self.a = 1.92249171+0.33701751 * self.logE
                self.b = self.Lrad / 0.34969748
                self.E0 = 0.29579368
                self.m = 0.19373018
                self.f0 = 0.02455403
                self.rms0 = 1.01094637
                self.gamma = 1.50438415

            else:
                self.a = 1.58357292+0.41886807 * self.logE
                self.b = self.Lrad / 0.33833116
                self.E0 = 0.18791678
                self.m = 0.16267529
                self.f0 = 0.30974123
                self.rms0 = 0.95899551
                self.gamma = 1.35589541

            e = max(2.71828183, energy)
            self.emScale = 1. - pow(e/self.E0, -self.m)*(1.-self.f0)
            self.emScaleSigma = \
                self.emScale*self.rms0*pow(np.log(e), -self.gamma)

        else:
            raise ValueError('Particle type {!r} is not a shower'.format(
                                                                particle_type))

        if (energy < 1.*I3Units.GeV):
            self.b = 0.  # this sets the cascade length to 0
