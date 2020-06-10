import os
import numpy as np
from matplotlib import pyplot as plt

from icecube import dataclasses, icetray

from ic3_labels.labels.utils import muon as mu_utils
from .utils import get_track_energy_depositions, compute_stochasticity


class MuonLossVisualizer(icetray.I3ConditionalModule):

    """Class to visualize muon energy losses and the associated labels.
    """

    def __init__(self, context):
        """initialize module.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox('OutBox')
        self.AddParameter('plot_dir', 'A directory for the plots', 'plots')
        self.AddParameter('mc_tree_name', 'Name of the I3MCTree', 'I3MCTree')

    def Configure(self):
        """Configures Module and loads BDT from file.
        """
        self.plot_dir = self.GetParameter('plot_dir')
        self.mc_tree_name = self.GetParameter('mc_tree_name')
        self.counter = 0

    def DAQ(self, frame):
        """Process DAQ frame

        Parameters
        ----------
        frame : I3Frame
            The current Q-frame.
        """

        mc_tree = frame[self.mc_tree_name]

        # get in-ice muon:
        muon = mu_utils.get_muon_of_inice_neutrino(frame)

        if muon is not None:
            self.make_plots(mc_tree, muon, file=os.path.join(
                self.plot_dir, 'energy_loss_cdf_{:08d}.png'.format(
                    self.counter)))

        # push frame to next modules
        self.PushFrame(frame)
        self.counter += 1

    def make_plots(self, mc_tree, track,
                   num_losses_to_remove=[0, 5],
                   correct_for_em_loss=True,
                   extend_boundary=300,
                   file=None):
        """Make energy loss plots for the provided track

        Parameters
        ----------
        mc_tree : I3MCTree
            The I3MCTree.
        track : I3Particle.
            The track particle (usually a muon or tau) for which to create
            the energy loss plots
        num_losses_to_remove : list of int, optional
            Number of highest energy losses to remove from CDF calculation.
            These highest energy losses would ideally be handled as separate
            cascades.
        correct_for_em_loss : bool, optional
            If True, energy depositions will be in terms of EM equivalent
            deposited energy.
            If False, the actual (but possibly invisible) energy depositions
            is used.
        extend_boundary : float, optional
            If provided only energy losses within convex hull + extend boundary
            are accepted and considered.
        file : str, optional
            Path to which plot will be saved.

        Raises
        ------
        NotImplementedError
            Description
        """

        if track.type not in [dataclasses.I3Particle.MuMinus,
                              dataclasses.I3Particle.MuPlus]:
            raise NotImplementedError(
                'Particle type {} not yet supported'.format(track.type))

        # Calculate total energy by not removing any cascades
        update_distances, update_energies, _, _ = \
            get_track_energy_depositions(
                mc_tree, track, 0,
                correct_for_em_loss=correct_for_em_loss,
                extend_boundary=extend_boundary)
        total_energy = update_energies[0] - update_energies[-1]

        # Calculate continous losses by removing all cascades
        cont_distances, cont_energies, cont_cascades, _ = \
            get_track_energy_depositions(
                mc_tree, track, 10000,
                correct_for_em_loss=correct_for_em_loss,
                extend_boundary=extend_boundary)
        cont_stochasticity, cont_area_above, cont_area_below = \
            compute_stochasticity(cont_distances, cont_energies)

        # define label format
        label_format = 'N: {:3d} | E: {:3.3f} TeV [{:2.2f}%]'
        label_format += ' | Stoch.: {:3.2f} [{:3.2f}, {:3.2f}]'

        fig, ax = plt.subplots(figsize=(9, 6))
        for num_remove in num_losses_to_remove:

            # compute energy losses
            update_distances, update_energies, cascades, track_updates = \
                get_track_energy_depositions(
                    mc_tree, track, num_remove,
                    correct_for_em_loss=correct_for_em_loss,
                    extend_boundary=extend_boundary)

            if len(update_distances) > 0:

                # compute stochasticity and area above/below diagonal
                stochasticity, area_above, area_below = compute_stochasticity(
                    update_distances, update_energies)

                cum_energies = np.cumsum(-np.diff(update_energies))
                ax.plot(
                    [update_distances[0]] + list(update_distances[1:]),
                    [0] + list(cum_energies / cum_energies[-1]),
                    label=label_format.format(
                        len(cascades),
                        cum_energies[-1] / 1000.,
                        100 * cum_energies[-1]/total_energy,
                        stochasticity,
                        area_below,
                        area_above)
                )

        # Plot continous losses
        cum_energies = np.cumsum(-np.diff(cont_energies))
        ax.plot([cont_distances[0]] + list(cont_distances[1:]),
                [0] + list(cum_energies / cum_energies[-1]),
                label=label_format.format(
                    len(cont_cascades),
                    cum_energies[-1] / 1000.,
                    100 * cum_energies[-1]/total_energy,
                    cont_stochasticity,
                    cont_area_below,
                    cont_area_above))

        ax.axvline(cont_distances[0], linestyle='--', color='0.8')
        ax.axvline(cont_distances[-1], linestyle='--', color='0.8',
                   label='Detector Boundary')
        ax.legend()
        ax.set_xlabel('Distance / meter')
        ax.set_ylabel('CDF')
        ax.set_title('Track Energy: {:3.3f} TeV'.format(track.energy/1000))
        # ax.grid(color='0.8', alpha=0.5)

        if file is not None:
            fig.savefig(file)
        else:
            plt.show()

        plt.close(fig)
