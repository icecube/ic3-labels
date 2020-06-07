import os
import numpy as np
from matplotlib import pyplot as plt

from icecube import dataclasses, icetray

from ic3_labels.labels.utils import muon as mu_utils


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
                   num_losses_to_remove=[0, 1, 5, 10, 100],
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

        Raises
        ------
        NotImplementedError
            Description
        """

        if track.type not in [dataclasses.I3Particle.MuMinus,
                              dataclasses.I3Particle.MuPlus]:
            raise NotImplementedError(
                'Particle type {} not yet supported'.format(track.type))

        # get all daughters of track
        daughters = mc_tree.get_daughters(track)

        last_segment = daughters[0]
        last_stoc_loss = None

        # walk through daughters and collect continous and stochastic losses
        stoch_distances = []
        stoch_losses = []
        cont_distances = []
        cont_losses = []
        for daughter in daughters[1:]:
            if daughter.type != track.type:
                # this is probably a stochastic energy loss
                stoch_distances.append((daughter.pos - track.pos).magnitude)
                stoch_losses.append(daughter.energy)
                last_stoc_loss = daughter
            else:
                # this is probably a segment
                cont_distances.append((daughter.pos - track.pos).magnitude)

                # if there is an energy loss in between: take this out
                cont_loss = last_segment.energy - daughter.energy
                if (last_stoc_loss.pos - daughter.pos).magnitude < 0.0001:
                    cont_loss -= last_stoc_loss.energy

                cont_losses.append(cont_loss)
                last_segment = daughter

        stoch_distances = np.array(stoch_distances)
        stoch_losses = np.array(stoch_losses)
        cont_distances = np.array(cont_distances)
        cont_losses = np.array(cont_losses)
        total_energy = np.sum(stoch_losses) + np.sum(cont_losses)

        # sort stochastic losses
        indices = np.argsort(stoch_losses)
        sorted_stoch_losses = stoch_losses[indices]
        sorted_stoch_distances = stoch_distances[indices]

        assert np.all(cont_losses > 0)

        fig, ax = plt.subplots(figsize=(9, 6))

        for num_remove in num_losses_to_remove:

            cum_energies = []
            for distance, cum_cont_energy in zip(cont_distances,
                                                 np.cumsum(cont_losses)):

                # truncate losses
                if num_remove > 0:
                    num_remove = min(num_remove, len(sorted_stoch_losses))
                    trunc_distances = sorted_stoch_distances[:-num_remove]
                    trunc_losses = sorted_stoch_losses[:-num_remove]
                else:
                    trunc_distances = sorted_stoch_distances
                    trunc_losses = sorted_stoch_losses

                # mask losses up to this distance
                mask = trunc_distances <= distance + 1e-4

                # sum up all deposited energy up to this point
                cum_energy = cum_cont_energy + np.sum(trunc_losses[mask])

                cum_energies.append(cum_energy)

            ax.plot(
                cont_distances, cum_energies / cum_energies[-1],
                label='Num removed: {:3d} | Energy: {:3.1f} [{:2.2f}%]'.format(
                    num_remove, cum_energies[-1],
                    100 * cum_energies[-1]/total_energy))

        ax.plot(cont_distances, np.cumsum(cont_losses) / np.sum(cont_losses),
                label='Continous Losses | Enegy: {:3.1f} [{:2.2f}%]'.format(
                    np.sum(cont_losses),
                    100 * np.sum(cont_losses) / total_energy))

        ax.legend()
        ax.set_xlabel('Distance / meter')
        ax.set_ylabel('CDF')
        ax.set_title('Track Energy: {:3.3f} TeV'.format(track.energy/1000))
        ax.grid(color='0.8', alpha=0.5)

        if file is not None:
            fig.savefig(file)
        else:
            plt.show()

        plt.close(fig)
