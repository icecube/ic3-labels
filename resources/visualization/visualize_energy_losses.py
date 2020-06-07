#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
#METAPROJECT /mnt/lfs7/user/mhuennefeld/software/icecube/py2-v3.0.1/combo_trunk/build
# -*- coding: utf-8 -*-
import os
import click

from I3Tray import I3Tray
from icecube import icetray, dataio

from ic3_labels.labels.modules.event_generator.visualization import (
    MuonLossVisualizer
)


@click.command()
@click.argument('input_file_pattern', type=click.Path(exists=True),
                required=True, nargs=-1)
@click.option('-o', '--outdir', default='plots',
              help='Name of output directory for plots.')
@click.option('-n', '--num_events', default=30,
              help='Number of plots to generate at maximum.')
def main(input_file_pattern, outdir, num_events):

    # create output directory if necessary
    if outdir != '' and not os.path.isdir(outdir):
        print('\nCreating directory: {}\n'.format(outdir))
        os.makedirs(outdir)

    tray = I3Tray()

    # read in files
    file_name_list = list(input_file_pattern)
    tray.AddModule('I3Reader', 'reader', Filenamelist=file_name_list)

    # Create plots
    tray.AddModule(
        MuonLossVisualizer, 'MuonLossVisualizer',
        plot_dir=outdir)

    tray.AddModule('TrashCan', 'YesWeCan')
    tray.Execute(num_events)


if __name__ == '__main__':
    main()
