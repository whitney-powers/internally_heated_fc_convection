"""
Script for plotting a movie of the evolution of a 2D dedalus simulation.  
This script plots time evolution of the fields specified in 'fig_type'

Usage:
    plot_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --static_cbar                       If flagged, don't evolve the cbar with time
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 4]
    --row_inch=<in>                     Number of inches / row [default: 2]

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T - horiz_avg(T), w
                                        [default: 1]
"""
import numpy as np
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter
import logging
logger = logging.getLogger(__name__)


start_fig = int(args['--start_fig'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']

plotter = SlicePlotter(root_dir, file_dir='slices', out_name=fig_name, start_file=start_file, n_files=n_files)

plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch'])}
plotter.setup_grid(num_rows=2, num_cols=3, **plotter_kwargs)
bases_kwargs_x = { 'x_basis' : 'y', 'y_basis' : 'z' }
bases_kwargs_y = { 'x_basis' : 'x', 'y_basis' : 'z' }
bases_kwargs_z = { 'x_basis' : 'x', 'y_basis' : 'y' }
plotter.add_colormesh('s1_midx', remove_x_mean=True, label='Entropy Fluctuations Mid-x', **bases_kwargs_x, cmap_exclusion=0.05)
plotter.add_colormesh('s1_midy', remove_x_mean=True, label='Entropy Fluctuations Mid-y', **bases_kwargs_y, cmap_exclusion=0.05)
plotter.add_colormesh('s1_midz', remove_mean=True, label='Entropy Fluctuations Mid-z', **bases_kwargs_z, cmap_exclusion=0.05)
#plotter.add_colormesh('vorticity', cmap='PiYG', **bases_kwargs,cmap_exclusion=0.005)
#plotter.add_colormesh('Ma', cmap='inferno', pos_def=True, **bases_kwargs)
#plotter.add_colormesh('enstrophy_midx', cmap='BuPu_r', pos_def=True, **bases_kwargs_x)
#plotter.add_colormesh('enstrophy_midy', cmap='BuPu_r', pos_def=True, **bases_kwargs_y)
#plotter.add_colormesh('enstrophy_midz', cmap='BuPu_r', pos_def=True, **bases_kwargs_y)
plotter.add_colormesh('s1_z0.05', remove_mean=True, label='Entropy Fluctuations z=0.05Lz', **bases_kwargs_z, cmap_exclusion=0.05)
plotter.add_colormesh('s1_z0.95', remove_mean=True, label='Entropy Fluctuations z=0.95Lz', **bases_kwargs_z, cmap_exclusion=0.05)
plotter.add_colormesh('enstrophy_z0.05', cmap='BuPu_r', pos_def=True, **bases_kwargs_z)


plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
