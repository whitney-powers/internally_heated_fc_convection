"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_avg_structure.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: profiles]
    --subdir_name=<subdir_name>               Name of figure output directory & base name of saved figures [default: rolled_structure]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --roll_writes=<int>                 Number of writes over which to take average [default: 50]
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                    Figure width (inches) [default: 6]
    --row_inch=<in>                   Figure height (inches) [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import RolledProfilePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_file  = int(args['--start_file'])
subdir_name    = args['--subdir_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

roll_writes = int(args['--roll_writes'])

# Create Plotter object, tell it which fields to plot
plotter = RolledProfilePlotter(root_dir, file_dir=data_dir, out_name=subdir_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files)
plotter.setup_grid(num_rows=1, num_cols=3, col_inch=float(args['--col_inch']), row_inch=float(args['--row_inch']))
plotter.add_line('z', 'flux', grid_num=0)
plotter.add_line('z', 'F_conv', grid_num=0)
plotter.add_line('z', 'F_cond', grid_num=0)
plotter.add_line('z', 'F_tot', grid_num=0)
plotter.add_line('z', 'F_KE', grid_num=0)
plotter.add_line('z', 's1_z', grid_num=1)
plotter.add_line('z', 's1', grid_num=2)
# plotter.add_line('z', 'grad', grid_num=1)
# plotter.add_line('z', 'grad_ad', grid_num=1)
# plotter.add_line('z', 'grad_rad', grid_num=1)
# plotter.add_line('z', 'm', grid_num=2)
# plotter.add_line('z', 'm_ad', grid_num=2)
# plotter.add_line('z', 'm_rad', grid_num=2)
plotter.plot_lines()
