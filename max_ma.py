"""
Script for calculating maximum mach number from slices output

Usage:
     max_ma.py <end_file>
"""

import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)

end_file = int(args['<end_file>'])+1
start_file = end_file - 100

f_name_base = 'slices/slices_s'
for j in range(start_file, end_file):
    f_name = f_name_base+str(j)+'.h5'
    with h5py.File(f_name,'r') as f:
        sim_time = f['scales']['sim_time'][:]
        Ma = f['tasks']['Ma'][:]
        for i, t in enumerate(sim_time):
            print(t, np.max(Ma[i,:,:]))
