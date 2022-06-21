"""
Script for generating spectra from simulation output files
Usage:
    make_spectra.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: spectra]
    --out_name=<out_name>               Name of figure output directory & base name of saved figures [default: spectra_actual]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
"""

import numpy as np
import h5py
from docopt import docopt
from glob import glob
args = docopt(__doc__)

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional arguments
start_file  = int(args['--start_file'])
out_name    = args['--out_name']
n_files     = args['--n_files']
if n_files is not None:
    n_files = int(n_files)

def read_slice(file,task):
    with h5py.File(file,'r') as f:
        data = f['tasks'][task][:,:,0]
        basis = f['tasks'][task].dims[1][0][:]
    return(data, basis)
def calculate_spectra(task):
    F = np.fft.rfft(task, norm='backward')
    n = len(task)
    power = F * np.conj(F)/n**2
    return(np.real(power))

in_dir = root_dir+'/'+data_dir
in_files = glob(in_dir+'/'+data_dir+'*.h5')
out_dir = root_dir+'/'+out_name+'/'
import os
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for file in in_files:
    file_tail = file.split('_')[-1]
    out_file = out_dir+out_name+'_'+file_tail
    u, u_basis = read_slice(file,'u')
    w, w_basis = read_slice(file,'w')
    iters = (np.shape(u)[0])
    n = np.shape(u)[1]
    L = n*(u_basis[1]-u_basis[0])
    k = np.fft.rfftfreq(u_basis.shape[-1], L/n)
    u_spectra=[]
    w_spectra=[]
    total_spectra=[]
    for i in range(iters):
        u_power = calculate_spectra(u[i,:])
        w_power = calculate_spectra(w[i,:])
        total_power = u_power + w_power
        u_spectra.append(u_power)
        w_spectra.append(w_power)
        total_spectra.append(total_power)
    with h5py.File(out_file, 'a') as f:
        f.create_dataset('u', data=np.array(u_spectra))
        f.create_dataset('w', data=np.array(w_spectra))
        f.create_dataset('u_vec', data=np.array(total_spectra))
        f.create_dataset('kx', data=k)
