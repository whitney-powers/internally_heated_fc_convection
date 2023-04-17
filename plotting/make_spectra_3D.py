"""
65;6003;1cScript for generating spectra from simulation output files
Usage:
    make_spectra.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
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
    print('reading file:',file)
    with h5py.File(file,'r') as f:
        data = f['tasks'][task][:,:,:,0]
        xbasis = f['tasks'][task].dims[1][0][:]
        ybasis = f['tasks'][task].dims[2][0][:]
    return(data, xbasis, ybasis)
def calculate_spectra(task):
    F = np.fft.fft2(task, norm='backward')
    print(np.shape(F))
    nx = np.shape(task)[0]
    ny = np.shape(task)[1]
    power = F * np.conj(F)/(nx**2*ny**2)
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
    u, u_xbasis, u_ybasis = read_slice(file,'u_midz')
    v, v_xbasis, v_ybasis = read_slice(file,'v_midz')
    w, w_xbasis, w_ybasis = read_slice(file,'w_midz')
    iters = (np.shape(u)[0])
    nx = np.shape(u)[1]
    ny = np.shape(u)[2]
    Lx = nx*(u_xbasis[1]-u_xbasis[0])
    Ly = ny*(u_ybasis[1]-u_ybasis[0])
    
    kx = np.fft.fftfreq(u_xbasis.shape[-1], Lx/nx)
    ky = np.fft.fftfreq(u_ybasis.shape[-1], Ly/ny)
    u_spectra=[]
    v_spectra=[]
    w_spectra=[]
    total_spectra=[]
    for i in range(iters):
        u_power = calculate_spectra(u[i,:,:])
        v_power = calculate_spectra(v[i,:,:])
        w_power = calculate_spectra(w[i,:,:])
        total_power = u_power + v_power + w_power
        u_spectra.append(u_power)
        v_spectra.append(v_power)
        w_spectra.append(w_power)
        total_spectra.append(total_power)
    with h5py.File(out_file, 'a') as f:
        f.create_dataset('u', data=np.array(u_spectra))
        f.create_dataset('v', data=np.array(v_spectra))
        f.create_dataset('w', data=np.array(w_spectra))
        f.create_dataset('u_vec', data=np.array(total_spectra))
        f.create_dataset('kx', data=kx)
        f.create_dataset('ky', data=ky)
