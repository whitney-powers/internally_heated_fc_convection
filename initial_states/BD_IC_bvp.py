"""
NLBVP to produce initial conditions for BD polytrope script
This attempts to produce mass conserving inintial conditions in hydrostatic equilibrium which match our expectation of the shape of the final state of grad_S

"""
# TO DO
#   - create output tasks


import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import h5py
from dedalus import public as de

import logging
logger = logging.getLogger(__name__)


def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


# Parameters
Nz = 128
ncc_cutoff = 1e-6
tolerance = 1e-12
m_rad = 1
nrho = 1
Lz = np.exp(nrho/m_rad)-1

# Build domain
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([z_basis], np.float64)
z = domain.grid(0)

gamma = 5/3
R = 1
Cv = R/(gamma-1)
Cp = gamma*Cv
R = 1
g = Cp

grad_rad = 1/(m_rad + 1)
grad_T_rad = - grad_rad * (g / R)
grad_T_ad = -g/Cp

delta_grad_T = grad_T_rad - grad_T_ad

m0 = (1-(1-grad_T_rad*Lz)**(m_rad+1)) / ((m_rad+1)*grad_T_rad)

T = domain.new_field()
Tz = domain.new_field()
Tz['g'] = grad_T_ad
Tz['g'] += delta_grad_T * zero_to_one(z, 0.9*Lz, width = 0.05*Lz)
Tz['g'] += delta_grad_T * one_to_zero(z, 0.1*Lz, width = 0.05*Lz) 
Tz.antidifferentiate('z', ('right', 1), out=T)


# Setup problem
problem = de.NLBVP(domain, variables=['m', 'ln_rho'], ncc_cutoff=ncc_cutoff)
problem.parameters['g'] = g
problem.parameters['R'] = R
problem.parameters['m0'] = m0
problem.parameters['Tz'] = Tz
problem.parameters['T'] = T

problem.add_equation("T*dz(ln_rho)  = -Tz - g/R")
problem.add_equation("dz(m) = exp(ln_rho)")
problem.add_bc("left(m) = 0")
problem.add_bc("right(m) = m0")

# Setup initial guess
solver = problem.build_solver()

ln_rho = solver.state['ln_rho']
m = solver.state['m']
T0 = domain.new_field()
rho = domain.new_field()
ln_rho0 = domain.new_field()
T0['g'] = (1 - grad_T_rad*(Lz - z))

ln_rho['g'] = np.log(T0['g']**m_rad)
ln_rho0['g'] = ln_rho['g']
rho['g'] = np.exp(ln_rho['g'])


rho.antidifferentiate('z', ('left', 0), out=m)
# Need m guess

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
start_time = time.time()
while np.sum(np.abs(pert)) > tolerance:
    solver.newton_iteration()
    logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
end_time = time.time()


logger.info('-'*20)
logger.info('Iterations: {}'.format(solver.iteration))
logger.info('Run time: %.2f sec' %(end_time-start_time))

T1 = (T-T0).evaluate()
ln_rho1 = (ln_rho-ln_rho0).evaluate()
T1.set_scales(1, keep_data=True)
ln_rho1.set_scales(1, keep_data=True)
T0.set_scales(1, keep_data=True)
ln_rho0.set_scales(1, keep_data=True)


with h5py.File('atmosphere.h5', 'w') as f:
    f['z'] = z
    f['ln_rho1'] = ln_rho1['g']
    f['T1'] = T1['g']
    f['T0'] = T0['g']
    f['ln_rho0'] = ln_rho0['g']
