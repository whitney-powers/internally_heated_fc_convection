"""
NLBVP to produce initial conditions for BD polytrope script
This attempts to produce mass conserving inintial conditions in hydrostatic equilibrium which match our expectation of the shape of the final state of grad_S

"""
# TO DO
#   - Add grad_S as ncc with appropriate shape
#   - Create initial guess
#   - create output tasks


import time
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)


# Parameters
Nz = 128
ncc_cutoff = 1e-6
tolerance = 1e-12

# Build domain
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([z_basis], np.float64)

gamma = 5/3
R = 1
Cv = R/(gamma-1)
Cp = gamma*Cv
R = 1
g = Cp
grad_T0 = -g/Cp
m_rad = 1
Lz = np.exp(n_rho/m)-1

m0 = 1/(m+1) * (-grad_T0*Lz**2 + 2*Lz)

# Setup problem
problem = de.NLBVP(domain, variables=['m', 'ln_rho', 'T','Tz'], ncc_cutoff=ncc_cutoff)
problem.parameters['g'] = g
problem.parameters['Cv'] = Cv
problem.parameters['Cp'] = Cp
problem.parameters['R'] = R
problem.parameters['m0'] = m0
problem.parameters['grad_T0'] = grad_T0

# Add grad_S as ncc somewhere
problem.add_equation("Tz = -T*dz(ln_rho) - g/R")
problem.add_equation("Cv/Cp * Tz - T * grad_S/Cp = R*T/Cp*dz(ln_rho)")
problem.add_equation("dz(m) = exp(ln_rho)")
problem.add_equation("Tz - dx(T) = 0")
problem.add_bc("left(m = 0")
problem.add_bc("right(m) = m0")
problem.add_bc("left(Tz) = grad_T0")
problem.add_bc("right(T) = 1")

# Setup initial guess
solver = problem.build_solver()
z = domain.grid(0)
f = solver.state['f']
fx = solver.state['fx']
R = solver.state['R']
f['g'] = np.cos(np.pi/2 * x)*0.9
f.differentiate('x', out=fx)
R['g'] = 3

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


