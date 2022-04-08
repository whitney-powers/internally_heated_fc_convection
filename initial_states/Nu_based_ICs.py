import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import h5py
from dedalus import public as de


from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)


comm = MPI.COMM_WORLD


def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

ncc_cutoff = 1e-6
tolerance = 1e-12


Ra_crit = 102

Nu = lambda Ra: 0.8 * (Ra/Ra_crit)**0.2

nrho=3
gamma = 5/3
R = 1
Cv = R/(gamma-1)
Cp = gamma*Cv
R = 1
g = Cp
m_ad = 1/(gamma-1)
Lz = np.exp(nrho/m_ad)-1

# Control Parameters
Pr = 1
eps_old = 0.1
Ra = 1e5

#Need

# total flux leaving the top
rad_m = m_ad - eps_old
grad_rad = 1/(rad_m + 1)
T_rad_z = - grad_rad * (g / R)

T_ad_z = -g/Cp # -g/Cp

κμ = g * Lz**4 * (-1)*(T_rad_z - T_ad_z) / Ra
κ = np.sqrt(κμ * Cp / Pr)
Q = κ/Lz * eps_old/(1+m_ad-eps_old)  # or κ/Lz * eps_new
flux_top = Q * Lz # perturbation only, the full flux is Q*Lz + flux_ad

T_z = -flux_top/κ # perturbation only

Nz = 128
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=1)
domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)
z = domain.grid(0)

T1_z = domain.new_field()
T1 = domain.new_field()

delta = Lz/(2 * Nu(Ra))
T1_z['g'] = T_z * zero_to_one(z, Lz-delta, width=delta/2)
T1_z.antidifferentiate('z', ('right', 0), out=T1)

plt.plot(z,T1_z['g'])
plt.axhline(T_z)
plt.show()

# Equations
# For a given temperature perturbation, solve for the density field which satisfies hydrostatic equilibrium and conserves mass

T0 = domain.new_field()
T0_z = domain.new_field()
rho0 = domain.new_field()

T0_z['g'] = T_ad_z
T0['g'] = (1 - T_ad_z*(Lz - z))


rho0['g'] =  T0['g']**m_ad
m0 = rho0.integrate('z')['g'][0]

# Setup problem                                                                                                                                                                                            
problem = de.NLBVP(domain, variables=['m', 'ln_rho'], ncc_cutoff=ncc_cutoff)
problem.parameters['Q'] = Q
problem.parameters['g'] = g
problem.parameters['R'] = R
problem.parameters['m0'] = m0
problem.parameters['T'] = T0+T1
problem.parameters['Tz'] = T0_z + T1_z



#problem.add_equation("-dz(Tz) = Q")
problem.add_equation("dz(ln_rho)  = -Tz/T - g/R/T")
problem.add_equation("dz(m) = exp(ln_rho)")
#problem.add_equation("dz(T) - Tz = 0")
problem.add_bc("left(m) = 0")
problem.add_bc("right(m) = m0")
#problem.add_bc("right(T) = right(T0)")
#problem.add_bc("left(Tz) = left(T0_z)")

solver = problem.build_solver()

ln_rho = solver.state['ln_rho']
m = solver.state['m']

ln_rho['g'] = np.log(rho0['g'])
rho0.antidifferentiate('z', ('left', 0), out=m)


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


plt.plot(z,ln_rho['g']-np.log(rho0['g']))
plt.show()
