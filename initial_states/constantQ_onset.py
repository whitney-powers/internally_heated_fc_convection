"""
NLBVP to produce initial conditions for BD polytrope script
This attempts to produce mass conserving inintial conditions in hydrostatic equilibrium which match our expectation of the shape of the final state of grad_S

"""
# TO DO
#   - Add Eignentools onset solver
#   - Add docopt header with parameters (γ, nρ, ε)

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import h5py
from dedalus import public as de
from eigentools import Eigenproblem, CriticalFinder

from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)


comm = MPI.COMM_WORLD


def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


# Parameters
Nz = 64
ncc_cutoff = 1e-6
tolerance = 1e-12
epsilon = 2.3
nrho = 3

file_name = sys.argv[0].strip('.py')+'_eps'+str(epsilon)

gamma = 5/3
R = 1
Cv = R/(gamma-1)
Cp = gamma*Cv
R = 1
g = Cp

Pr = 1

grad_T_ad = -g/Cp
m_ad = 1/(gamma-1)

Lz = np.exp(nrho/m_ad)-1

# Build domain
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=1)
domain = de.Domain([z_basis], np.complex128, comm=MPI.COMM_SELF)
z = domain.grid(0)




Q = 1/Lz * epsilon/(1+m_ad-epsilon)

T0 = domain.new_field()
T0_z = domain.new_field()
rho0 = domain.new_field()

T0_z['g'] = grad_T_ad
T0['g'] = (1 - grad_T_ad*(Lz - z))

rho0['g'] =  T0['g']**m_ad
m0 = rho0.integrate('z')['g'][0]

# Setup problem
problem = de.NLBVP(domain, variables=['T','Tz', 'rho'], ncc_cutoff=ncc_cutoff)
problem.parameters['Q'] = Q
problem.parameters['g'] = g
problem.parameters['R'] = R
problem.parameters['m0'] = m0
problem.parameters['T0'] = T0
problem.parameters['T0_z'] = T0_z



problem.add_equation("-dz(Tz) = Q")
problem.add_equation("dz(rho)  = -Tz*rho/T - g*rho/R/T")
problem.add_equation("dz(T) - Tz = 0")
problem.add_equation("integ(rho) = m0")
problem.add_bc("right(T) = right(T0)")
problem.add_bc("left(Tz) = left(T0_z)")

# Setup initial guess
solver = problem.build_solver()

rho = solver.state['rho']
T = solver.state['T']
Tz = solver.state['Tz']

rho['g'] = rho0['g']
T['g'] = T0['g']
Tz['g'] = T0_z['g']

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


#plt.plot(z,T['g'])
#plt.show()
ln_rho = domain.new_field()
ln_rho['g'] = np.log(rho['g'])
ln_rho_z = domain.new_field()
ln_rho.differentiate('z', out=ln_rho_z)

onset =de.EVP(domain, ['T1', 'T1_z', 'u', 'u_z', 'w', 'w_z', 'ln_rho1', ], eigenvalue='omega') # Check if domain is ok
onset.parameters['k'] = 0.5 # horizonal wavenumber
onset.parameters['Ra'] = 625 # Rayleigh number
onset.parameters['T0_z'] =  Tz
onset.parameters['T0'] = T
onset.parameters['ln_rho0'] = ln_rho 
onset.parameters['grad_ln_rho0'] = ln_rho_z 
onset.parameters['R'] = R # OK
onset.parameters['γ'] = gamma # OK
onset.parameters['epsilon'] = epsilon # DEFINE THIS
onset.parameters['Cp'] = Cp # OK
onset.parameters['Cv'] = Cv # OK
onset.parameters['g'] = g #OK
onset.parameters['Pr'] = Pr # DEFINE THIS
onset.parameters['Lz'] = Lz # OK

#eigenproblem subs
onset.substitutions['dt(A)'] = 'omega*A'
onset.substitutions['dx(A)'] = '1j*k*A'
# 2D restriction
onset.substitutions['dy(A)'] = '0'
onset.substitutions['v']     = '0'
onset.substitutions['v_z']   = '0'

onset.substitutions['Lap(A, A_z)']  = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
onset.substitutions['UdotGrad(A, A_z)'] = '(u*dx(A) + v*dy(A) + w*A_z)'
onset.substitutions['Div_u'] = '(dx(u) + dy(v) + w_z)'

    #κμ = g * Lz**4 * np.abs(T_rad_z - T_ad_z) / Ra #Pr = mu * cp / kappa, so  mu kappa = Pr * kappa**2 / cp                                                                                                
    #κ = np.sqrt(κμ * Cp / Pr)
    #μ = Pr * κ / Cp


#onset.substitutions['epsilon'] = '' # function of eps_new
onset.substitutions['rho0'] = 'exp(ln_rho0)'
onset.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'
onset.substitutions['T']         = '(T0 + T1)'
onset.substitutions['T_z']       = '(T0_z + T1_z)'


onset.substitutions['m_ad'] = '1/(γ-1)'
onset.substitutions['rad_m'] = 'm_ad - epsilon'
onset.substitutions['grad_rad'] = '1/(rad_m + 1)'
onset.substitutions['T_rad_z'] = '- grad_rad * (g / R)'
onset.substitutions['T_ad_z'] = '-g/Cp'
onset.substitutions['κμ'] = 'g * Lz**4 * (-1)*(T_rad_z - T_ad_z) / Ra'
onset.substitutions['κ'] = 'sqrt(κμ * Cp / Pr)'
onset.substitutions['μ'] = 'Pr * κ / Cp'
onset.substitutions['Q'] = 'κ/Lz * epsilon/(1+m_ad-epsilon)'

onset.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
onset.substitutions["σyy"] = "(2*dy(v) - 2/3*Div_u)"
onset.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
onset.substitutions["σxy"] = "(dx(v) + dy(u))"
onset.substitutions["σxz"] = "(dx(w) +  u_z )"
onset.substitutions["σyz"] = "(dy(w) +  v_z )"

onset.substitutions['visc_div_stress_x'] = 'dx(σxx) + dy(σxy) + dz(σxz)'
onset.substitutions['visc_div_stress_y'] = 'dx(σxy) + dy(σyy) + dz(σyz)'
onset.substitutions['visc_div_stress_z'] = 'dx(σxz) + dy(σyz) + dz(σzz)'

onset.substitutions['visc_L_x'] = '((μ/rho0)*visc_div_stress_x)'
onset.substitutions['visc_L_z'] = '((μ/rho0)*visc_div_stress_z)'
onset.substitutions['visc_R_x'] = '((μ/rho_full)*visc_div_stress_x - visc_L_x)'
onset.substitutions['visc_R_z'] = '((μ/rho_full)*visc_div_stress_z - visc_L_z)'

onset.substitutions['diff_L'] = '((κ/(rho0*Cv))*Lap(T1, T1_z))'
onset.substitutions['diff_R'] = '((1/(rho_full*Cv))*(Q + κ*Lap(T, T_z)) - diff_L)'

onset.substitutions['visc_heat'] = '((μ/(rho_full*Cv))*(dx(u)*σxx + dy(v)*σyy + w_z*σzz + σxy**2 + σxz**2 + σyz**2))'



onset.add_equation("T1_z - dz(T1) = 0")
onset.add_equation("u_z - dz(u)   = 0")
onset.add_equation("w_z - dz(w)   = 0")
onset.add_equation("T0*(dt(ln_rho1) + Div_u + w*grad_ln_rho0) = 0")#-T0*UdotGrad(ln_rho1, dz(ln_rho1))")
onset.add_equation("dt(u) - visc_L_x  + R*( dx(T1) + T0*dx(ln_rho1)                  ) = 0 ") #-UdotGrad(u, u_z) - R*T1*dx(ln_rho1) + visc_R_x ")
onset.add_equation("T0*(dt(w) - visc_L_z  + R*( T1_z  + T1*grad_ln_rho0 + T0*dz(ln_rho1) ) ) = 0")#T0*(-UdotGrad(w, w_z) - R*T1*dz(ln_rho1) + visc_R_z )")
onset.add_equation("T0*(dt(T1) + w*T0_z + (γ-1)*T0*Div_u - diff_L ) = 0")#T0*(-UdotGrad(T1, T1_z) - (γ-1)*T1*Div_u + visc_heat + diff_R)")

# BCs
    # boundaries = ( (True, " left(T1_z) = 0", "True"),
    #                (True, "right(T1) = 0", "True"),
    #                (True, " left(u) = 0", "True"),
    #                (True, "right(u) = 0", "True"),
    #                (True, " left(w) = 0", "True"),
    #                (True, "right(w) = 0", "True"),
onset.add_bc(" left(T1_z) = 0")
onset.add_bc("right(T1) = 0")
onset.add_bc(" left(u) = 0")
onset.add_bc("right(u) = 0")
onset.add_bc(" left(w) = 0")
onset.add_bc("right(w) = 0")


# Eigneproblem
EP = Eigenproblem(onset)

cf = CriticalFinder(EP, ("k","Ra"), comm, find_freq = True)

start = time.time()
# Root Finder

nk = 20
nRa = 20
kpoints = np.linspace(0.2, 0.8, nk)
Rapoints = np.linspace(300, 1000, nRa)

#try:
#    cf.load_grid('{}.h5'.format(file_name))
#except:
cf.grid_generator((kpoints, Rapoints), sparse=True)
cf.save_grid(file_name)

end = time.time()
if comm.rank == 0:
    logger.info("grid generation time: {:10.5f} sec".format(end-start))

logger.info("Beginning critical finding with root polishing...")
begin = time.time()
crit = cf.crit_finder(polish_roots=True, tol=1e-5)
end = time.time()
logger.info("critical finding/root polishing time: {:10.5f} sec".format(end-start))

if comm.rank == 0:
    print("crit = {}".format(crit))
    print("critical wavenumber k = {:10.5f}".format(crit[0]))
    print("critical Ra = {:10.5f}".format(crit[1]))
    print("critical freq = {:10.5f}".format(crit[2]))

    pax, cax = cf.plot_crit(xlabel=r'$k_x$', ylabel=r'$\mathrm{Ra}$')
    pax.figure.savefig("constQ_2d_growth_rates_eps"+str(epsilon)+".png",dpi=300)
    
#VARS T1, T1_z, u, u_z, w, w_z, ln_rho1
#PARS T0, grad_ln_rho0, R, T0_z, γ,                                                              Q, Cv, rho0
#SUBS Div_u, UdotGrad, visc_L_x, visc_R_x, visc_L_z, visc_R_z, diff_L, diff_R, visc_heat,        μ, κ, rho_full

# with h5py.File('atmosphere.h5', 'w') as f:
#     f['z'] = z
#     f['ln_rho1'] = ln_rho1['g']
#     f['T1'] = T1['g']
#     f['T0'] = T0['g']
#     f['ln_rho0'] = ln_rho0['g']
