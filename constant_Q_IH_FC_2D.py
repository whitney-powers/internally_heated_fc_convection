"""
Dedalus script for convection in a fully-compressible polytrope.
The convection is driven by an internal heating and internal cooling layer.

There are 5 control parameters:
    Ra      - The flux rayleigh number of the convection.
    epsilon     - Epsilon (ε) specifies superadiabicity
    Pr      - The Prandtl number = (viscous diffusivity / thermal diffusivity)
    nrho    - The height of the box
    aspect  - The aspect ratio (Lx = aspect * Lz)

Usage:
    polytrope_IH_FC_2D.py [options] 
    polytrope_IH_FC_2D.py <config> [options] 

Options:
    --Ra=<Ra>                  Flux Ra of convection [default: 1e4]
    --epsilon=<epsilon>        Superadiabicity [default: 0.1]
    --Pr=<Prandtl>             Prandtl number = nu/kappa [default: 1]
    --nrho=<n>                 Depth of domain [default: 3]
    --aspect=<aspect>          Aspect ratio of domain [default: 4]

    --nz=<nz>                  Vertical resolution   [default: 64]
    --nx=<nx>                  Horizontal (x) resolution [default: 128]
    --RK222                    Use RK222 timestepper (default: RK443)
    --SBDF2                    Use SBDF2 timestepper (default: RK443)
    --safety=<s>               CFL safety factor [default: 0.75]

    --run_time_wall=<time>     Run time, in hours [default: 119.5]
    --run_time_ff=<time>       Run time, in freefall times [default: 1.6e3]

    --restart=<restart_file>   Restart from checkpoint
    --seed=<seed>              RNG seed for initial conditions [default: 42]

    --label=<label>            Optional additional case name label
    --root_dir=<dir>           Root directory for output [default: ./]
"""
import logging
import os
import sys
import time
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from mpi4py import MPI
from scipy.special import erf

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

logger = logging.getLogger(__name__)
args = docopt(__doc__)

#Read config file
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def global_noise(domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    return noise_field

def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def set_equations(problem):
#    kx_0  = "(nx == 0) and (ny == 0)"
#    kx_n0 = "(nx != 0) or  (ny != 0)"
    kx_0  = "(nx == 0)"
    kx_n0 = "(nx != 0)"
    equations = ( (True, "True", "T1_z - dz(T1) = 0"),
                  (True, "True", "u_z - dz(u)   = 0"),
                  (True, "True", "w_z - dz(w)   = 0"),
                  (True, "True", "T0*(dt(ln_rho1) + Div_u + w*grad_ln_rho0) = -T0*UdotGrad(ln_rho1, dz(ln_rho1))"), #Continuity
                  (True, "True", "dt(u) - visc_L_x  + R*( dx(T1) + T0*dx(ln_rho1)                  ) = -UdotGrad(u, u_z) - R*T1*dx(ln_rho1) + visc_R_x "), #momentum-x
                  (True, "True", "T0*(dt(w) - visc_L_z  + R*( T1_z  + T1*grad_ln_rho0 + T0*dz(ln_rho1) ) ) = T0*(-UdotGrad(w, w_z) - R*T1*dz(ln_rho1) + visc_R_z )"), #momentum-z
                  (True, "True", "T0*(dt(T1) + w*T0_z + (γ-1)*T0*Div_u - diff_L ) = T0*(-UdotGrad(T1, T1_z) - (γ-1)*T1*Div_u + visc_heat + diff_R)"), #energy eqn
                )
    for solve, cond, eqn in equations:
        if solve:
            logger.info('solving eqn {} under condition {}'.format(eqn, cond))
            problem.add_equation(eqn, condition=cond)

    boundaries = ( (True, " left(T1_z) = 0", "True"),
                   (True, "right(T1) = 0", "True"),
                   (True, " left(u) = 0", "True"),
                   (True, "right(u) = 0", "True"),
                   (True, " left(w) = 0", "True"),
                   (True, "right(w) = 0", "True"),
                 )
    for solve, bc, cond in boundaries:
        if solve: 
            logger.info('solving bc {} under condition {}'.format(bc, cond))
            problem.add_bc(bc, condition=cond)

    return problem

def set_subs(problem):
    # Set up useful algebra / output substitutions
    problem.substitutions['dy(A)'] = '0'
    problem.substitutions['v']     = '0'
    problem.substitutions['v_z']   = '0'
    problem.substitutions['Lap(A, A_z)']                   = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    problem.substitutions['UdotGrad(A, A_z)']              = '(u*dx(A) + v*dy(A) + w*A_z)'
    problem.substitutions['GradAdotGradB(A, B, A_z, B_z)'] = '(dx(A)*dx(B) + dy(A)*dy(B) + A_z*B_z)'
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
    problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

    problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'

    problem.substitutions['ωx'] = "dy(w) - v_z"
    problem.substitutions['ωy'] = "u_z - dx(w)"
    problem.substitutions['ωz'] = "dx(v) - dy(u)"
    problem.substitutions['enstrophy'] = '(ωx**2 + ωy**2 + ωz**2)'
    problem.substitutions['vel_rms2']  = 'u**2 + v**2 + w**2'
    problem.substitutions['vel_rms']   = 'sqrt(vel_rms2)'
    problem.substitutions['KE']        = 'rho0*vel_rms2/2'
    problem.substitutions['ν']  = 'μ/rho_full'
    problem.substitutions['χ']  = 'κ/(rho_full*Cp)'

    problem.substitutions['T']         = '(T0 + T1)'
    problem.substitutions['T_z']       = '(T0_z + T1_z)'
    problem.substitutions['s1']        = '(Cv*log(1+T1/T0) - ln_rho1)'
    problem.substitutions['s0']        = '(Cv*log(T0) - ln_rho0)'
    problem.substitutions['dz_lnT']    = '(T_z/T)'
    problem.substitutions['dz_lnP']    = '(dz_lnT + grad_ln_rho0 + dz(ln_rho1))'

    problem.substitutions['Re'] = '(vel_rms/ν)'
    problem.substitutions['Pe'] = '(vel_rms/χ)'
    problem.substitutions['Ma'] = '(vel_rms/sqrt(T))'

    problem.substitutions['Div_u'] = '(dx(u) + dy(v) + w_z)'
    problem.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
    problem.substitutions["σyy"] = "(2*dy(v) - 2/3*Div_u)"
    problem.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
    problem.substitutions["σxy"] = "(dx(v) + dy(u))"
    problem.substitutions["σxz"] = "(dx(w) +  u_z )"
    problem.substitutions["σyz"] = "(dy(w) +  v_z )"

    problem.substitutions['visc_div_stress_x'] = 'dx(σxx) + dy(σxy) + dz(σxz)'
    problem.substitutions['visc_div_stress_y'] = 'dx(σxy) + dy(σyy) + dz(σyz)'
    problem.substitutions['visc_div_stress_z'] = 'dx(σxz) + dy(σyz) + dz(σzz)'

    problem.substitutions['visc_L_x'] = '((μ/rho0)*visc_div_stress_x)'
    problem.substitutions['visc_L_z'] = '((μ/rho0)*visc_div_stress_z)'
    problem.substitutions['visc_R_x'] = '((μ/rho_full)*visc_div_stress_x - visc_L_x)'
    problem.substitutions['visc_R_z'] = '((μ/rho_full)*visc_div_stress_z - visc_L_z)'
    
    problem.substitutions['diff_L'] = '((κ/(rho0*Cv))*Lap(T1, T1_z))'
    problem.substitutions['diff_R'] = '((1/(rho_full*Cv))*(Q + κ*Lap(T, T_z)) - diff_L)'

    problem.substitutions['visc_heat'] = '((μ/(rho_full*Cv))*(dx(u)*σxx + dy(v)*σyy + w_z*σzz + σxy**2 + σxz**2 + σyz**2))'

    problem.substitutions['grad']      = '(dz_lnT/dz_lnP)'
    problem.substitutions['grad_rad']  = '(flux/(R*κ*g))'
    problem.substitutions['grad_ad']   = '((γ-1)/γ)'

    # Old IH boundary layer Nu
    # Nu_IH defn inspired by https://ui.adsabs.harvard.edu/abs/2021arXiv211110906K/abstract
    problem.substitutions['delta_T'] = '(right(T) - left(T))'
    #problem.substitutions['Nu_IH'] = '(delta_T_rad - delta_T_ad) / (delta_T - delta_T_ad)'

    
    problem.substitutions['phi']    = '(-g*z)'
    problem.substitutions['F_cond'] = '(-κ*T_z)'
    problem.substitutions['F_enth'] = '( rho_full * w * ( Cp * T ) )'
    problem.substitutions['F_KE']   = '( rho_full * w * ( vel_rms2 / 2 ) )'
    problem.substitutions['F_PE']   = '( rho_full * w * phi )'
    problem.substitutions['F_visc'] = '( - μ * ( u*σxz + v*σyz + w*σzz ) )'
    problem.substitutions['F_conv'] = '( F_enth + F_KE + F_PE + F_visc )'
    problem.substitutions['F_tot']  = '( F_cond + F_conv )'
    problem.substitutions['F_dif_top']  = 'right( (flux - F_tot)/flux)'

    # Anders & Brown 2017 Nusselt Number
    problem.substitutions['F_A']     = '(-κ*(-g/Cp))'
    problem.substitutions['Nu_IH']  = '( 1 + vol_avg(F_conv)/vol_avg(F_cond - F_A))'
    
    return problem

def initialize_output(solver, data_dir, mode='overwrite', output_dt=2, iter=np.inf):
    Lx = solver.problem.parameters['Lx']
    analysis_tasks = OrderedDict()
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=40, mode=mode, iter=iter)
    slices.add_task('w')
    slices.add_task('s1')
    slices.add_task('T1')
    slices.add_task('enstrophy')
    analysis_tasks['slices'] = slices

    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=40, mode=mode)
    profiles.add_task("plane_avg(s1)", name='s1')
    profiles.add_task("plane_avg(sqrt((s1 - plane_avg(s1))**2))", name='s1_fluc')
    profiles.add_task("plane_avg(dz(s1))", name='s1_z')
    profiles.add_task("plane_avg(T_z - T_ad_z)", name='grad_T_superad')
    profiles.add_task("plane_avg(grad_ln_rho0 + dz(ln_rho1))", name='dz_lnrho')
    profiles.add_task("plane_avg(T1_z)", name='T1_z')
    profiles.add_task("plane_avg(u)", name='u')
    profiles.add_task("plane_avg(w)", name='w')
    profiles.add_task("plane_avg(vel_rms)", name='vel_rms')
    profiles.add_task("plane_avg(vel_rms2)", name='vel_rms2')
    profiles.add_task("plane_avg(KE)", name='KE')
    profiles.add_task("plane_avg(sqrt((v*ωz - w*ωy)**2 + (u*ωy - v*ωx)**2 + (w*ωx - u*ωz)**2))", name='advection')
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(grad)", name="grad")
    profiles.add_task("plane_avg(grad_ad*ones)", name="grad_ad")
    profiles.add_task("plane_avg(grad_rad)", name="grad_rad")
    profiles.add_task("plane_avg(1/grad-1)", name="m")
    profiles.add_task("plane_avg((1/grad_ad-1)*ones)", name="m_ad")
    profiles.add_task("plane_avg(1/grad_rad-1)", name="m_rad")
    profiles.add_task("plane_avg(F_cond)", name="F_cond")
    profiles.add_task("plane_avg(F_enth)", name="F_enth")
    profiles.add_task("plane_avg(F_KE)", name="F_KE")
    profiles.add_task("plane_avg(F_PE)", name="F_PE")
    profiles.add_task("plane_avg(F_visc)", name="F_visc")
    profiles.add_task("plane_avg(F_conv)", name="F_conv")
    profiles.add_task("plane_avg(F_tot)", name="F_tot")
    profiles.add_task("plane_avg(flux)", name="flux")
    analysis_tasks['profiles'] = profiles

    scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=output_dt*5, max_writes=np.inf, mode=mode)
    scalars.add_task("vol_avg(Re)", name="Re")
    scalars.add_task("vol_avg(Pe)", name="Pe")
    scalars.add_task("vol_avg(KE)", name="KE")
    scalars.add_task("vol_avg(Ma)", name="Ma")
    scalars.add_task("vol_avg(Nu_IH)", name="Nu")
    analysis_tasks['scalars'] = scalars

    checkpoint_min = 60
    checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoint', wall_dt=checkpoint_min*60, sim_dt=np.inf, iter=np.inf, max_writes=1, mode=mode)
    checkpoint.add_system(solver.state, layout = 'c')
    analysis_tasks['checkpoint'] = checkpoint

    return analysis_tasks

def run_cartesian_convection(args):
    #############################################################################################
    ### 1. Read in command-line args, set up data directory
    data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]
    data_dir += "_Ra{}_eps{}_nrho{}_Pr{}_a{}_{}x{}".format(args['--Ra'], args['--epsilon'], args['--nrho'], args['--Pr'], args['--aspect'], args['--nx'], args['--nz'])
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}'.format(data_dir)):
            os.makedirs('{:s}'.format(data_dir))
    logger.info("saving run in: {}".format(data_dir))

    ########################################################################################
    ### 2. Organize simulation parameters
    aspect   = float(args['--aspect'])
    nx = int(args['--nx'])
    nz = int(args['--nz'])
    Ra = float(args['--Ra'])
    Pr = float(args['--Pr'])
    epsilon = float(args['--epsilon'])
    nrho = float(args['--nrho'])

    # Thermo
    gamma = 5/3
    R = 1
    Cv = R/(gamma-1)
    Cp = gamma*Cv
    m_ad = 1/(gamma-1)
    g = Cp 
    T_ad_z = -g/Cp

    #Length scales
    Lz    = np.exp(nrho/np.abs(m_ad))-1
    Lx    = aspect * Lz
    Ly    = Lx
    delta = 0.05*Lz
    delta_h = 0.2*Lz

    #Radiative gradient
    rad_m = m_ad - epsilon
    grad_rad = 1/(rad_m + 1)
    T_rad_z = - grad_rad * (g / R)

    #Heating and diffusivities
    κμ = g * Lz**4 * np.abs(T_rad_z - T_ad_z) / Ra #Pr = mu * cp / kappa, so  mu kappa = Pr * kappa**2 / cp
    κ = np.sqrt(κμ * Cp / Pr)
    μ = Pr * κ / Cp
    Q_mag = κ * -(T_rad_z - T_ad_z) / delta_h
    Ma = Q_mag**(1/3)
    t_heat = 1/Ma

    #Adjust to account for expected velocities. and larger m = 0 diffusivities.
    logger.info("Running polytrope with the following parameters:")
    logger.info("   m = {:.3f}, Ra = {:.3e}, Pr = {:.2g}, resolution = {}x{}, aspect = {}".format(rad_m, Ra, Pr, nx, nz, aspect))
    logger.info("   heating timescale: {:8.3e}, kappa = {:.3e}, mu = {:.3e}".format(t_heat, κ, μ))
    
    ###########################################################################################################3
    ### 3. Setup Dedalus domain, problem, and substitutions/parameters
    x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)
    bases = [x_basis, z_basis]
    domain = de.Domain(bases, grid_dtype=np.float64, mesh=None)
    reducer = flow_tools.GlobalArrayReducer(domain.distributor.comm_cart)
    z = domain.grid(-1)
    z_de = domain.grid(-1, scales=domain.dealias)

    #Establish variables and setup problem
    variables = ['ln_rho1', 'T1', 'T1_z', 'u',  'w', 'u_z', 'w_z']
    problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

    # Set up background / initial state vs z.
    grad_ln_rho0 = domain.new_field()
    rho0         = domain.new_field()
    ln_rho0      = domain.new_field()
    s0_z         = domain.new_field()
    T_ad_f   = domain.new_field()
    T_ad_z_f = domain.new_field()
    T_rad_f   = domain.new_field()
    T_rad_z_f = domain.new_field()
    T0   = domain.new_field()
    T0_z = domain.new_field()
    T0_zz = domain.new_field()
    Q = domain.new_field()
    flux = domain.new_field()
    for f in [ln_rho0, grad_ln_rho0, rho0, s0_z, T0, T0_z, T0_zz, Q, T_ad_f, T_ad_z_f, T_rad_f, T_rad_z_f]:
        f.set_scales(domain.dealias)
    for f in [ln_rho0, grad_ln_rho0, T0, T0_z, rho0]:
        f.meta['x']['constant'] = True

    #radiative and adiabatic temp profiles
    T_ad_z_f['g'] = T_ad_z
    T_rad_z_f['g'] = T_rad_z
    T_ad_z_f.antidifferentiate('z', ('right', 1), out=T_ad_f)
    T_rad_z_f.antidifferentiate('z', ('right', 1), out=T_rad_f)
    delta_T_rad = (T_rad_f.interpolate(z=Lz) - T_rad_f.interpolate(z=0)).evaluate()['g'].min()
    delta_T_ad  = (T_ad_f.interpolate(z=Lz) - T_ad_f.interpolate(z=0)).evaluate()['g'].min()

    #Adiabatic polytropic stratification
    T0_zz['g'] = 0        
    T0_z['g'] = T_ad_z
    T0['g'] = (1 - T_ad_z*(Lz - z_de))

    rho0['g']         = T0['g']**m_ad
    ln_rho0['g']      = np.log(rho0['g'])
    ln_rho0.differentiate('z', out=grad_ln_rho0)

    s0_z['g'] = 0

    Q['g'] = κ/Lz * epsilon/(1+m_ad-epsilon)
    Q.antidifferentiate('z', ('left', -κ*T_ad_z), out=flux)

    #Plug in default parameters
    ones = domain.new_field()
    ones['g'] = 1
    problem.parameters['ones']   = ones
    problem.parameters['g']      = g
    problem.parameters['R']      = R
    problem.parameters['γ']      = gamma
    problem.parameters['κ']      = κ
    problem.parameters['μ']      = μ
    problem.parameters['Lx']     = Lx
    problem.parameters['Lz']     = Lz
    problem.parameters['T0']     = T0
    problem.parameters['T0_z']     = T0_z
    problem.parameters['T0_zz']    = T0_zz
    problem.parameters['Q'] = Q
    problem.parameters['grad_ln_rho0'] = grad_ln_rho0
    problem.parameters['ln_rho0'] = ln_rho0
    problem.parameters['rho0'] = rho0
    problem.parameters['s0_z'] = s0_z
    problem.parameters['Cp'] = Cp
    problem.parameters['Cv'] = Cv
    problem.parameters['T_ad_z'] = T_ad_z
    problem.parameters['flux'] = flux
    problem.parameters['delta_T_ad'] = delta_T_ad
    problem.parameters['delta_T_rad'] = delta_T_rad

    problem = set_subs(problem)
    problem = set_equations(problem)

    if args['--RK222']:
        logger.info('using timestepper RK222')
        ts = de.timesteppers.RK222
    elif args['--SBDF2']:
        logger.info('using timestepper SBDF2')
        ts = de.timesteppers.SBDF2
    else:
        logger.info('using timestepper RK443')
        ts = de.timesteppers.RK443
    solver = problem.build_solver(ts)
    logger.info('Solver built')

    ###########################################################################
    ### 4. Set initial conditions or read from checkpoint.
    mode = 'overwrite'
    if args['--restart'] is None:
        T1 = solver.state['T1']
        T1_z = solver.state['T1_z']
        z_de = domain.grid(-1, scales=domain.dealias)
        for f in [T1]:
            f.set_scales(domain.dealias, keep_data=True)

        noise = global_noise(domain, int(args['--seed']))
        T1['g'] = 1e-3*Ma*np.sin(np.pi*(z_de))*noise['g']
        T1.differentiate('z', out=T1_z)
        dt = None
    else:
#        write, dt = solver.load_state(args['--restart'], -1) 
        mode = 'append'
        raise NotImplementedError('need to implement checkpointing')

    ###########################################################################
    ### 5. Set simulation stop parameters, output, and CFL
    t_therm = Lz**2/κ
    max_dt = 0.1*t_heat
    if dt is None:
        dt = max_dt

    cfl_safety = float(args['--safety'])
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.25, max_dt=max_dt, threshold=0.2)
    CFL.add_velocities(('u', 'w'))

    run_time_ff   = float(args['--run_time_ff'])
    run_time_wall = float(args['--run_time_wall'])
    solver.stop_sim_time  = run_time_ff*t_heat
    solver.stop_wall_time = run_time_wall*3600.
 
    ###########################################################################
    ### 6. Setup output tasks; run main loop.
    analysis_tasks = initialize_output(solver, data_dir, mode=mode, output_dt=0.1*t_heat)

    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re", name='Re')
    flow.add_property("Pe", name='Pe')
    flow.add_property("Ma", name='Ma')
    flow.add_property("Nu_IH", name='Nu')
    flow.add_property("F_dif_top", name='F_dif_top')
    Hermitian_cadence = 100

    def main_loop(dt):
        Re_avg = 0
        try:
            logger.info('Starting loop')
            start_iter = solver.iteration
            start_time = time.time()
            while solver.ok and np.isfinite(Re_avg):
                effective_iter = solver.iteration - start_iter
                solver.step(dt)

                if effective_iter % Hermitian_cadence == 0:
                    for f in solver.state.fields:
                        f.require_grid_space()

                if effective_iter % 1 == 0:
                    Re_avg = flow.grid_average('Re')

                    log_string =  'Iteration: {:7d}, '.format(solver.iteration)
                    log_string += 'Time: {:8.3e} heat ({:8.3e} therm), dt: {:8.3e}, dt/t_h: {:8.3e}, '.format(solver.sim_time/t_heat, solver.sim_time/t_therm,  dt, dt/t_heat)
                    log_string += 'Pe: {:8.3e}/{:8.3e}, '.format(flow.volume_average('Pe'), flow.max('Pe'))
                    log_string += 'Ma: {:8.3e}/{:8.3e}, '.format(flow.volume_average('Ma'), flow.max('Ma'))
                    log_string += 'Nu: {:8.3e}, '.format(flow.volume_average('Nu'))
                    log_string += 'F_dif_top: {:8.3e}'.format(flow.grid_average('F_dif_top'))
                    logger.info(log_string)

                dt = CFL.compute_dt()
                    
        except:
            raise
            logger.error('Exception raised, triggering end of main loop.')
        finally:
            end_time = time.time()
            main_loop_time = end_time-start_time
            n_iter_loop = solver.iteration-start_iter
            logger.info('Iterations: {:d}'.format(n_iter_loop))
            logger.info('Sim end time: {:f}'.format(solver.sim_time))
            logger.info('Run time: {:f} sec'.format(main_loop_time))
            logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
            logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
            try:
                final_checkpoint = solver.evaluator.add_file_handler(data_dir+'final_checkpoint', wall_dt=np.inf, sim_dt=np.inf, iter=1, max_writes=1)
                final_checkpoint.add_system(solver.state, layout = 'c')
                solver.step(1e-5*dt) #clean this up in the future...works for now.
                post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
            except:
                raise
                print('cannot save final checkpoint')
            finally:
                logger.info('beginning join operation')
                for key, task in analysis_tasks.items():
                    logger.info(task.base_path)
                    post.merge_analysis(task.base_path)
            domain.dist.comm_cart.Barrier()
        return Re_avg

    Re_avg = main_loop(dt)
    if np.isnan(Re_avg):
        return False, data_dir
    else:
        return True, data_dir

if __name__ == "__main__":
    ended_well, data_dir = run_cartesian_convection(args)
    if MPI.COMM_WORLD.rank == 0:
        print('ended with finite Re? : ', ended_well)
        print('data is in ', data_dir)
