import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('apj.mplstyle')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
#import palettable
import h5py
from collections import OrderedDict
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#from scipy import interpolate as interpolate
#from scipy.optimize import least_squares
import matplotlib.patheffects as PathEffects
from multiprocessing.pool import Pool

# manually input static colormaps

#Panel A

v_A = [-0.6562225044137092, 0.03059020996977796]
s_ad_A = 0.2395253038547343
divnorm_A = matplotlib.colors.TwoSlopeNorm(vmin=v_A[0], vcenter=0, vmax=v_A[1])

#Panel B

v_B = [-0.2960404600666942, 0.0059984835768767675]
s_ad_B = 0.09733582493492782
divnorm_B = matplotlib.colors.TwoSlopeNorm(vmin=v_B[0], vcenter=0, vmax=v_B[1])

#Panel C

v_C = [-0.12255634355224203, 0.0020491095545264668]
s_ad_C = 0.03878349375739232
divnorm_C = matplotlib.colors.TwoSlopeNorm(vmin=v_C[0], vcenter=0, vmax=v_C[1])

#Panel D

v_D = [-1.0617299766023411, 0.022692472305744504]
s_ad_D = 0.43979093132084884
divnorm_D = matplotlib.colors.TwoSlopeNorm(vmin=v_D[0], vcenter=0, vmax=v_D[1])

#Panel E

v_E = [-0.5890720703087532, 0.013896158602787673]
s_ad_E = 0.21396138097156694
divnorm_E = matplotlib.colors.TwoSlopeNorm(vmin=v_E[0], vcenter=0, vmax=v_E[1])

#Panel F

v_F = [-0.2551483691735061, 0.004855716207166419]
s_ad_F = 0.08483876021639364
divnorm_F = matplotlib.colors.TwoSlopeNorm(vmin=v_F[0], vcenter=0, vmax=v_F[1])

def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or):
    """
    Takes grid coordinates and data on grid and prepares it for 3D surface plotting in plotly
    
    Arguments:
    x_vals : NumPy array (1D) or float
        Gridspace x values of the data
    y_vals : NumPy array (1D) or float
        Gridspace y values of the data
    z_vals : NumPy array (1D) or float
        Gridspace z values of the data
    data_vals : NumPy array (2D)
        Gridspace values of the data
        
    Keyword Arguments:
    x_bounds : Tuple of floats of length 2
        If specified, the min and max x values to plot
    y_bounds : Tuple of floats of length 2
        If specified, the min and max y values to plot
    z_bounds : Tuple of floats of length 2
        If specified, the min and max z values to plot
        
    Returns a dictionary of keyword arguments for plotly's surface plot function
    """
    x_vals=np.array(x_vals)    
    y_vals=np.array(y_vals)    
    z_vals=np.array(z_vals)    
    if z_vals.size == 1: #np.ndarray and type(y_vals) == np.ndarray :
        yy, xx = np.meshgrid(y_vals, x_vals)
        zz = z_vals * np.ones_like(xx)
    elif y_vals.size  == 1: # np.ndarray and type(z_vals) == np.ndarray :
        zz, xx = np.meshgrid(z_vals, x_vals)
        yy = y_vals * np.ones_like(xx)
    elif x_vals.size == 1: #np.ndarray and type(z_vals) == np.ndarray :
        zz, yy = np.meshgrid(z_vals, y_vals)
        xx = x_vals * np.ones_like(yy)
    else:
        raise ValueError('x,y,or z values must have size 1')
    if x_bounds is None:
        if x_vals.size == 1 and bool_function == np.logical_or :
            x_bool = np.zeros_like(yy)
        else:
            x_bool = np.ones_like(yy)
    else:
        x_bool = (xx >= x_bounds[0])*(xx <= x_bounds[1])

    if y_bounds is None:
        if y_vals.size == 1 and bool_function == np.logical_or :
            y_bool = np.zeros_like(xx)
        else:
            y_bool = np.ones_like(xx)
    else:
        y_bool = (yy >= y_bounds[0])*(yy <= y_bounds[1])

    if z_bounds is None:
        if z_vals.size  == 1 and bool_function == np.logical_or :
            z_bool = np.zeros_like(xx)
        else:
            z_bool = np.ones_like(xx)
    else:
        z_bool = (zz >= z_bounds[0])*(zz <= z_bounds[1])


    side_bool = bool_function.reduce((x_bool, y_bool, z_bool))


    side_info = OrderedDict()
    side_info['x'] = np.where(side_bool, xx, np.nan)
    side_info['y'] = np.where(side_bool, yy, np.nan)
    side_info['z'] = np.where(side_bool, zz, np.nan)
    side_info['surfacecolor'] = np.where(side_bool, data_vals, np.nan)

    return side_info

def plot_3D_box(ax, s1_sidex, s1_sidey, s1_midx, s1_midy, s1_top, s1_bottom, s1_midz, x, y, z, divnorm, s1_ad_guess):
    
    nx = len(x)
    Lx = (x[1]-x[0])*nx
    Ly = Lx
    Lz = 1/4 * Lx

    x_mid = Lx/2
    y_mid = Ly/2
    z_mid = Lz/2
    #-.01#, axis=0)

    
    xy_side = construct_surface_dict(x, y, np.max(z), s1_top-s1_ad_guess, x_bounds=(0, x_mid), y_bounds=(0, y_mid))
    xz_side = construct_surface_dict(x, np.max(y), z, s1_sidey-s1_ad_guess, x_bounds=(0, x_mid), z_bounds=(0, z_mid))
    yz_side = construct_surface_dict(np.max(Lx), y, z, s1_sidex-s1_ad_guess, y_bounds=(0, y_mid), z_bounds=(0, z_mid))
               
    xy_mid = construct_surface_dict(x, y, z_mid, s1_midz-s1_ad_guess,x_bounds=(x_mid, np.max(x)), y_bounds=(y_mid, np.max(y)), bool_function=np.logical_and)
    xz_mid = construct_surface_dict(x, y_mid, z, s1_midy-s1_ad_guess, x_bounds=(x_mid, np.max(x)), z_bounds=(z_mid, np.max(z)), bool_function=np.logical_and)
    yz_mid = construct_surface_dict(x_mid, y, z, s1_midx-s1_ad_guess, y_bounds=(y_mid, np.max(y)), z_bounds=(z_mid, np.max(z)), bool_function=np.logical_and)

    side_list = [xy_side, xz_side, yz_side, xy_mid, xz_mid, yz_mid]
    
    
    
    ax.view_init(25,10)
    cmap = cm.get_cmap('RdBu_r')

    for d in side_list:
            x_3d = d['x']
            y_3d = d['y']
            z_3d = d['z']
            sfc = cmap(divnorm(d['surfacecolor']))
            surf = ax.plot_surface(x_3d, y_3d, z_3d, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
            ax.plot_wireframe(x_3d, y_3d, z_3d, ccount=1, rcount=1, linewidth=1, color='black')
    x_a = np.array([[Lx/2, Lx/2], [np.max(x), np.max(x)]])
    y_a = np.array([[Ly/2, np.max(y)], [Ly/2,np.max(y)]])
    z_a = np.array([[Lz/2, Lz/2], [Lz/2, Lz/2]])

    x_b = np.array([[Lx/2, Lx/2], [Lx/2, Lx/2]])
    y_b = np.array([[Ly/2, np.max(y)], [Ly/2,np.max(y)]])
    z_b = np.array([[Lz/2, Lz/2], [np.max(z), np.max(z)]])

    ax.plot_wireframe(x_a, y_a, z_a, ccount=1, rcount=1, linewidth=1, color='black')
    ax.plot_wireframe(x_b, y_b, z_b, ccount=1, rcount=1, linewidth=1, color='black')
    
    ax.set_box_aspect((Lx,Ly,Lz), zoom=1.2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    ax.patch.set_alpha(0)
    ax.grid(False)
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0,Ly)
    ax.set_zlim(0,Lz)
    ax.xaxis._axinfo['juggled'] = (1,1,0)
    ax.yaxis._axinfo['juggled'] = (0,0,0)
    ax.zaxis._axinfo['juggled'] = (0,0,0)
    
    
# Try this with first 40 frames first

# Panel A data
print('Loading Data')
with h5py.File('Ra1e5_eps1.5_2D/slices_s243.h5') as f:
    x_A = f['scales']['x']['1.0'][:]
    z_A = f['scales']['z']['1.0'][:]
    data_A = f['tasks']['s1'][:,:,:] - s_ad_A

for i in range(244,251):
    with h5py.File('Ra1e5_eps1.5_2D/slices_s'+str(i)+'.h5') as f:
        data_A = np.append(data_A, f['tasks']['s1'][:,:,:] - s_ad_A, axis=0)
print('Panel A', np.shape(data_A))

    
# Panel B data
with h5py.File('Ra1e7_eps1.5_2D/slices_s86.h5') as f:
    x_B = f['scales']['x']['1.0'][:]
    z_B = f['scales']['z']['1.0'][:]
    data_B = f['tasks']['s1'][:,:,:] - s_ad_B

for i in range(87,94):
    with h5py.File('Ra1e7_eps1.5_2D/slices_s'+str(i)+'.h5') as f:
        data_B = np.append(data_B, f['tasks']['s1'][:,:,:] - s_ad_B, axis=0)
print('Panel B', np.shape(data_B))

# Panel C data
with h5py.File('Ra1e9_eps1.5_2D/slices_s12.h5') as f:
    x_C = f['scales']['x']['1.0'][:]
    z_C = f['scales']['z']['1.0'][:]
    data_C = f['tasks']['s1'][:,:,:] - s_ad_C
for i in range(13,20):
    with h5py.File('Ra1e9_eps1.5_2D/slices_s'+str(i)+'.h5') as f:
        data_C = np.append(data_C, f['tasks']['s1'][:,:,:] - s_ad_C, axis=0)
print('Panel C', np.shape(data_C))

# Panel D data
with h5py.File('Ra4.3e9_eps3.3_2D/slices_s11.h5') as f:
    x_D = f['scales']['x']['1.0'][:]
    z_D = f['scales']['z']['1.0'][:]
    data_D = f['tasks']['s1'][:,:,:] - s_ad_D
for i in range(12,19):
    with h5py.File('Ra4.3e9_eps3.3_2D/slices_s'+str(i)+'.h5') as f:
        data_D = np.append(data_D, f['tasks']['s1'][:,:,:] - s_ad_D, axis=0)
print('Panel D', np.shape(data_D))

#Panel E data
with h5py.File('Ra1e5_eps1.5_3D/slices_s437.h5') as f:
    sidex_E = f['tasks']['s1_sidex'][:,0,:,:]
    sidey_E = f['tasks']['s1_sidey'][:,:,0,:]
    midx_E = f['tasks']['s1_midx'][:,0,:,:]
    midy_E = f['tasks']['s1_midy'][:,:,0,:]
    top_E = f['tasks']['s1_z0.95'][:,:,:,0]
    bottom_E = f['tasks']['s1_z0.05'][:,:,:,0]
    midz_E = f['tasks']['s1_midz'][:,:,:,0]
    x_E = f['scales/x']['1.0'][:]
    y_E = f['scales/y']['1.0'][:]
    z_E = f['scales/z']['1.0'][:]

for i in range(438, 453):
    with h5py.File('Ra1e5_eps1.5_3D/slices_s'+str(i)+'.h5') as f:
        sidex_E = np.append(sidex_E, f['tasks']['s1_sidex'][:,0,:,:], axis=0)
        sidey_E = np.append(sidey_E, f['tasks']['s1_sidey'][:,:,0,:], axis=0)
        midx_E = np.append(midx_E, f['tasks']['s1_midx'][:,0,:,:], axis=0)
        midy_E = np.append(midy_E, f['tasks']['s1_midy'][:,:,0,:], axis=0)
        top_E = np.append(top_E, f['tasks']['s1_z0.95'][:,:,:,0], axis=0)
        bottom_E = np.append(bottom_E, f['tasks']['s1_z0.05'][:,:,:,0], axis=0)
        midz_E = np.append(midz_E, f['tasks']['s1_midz'][:,:,:,0], axis=0)
print('Panel E', np.shape(midz_E))

# Panel F data
with h5py.File('Ra1e7_eps1.5_3D/slices_s95.h5') as f:
    sidex_F = f['tasks']['s1_sidex'][:,0,:,:]
    sidey_F = f['tasks']['s1_sidey'][:,:,0,:]
    midx_F = f['tasks']['s1_midx'][:,0,:,:]
    midy_F = f['tasks']['s1_midy'][:,:,0,:]
    top_F = f['tasks']['s1_z0.95'][:,:,:,0]
    bottom_F = f['tasks']['s1_z0.05'][:,:,:,0]
    midz_F = f['tasks']['s1_midz'][:,:,:,0]
    x_F = f['scales/x']['1.0'][:]
    y_F = f['scales/y']['1.0'][:]
    z_F = f['scales/z']['1.0'][:]

for i in range(96,108):
    with h5py.File('Ra1e7_eps1.5_3D/slices_s'+str(i)+'.h5') as f:
        sidex_F = np.append(sidex_F, f['tasks']['s1_sidex'][:,0,:,:], axis=0)
        sidey_F = np.append(sidey_F, f['tasks']['s1_sidey'][:,:,0,:], axis=0)
        midx_F = np.append(midx_F, f['tasks']['s1_midx'][:,0,:,:], axis=0)
        midy_F = np.append(midy_F, f['tasks']['s1_midy'][:,:,0,:], axis=0)
        top_F = np.append(top_F, f['tasks']['s1_z0.95'][:,:,:,0], axis=0)
        bottom_F = np.append(bottom_F, f['tasks']['s1_z0.05'][:,:,:,0], axis=0)
        midz_F = np.append(midz_F, f['tasks']['s1_midz'][:,:,:,0], axis=0)


for i in range(2,6):
    with h5py.File('Ra1e7_eps1.5_3D/slices_s'+str(i)+'.h5') as f:
        sidex_F = np.append(sidex_F, f['tasks']['s1_sidex'][:,0,:,:], axis=0)
        sidey_F = np.append(sidey_F, f['tasks']['s1_sidey'][:,:,0,:], axis=0)
        midx_F = np.append(midx_F, f['tasks']['s1_midx'][:,0,:,:], axis=0)
        midy_F = np.append(midy_F, f['tasks']['s1_midy'][:,:,0,:], axis=0)
        top_F = np.append(top_F, f['tasks']['s1_z0.95'][:,:,:,0], axis=0)
        bottom_F = np.append(bottom_F, f['tasks']['s1_z0.05'][:,:,:,0], axis=0)
        midz_F = np.append(midz_F, f['tasks']['s1_midz'][:,:,:,0], axis=0)

print('Panel F', np.shape(midz_F))


    

cmap_Ra =cm.get_cmap('inferno')
colors_Ra = cmap_Ra(np.linspace(0.1,0.8,6))
#colors_H = palettable.cmocean.sequential.Phase_11.mpl_colors
colors_H = [(0.6588235294117647, 0.47058823529411764, 0.050980392156862744), (0.788235294117647, 0.36470588235294116, 0.2196078431372549), (0.8627450980392157, 0.23529411764705882, 0.42745098039215684), (0.8509803921568627, 0.14901960784313725, 0.7254901960784313), (0.7137254901960784, 0.29411764705882354, 0.9294117647058824), (0.49019607843137253, 0.45098039215686275, 0.9411764705882353), (0.23137254901960785, 0.5490196078431373, 0.796078431372549), (0.09019607843137255, 0.5843137254901961, 0.596078431372549), (0.09803921568627451, 0.6039215686274509, 0.3686274509803922), (0.4470588235294118, 0.5607843137254902, 0.0784313725490196), (0.6588235294117647, 0.47058823529411764, 0.050980392156862744)]
#plot data


def plot_frame(i):
    print('Plotting Frame', i)
    # Change cutoff max and min inde
    cmap = cm.get_cmap('RdBu_r')
    
    
    fig =plt.figure(figsize=(15,3))
    ax1 = fig.add_axes([0, 0.6, 0.26, 0.4])
    ax2 = fig.add_axes([0, 0, 0.26, 0.4])
    ax3 = fig.add_axes([0.25, -.25, 0.5, 1.5], projection='3d')
    
    ax4 = fig.add_axes([0, -0.55, 0.26, 0.4])
    ax5 = fig.add_axes([0, -1.15, 0.26, 0.4])
    ax6 = fig.add_axes([0.25, -1.4, 0.5, 1.5], projection='3d')
    
    
    ax4.annotate('', xy=(-0.1,-0.45), xycoords='axes fraction', xytext=(1.1, -0.45), arrowprops=dict(arrowstyle="-", color='grey'))
    
    fig.subplots_adjust(top=0.8)
    
    # plot 3D data
    plot_3D_box(ax6, sidex_F[i,:,:], sidey_F[i,:,:], midx_F[i,:,:], midy_F[i,:,:], top_F[i,:,:], bottom_F[i,:,:], midz_F[i,:,:], x_F, y_F, z_F, divnorm_F, s_ad_F)
    plot_3D_box(ax3, sidex_E[i,:,:], sidey_E[i,:,:], midx_E[i,:,:], midy_E[i,:,:], top_E[i,:,:], bottom_E[i,:,:], midz_E[i,:,:], x_E, y_E, z_E, divnorm_E, s_ad_E)
    
    # Plot 2D data
    ax1.pcolormesh(x_A, z_A, data_A[i,:,:].T, cmap='RdBu_r',rasterized=True, shading='nearest', norm=divnorm_A)
    ax2.pcolormesh(x_B, z_B, data_B[i,:,:].T, cmap='RdBu_r',rasterized=True, shading='nearest', norm=divnorm_B)
    
    ax4.pcolormesh(x_C, z_C, data_C[i,:,:].T, cmap='RdBu_r',rasterized=True, shading='nearest', norm=divnorm_C)
    ax5.pcolormesh(x_D, z_D, data_D[i,:,:].T, cmap='RdBu_r',rasterized=True, shading='nearest', norm=divnorm_D)
    
    # set 2d aspect ratios
    ax1.set_box_aspect(1/4)
    ax2.set_box_aspect(1/4)
    ax4.set_box_aspect(1/4)
    ax5.set_box_aspect(1/4)
    
    
    
    ### PANEL A ANNOTATION
    fig.text(x=.005,y= 1, s=r'$\mathbf{{a)}}\;\;_{{{:.3f}}}^{{+{:.3f}}}$'.format(v_A[0], v_A[1]), fontsize=14)
    fig.text(x=0.06, y=1, s=r'2D $|$', fontsize=12)
    txt_a_Ra = fig.text(x=0.087, y=1, s=r'${\mathrm{Ra} = 2\times 10^3 \mathrm{Ra}_\mathrm{crit}}$', color=colors_Ra[1], fontsize=12)
    fig.text(x=0.18, y=1, s=r'$|$', fontsize=12)
    txt_a_H = fig.text(x=0.187, y=1, s=r'$\mathcal{H}=0.75$',color=colors_H[3], fontsize=12)
    
    
    ### PANEL B ANNOTATION
    fig.text(x=.005,y= .4, s=r'$\mathbf{{b)}}\;\;_{{{:.3f}}}^{{+{:.3f}}}$'.format(v_B[0], v_B[1]), fontsize=14)
    fig.text(x=0.06, y=.4, s=r'2D $|$', fontsize=12)
    txt_b_Ra = fig.text(x=0.087, y=.4, s=r'${\mathrm{Ra} = 2\times 10^5 \mathrm{Ra}_\mathrm{crit}}$', color=colors_Ra[3], fontsize=12)
    fig.text(x=0.18, y=.4, s=r'$|$', fontsize=12)
    txt_b_H = fig.text(x=0.187, y=.4, s=r'$\mathcal{H}=0.75$',color=colors_H[3], fontsize=12)
    
    
    ### PANEL C ANNOTATION
    fig.text(x=.005,y= -0.15, s=r'$\mathbf{{c)}}\;\;_{{{:.3f}}}^{{+{:.3f}}}$'.format(v_C[0], v_C[1]), fontsize=14)
    fig.text(x=0.06, y=-0.15, s=r'2D $|$', fontsize=12)
    txt_c_Ra = fig.text(x=0.087, y=-0.15, s=r'${\mathrm{Ra} = 2\times 10^7 \mathrm{Ra}_\mathrm{crit}}$', color=colors_Ra[5], fontsize=12)
    fig.text(x=0.18, y=-0.15, s=r'$|$', fontsize=12)
    txt_c_H = fig.text(x=0.187, y=-0.15, s=r'$\mathcal{H}=0.75$',color=colors_H[3], fontsize=12)
    
    
    ### PANEL D ANNOTATION
    fig.text(x=.005,y=-0.75,  s=r'$\mathbf{{d)}}\;\;_{{{:.3f}}}^{{+{:.3f}}}$'.format(v_D[0], v_D[1]), fontsize=14)
    fig.text(x=0.06, y=-0.75, s=r'2D $|$', fontsize=12)
    txt_d_Ra = fig.text(x=0.087, y=-0.75, s=r'${\mathrm{Ra} = 2\times 10^7 \mathrm{Ra}_\mathrm{crit}}$', color=colors_Ra[5], fontsize=12)
    fig.text(x=0.18, y=-0.75, s=r'$|$', fontsize=12)
    txt_d_H = fig.text(x=0.187, y=-0.75, s=r'$\mathcal{H}=16.5$',color=colors_H[7], fontsize=12)
    
    
    ### PANEL E ANNOTATION
    fig.text(x=.4,y= 1,  s=r'$\mathbf{{e)}}\;\;_{{{:.3f}}}^{{+{:.3f}}}$'.format(v_E[0], v_E[1]), fontsize=14)
    fig.text(x=0.46, y=1, s=r'3D $|$', fontsize=12)
    txt_e_Ra = fig.text(x=0.487, y=1, s=r'${\mathrm{Ra} = 2\times 10^3 \mathrm{Ra}_\mathrm{crit}}$', color=colors_Ra[1], fontsize=12)
    fig.text(x=0.58, y=1, s=r'$|$', fontsize=12)
    txt_e_H = fig.text(x=0.587, y=1, s=r'$\mathcal{H}=0.75$',color=colors_H[3], fontsize=12)
    
    
    ### PANEL F ANNOTATION
    fig.text(x=.4,y= -0.15, s=r'$\mathbf{{f)}}\;\;_{{{:.3f}}}^{{+{:.3f}}}$'.format(v_F[0], v_F[1]), fontsize=14)
    fig.text(x=0.46, y=-0.15, s=r'3D $|$', fontsize=12)
    txt_f_Ra = fig.text(x=0.487, y=-0.15, s=r'${\mathrm{Ra} = 2\times 10^5 \mathrm{Ra}_\mathrm{crit}}$', color=colors_Ra[3], fontsize=12)
    fig.text(x=0.58, y=-0.15, s=r'$|$', fontsize=12)
    txt_f_H = fig.text(x=0.587, y=-0.15, s=r'$\mathcal{H}=0.75$',color=colors_H[3], fontsize=12)
    
    
    
    # outline colored text
    txt_list = [txt_a_Ra, txt_b_Ra, txt_c_Ra, txt_d_Ra, txt_e_Ra, txt_f_Ra, txt_a_H, txt_b_H, txt_c_H, txt_d_H, txt_e_H, txt_f_H ]
    for txt in txt_list:
        txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='k')])

    # plot colorbar
    cbar_ax = fig.add_axes([0.01, 1.1, 0.64, 0.05])
    sm = plt.cm.ScalarMappable(norm=divnorm_F, cmap='RdBu_r')
    
    
    cb = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', boundaries=np.linspace(v_F[0],v_F[1],1000), ticks=[v_F[0],0,v_F[1]])
    cb.ax.set_xticklabels(['Min','0','Max'], fontsize=11)
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('top')
    plt.savefig(f'parallel_test/dynamics_movie_{i:03}.png', dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


with Pool(40) as pool:
    pool.map(plot_frame, range(320))
