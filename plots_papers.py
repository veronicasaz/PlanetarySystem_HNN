import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import rebound
import h5py

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.formatter.useoffset'] = False

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from plot_tools import color1, color2, color3, trunc, CustomTicker


def plot_NeurIPS(sim, sim2, sim3, t, asteroids, asteroids_extra, t_end, h, typePlot):
    """
    plot_general: plot simulation with WH, HNN and DNN only 3 columns
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN and DNN
        t: computation time for WH, HNN and DNN
        asteroids: number of asteroids
        asteroids_extra: number of extrapolation asteroids
    """
    ########################################################
    #### Plots together
    ########################################################
    asteroids += asteroids_extra
    t_num, t_nn, t_dnn = t

    fig, axes = plt.subplots(2,3, figsize=(20,8))
    data_nb = sim.buf.recorder.data
    data_nih = sim2.buf.recorder.data
    data_dnn = sim3.buf.recorder.data
    time = np.linspace(0, t_end, data_nih['x'].shape[0])

    # Add names of asteroids
    names = ['Sun', 'Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))

    labelsize = 25
    titlesize = 24
    line = ['-', '--', '-.', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
    lnwidth = 3
    # axes[0,0].set_title("Numerical integrator result: %.1f s"%(t_num), fontsize = titlesize)
    # axes[0,1].set_title("Hamiltonian Neural Network result: %.1f s"%(t_nn), fontsize = titlesize)
    # axes[0,2].set_title("Deep Neural Network result: %.1f s"%(t_dnn), fontsize = titlesize)
    axes[0,0].set_title("Numerical integrator", fontsize = titlesize)
    axes[0,1].set_title("Hamiltonian Neural Network", fontsize = titlesize)
    axes[0,2].set_title("Deep Neural Network", fontsize = titlesize)
    for i in range(1, data_nih['x'].shape[1]):
        axes[0,0].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[0,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[0,2].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = color2[i-1],linewidth = lnwidth,  label=names[i])
    
    axes[0,0].set_ylabel('$e$',fontsize = labelsize)
    axes[0,0].legend(loc = 'lower left', fontsize = 22, \
                framealpha = 0.9, bbox_to_anchor=(0.5, 1.22, 2.7, 1.5),\
                ncol=5, mode="expand", borderaxespad=0.)

    # energy drift in the second row
    axes[1,1].plot(time, sim2.energy, linestyle = '-',  color = color3[1],alpha=1, label= 'Error with WH-HNN')
    axes[1,2].plot(time, sim3.energy, alpha=1, linestyle = '-',  color = color3[2],label= 'Error with WH-DNN')
    for i in range(3):
        axes[1,i].plot(time, sim.energy, alpha=1, linestyle = '-',  color = color3[0], label= 'Error with WH')
        axes[1,i].ticklabel_format(useOffset=False)
        axes[0,i].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[1,i].set_xlabel('$t$ (yr)',fontsize = labelsize)
        
    axes[1,0].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1,0].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9)
    axes[1,1].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9)
    if typePlot == 'JS':
        axes[1,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)
    else:
        axes[1,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)

    for i in range(2):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.png' % t_end)
    plt.show()

def plot_CompPhys_trajectory(sim, sim2, sim3, t, t_end, asteroids, asteroids_extra, typePlot):
    """
    plot_general: plot simulation with WH, HNN and DNN only 3 columns
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN and DNN
        t: computation time for WH, HNN and DNN
        asteroids: number of asteroids
        asteroids_extra: number of extrapolation asteroids
    """
    ########################################################
    #### Plots together
    ########################################################
    asteroids += asteroids_extra
    t_num, t_nn, t_dnn = t

    fig, axes = plt.subplots(2,3, figsize=(20,10))
    data_nb = sim.buf.recorder.data
    data_nih = sim2.buf.recorder.data
    data_dnn = sim3.buf.recorder.data
    time = np.linspace(0, t_end, data_nih['x'].shape[0])

    # Add names of asteroids
    names = ['Sun', 'Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))

    labelsize = 29
    titlesize = 27
    line = ['-', '-',  '-', '-', '-','-', '-', '-', '-','-', '-', '-', '-']
    lnwidth = 3
    
    for col in range(3):
        axes[0,col].scatter(data_nb['x'][:,0], data_nb['y'][:,0], linestyle = line[0], s = 20, color = 'black', label=names[0])

    for i in range(1, data_nih['x'].shape[1]):
        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = color2[i-1], label=names[i])
        axes[0,1].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = color2[i-1], label=names[i])
        axes[0,2].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = color2[i-1], label=names[i])
        
        # eccentricity in the second column
        axes[1,0].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[1,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[1,2].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = color2[i-1],linewidth = lnwidth,  label=names[i])            

    axes[0,0].set_title("Numerical integrator", fontsize = titlesize)
    axes[0,1].set_title("Hamiltonian Neural Network", fontsize = titlesize)
    axes[0,2].set_title("Deep Neural Network", fontsize = titlesize)


    min_lim = data_nb['ecc'][:, 1:].min()
    max_lim = data_nb['ecc'][:, 1:].max()
    for col in range(3):
        axes[0,col].axis('equal')
        axes[0,col].set_xlabel('$x$ (au)',fontsize = labelsize)
        axes[0,col].set_ylabel('$y$ (au)',fontsize = labelsize)
        
        axes[1,col].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[1,col].set_ylabel('$e$',fontsize = labelsize)

        axes[0, col].set_xlim(data_nb['x'].min()*1.1, data_nb['x'].max()*1.1)
        axes[0, col].set_ylim(data_nb['y'].min()*1.1, data_nb['y'].max()*1.1)
        axes[1, col].set_xlim(time.min()-50, time.max()+50)
        axes[1, col].set_ylim(min_lim*0.9, max_lim*1.2)
        # axes[1, col].set_xlim(min(data_nb['ecc'])*0.9, max(data_nb['ecc'])*1.1)
        # axes[1, col].set_ylim(min(data_nb['ecc'])*0.9, max(data_nb['ecc'])*1.1)


    axes[0,0].legend(loc = 'lower left', fontsize = 27, \
                framealpha = 0.9, bbox_to_anchor=(0.0, 1.22, 3.5, 1.5),\
                ncol=6, mode="expand", borderaxespad=0.)

    for i in range(2):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=25)
            axes[i,j].tick_params(axis='both', which='minor', labelsize=25)
            axes[i, j].locator_params(axis = 'x', nbins=5)
            axes[i, j].locator_params(axis = 'y', nbins=5)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.png' % t_end)
    plt.show()

def plot_CompPhys_trajectory_JS(sim, sim2, sim3, t, t_end, asteroids, asteroids_extra, typePlot):
    """
    plot_general: plot simulation with WH, HNN and DNN only 3 columns
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN and DNN
        t: computation time for WH, HNN and DNN
        asteroids: number of asteroids
        asteroids_extra: number of extrapolation asteroids
    """
    ########################################################
    #### Plots together
    ########################################################
    asteroids += asteroids_extra
    t_num, t_nn, t_dnn = t

    fig, axes = plt.subplots(3,3, figsize=(19,12))
    data_nb = sim.buf.recorder.data
    data_nih = sim2.buf.recorder.data
    data_dnn = sim3.buf.recorder.data
    time = np.linspace(0, t_end, data_nih['x'].shape[0])

    # Add names of asteroids
    names = ['Sun', 'Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))

    labelsize = 29
    titlesize = 27
    line = ['-', '-',  '-', '-', '-','-', '-', '-', '-','-', '-', '-', '-']
    lnwidth = 3
    
    for col in range(3):
        axes[0,col].scatter(data_nb['x'][:,0], data_nb['y'][:,0], linestyle = line[0], s = 20, color = 'black', label=names[0])

    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = color2[i-1], label=names[i])        
        axes[0,1].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = color2[i-1], label=names[i])
        axes[0,2].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = color2[i-1], label=names[i])

        # eccentricity in the second column
        axes[1,0].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[1,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[1,2].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = color2[i-1],linewidth = lnwidth,  label=names[i])            

    # energy drift in the second column
    color_e = color2[5]
    color_e2 = color2[6]
    color_e3 = color2[7]
    lnwidth = 2

    axes[2,0].plot(time, np.array(sim.energy) *1e3 , linestyle = '-', linewidth = lnwidth,color = color_e2, alpha=1, label = 'Error with WH')
    axes[2,1].plot(time, np.array(sim2.energy)*1e3,  linestyle = '-',  linewidth = lnwidth,color = color_e2,alpha=1, label= 'Error with WH-HNN')
    axes[2,2].plot(time, np.array(sim3.energy) *1e3, alpha=1, linestyle = '-',linewidth = lnwidth, color = color_e2,label= 'Error with WH-DNN')

    axes[0,0].set_title("Numerical integrator", fontsize = titlesize)
    axes[0,1].set_title("Hamiltonian Neural Network", fontsize = titlesize)
    axes[0,2].set_title("Deep Neural Network", fontsize = titlesize)
    
    axes[0,0].set_ylabel('$y$ (au)',fontsize = labelsize)
    axes[1,0].set_ylabel('$e$',fontsize = labelsize)  
    axes[2,0].set_ylabel(r'$dE/E_0 \;(\times 10^{-3})$',fontsize = labelsize)

    min_lim = data_nb['ecc'][:, 1:].min()
    max_lim = data_nb['ecc'][:, 1:].max()
    for col in range(3):
        axes[0,col].axis('equal')
        axes[0,col].set_xlabel('$x$ (au)',fontsize = labelsize)
    
        axes[1,col].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[1,col].get_yaxis().set_major_formatter('{x:1.3f}')

        # axes[2,col].plot(time, np.array(sim.energy) *1e3 , linestyle = '-', linewidth = lnwidth,color = color_e, alpha=1, label = 'Error with WH')
        axes[2,col].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[2,col].ticklabel_format(useOffset=False)

        axes[0, col].set_xlim(data_nb['x'].min()*1.1, data_nb['x'].max()*1.1)
        axes[0, col].set_ylim(data_nb['y'].min()*1.1, data_nb['y'].max()*1.1)
        
        axes[1, col].set_xlim(time.min()-50, time.max()+50)
        axes[1, col].set_ylim(min_lim*0.99, max_lim*1.01)
        
        axes[2, col].set_xlim(time.min()-50, time.max()+50)
    min_lim2 = min(np.array(sim.energy)*1e3)
    max_lim2 = max(np.array(sim2.energy)*1e3)
    min_lim3 = min(np.array(sim3.energy)*1e3)
    max_lim3 = max(np.array(sim3.energy)*1e3)
    
    axes[2, 0].set_ylim(min_lim2 - min_lim2*1e-6, max_lim2 + max_lim2*1e-6)
    axes[2, 1].set_ylim(min_lim2 - min_lim2*1e-6, max_lim2+ max_lim2*1e-6)
    axes[2, 2].set_ylim(min_lim3 -min_lim3*1e-4, max_lim3+ max_lim3*1e-5)
        
    axes[0,0].legend(loc = 'lower left', fontsize = 27, \
                framealpha = 0.9, bbox_to_anchor=(0.0, 1.24, 2.0, 1.5),\
                ncol=6, mode="expand", borderaxespad=0.)
    # axes[2,0].legend(loc = 'lower left', fontsize = 23, framealpha = 0.9)
    # axes[2,1].legend(loc = 'lower left', fontsize = 23, framealpha = 0.9)
    # axes[2,2].legend(loc = 'upper left', fontsize = 23, framealpha = 0.8)

    for i in range(3):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=25)
            axes[i,j].tick_params(axis='both', which='minor', labelsize=25)
            axes[i, j].locator_params(axis = 'x', nbins=6)
            axes[i, j].locator_params(axis = 'y', nbins=6)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.png' % t_end)
    plt.show()

def plot_general_flagvsnoflag(sim, sim2, sim3, t, asteroids, asteroids_extra):
    """
    plot_general_flagvsnoflag: plot simulation with WH, HNN flag and HNN no flag
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN flag and HNN no flag
        t: computation time for WH, HNN flag and HNN no flag
        asteroids: number of asteroids
        asteroids_extra: number of extrapolation asteroids
    """
    asteroids += asteroids_extra

    # Using the HNN
    t_num = t[0][0]
    t_nn = t[0][1]
    t_nn2 = t[1][1]

    fig, axes = plt.subplots(3,5, figsize=(18,6))
    data_nb = sim.buf.recorder.data
    data_nih = sim2.buf.recorder.data
    data_nih2 = sim3.buf.recorder.data
    time = np.linspace(0, t_end, data_nih['x'].shape[0])

    # Add names of asteroids
    names = ['Sun', 'Jupiter', 'Saturn']

    for j in range(asteroids):
        names.append("Asteroid%i"%(j+1))

    axes[0,2].set_title("Numerical integrator result: %.3f s"%(t_num))
    axes[1,2].set_title("Neural Network result without flag: %.3f s"%(t_nn))
    axes[2,2].set_title("Neural Network result with flag: %.3f s"%(t_nn2))
    
    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[2,0].plot(data_nih2['x'][:,i], data_nih2['y'][:,i], '-', label=names[i], alpha=0.6)

        # semi-major in the second column
        axes[0,1].plot(time, data_nb['a'][:,i], label=names[i], alpha=0.6)
        axes[1,1].plot(time, data_nih['a'][:,i], label=names[i], alpha=0.6)
        axes[2,1].plot(time, data_nih2['a'][:,i], label=names[i], alpha=0.6)
        
        # eccentricity in the second column
        axes[0,2].plot(time, data_nb['ecc'][:,i], label=names[i], alpha=0.6)
        axes[1,2].plot(time, data_nih['ecc'][:,i], label=names[i], alpha=0.6)
        axes[2,2].plot(time, data_nih2['ecc'][:,i], label=names[i], alpha=0.6)
        
        axes[0,3].plot(time, np.degrees(data_nb['inc'][:,i]), label=names[i], alpha=0.6)
        axes[1,3].plot(time, np.degrees(data_nih['inc'][:,i]), label=names[i], alpha=0.6)
        axes[2,3].plot(time, np.degrees(data_nih2['inc'][:,i]), label=names[i], alpha=0.6)

    # energy drift in the second column
    axes[0,4].plot(time, sim.energy, alpha=0.6)
    axes[1,4].plot(time, sim2.energy, alpha=0.6)
    axes[2,4].plot(time, sim3.energy, alpha=0.6)
        
    for i in range(3):
        axes[i,0].axis('equal')
        axes[i,0].legend(loc = 'upper right')
        axes[i,0].set_ylabel('$y$')        
        axes[i,1].set_ylabel('$a$ [au]')
        axes[i,2].set_ylabel('$e$')
        axes[i,3].set_ylabel('$i$ [deg]')
        axes[i,4].set_ylabel('$dE/E_0$')

    axes[2,0].set_xlabel('$x$')
    axes[2,1].set_xlabel('$t$ [years]')
    axes[2,2].set_xlabel('$t$ [years]')
    axes[2,3].set_xlabel('$t$ [years]')   
    axes[2,4].set_xlabel('$t$ [years]')

    plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_flagcomparison%dyr.pdf' % t_end)
    plt.show()

def plot_accel_flagvsnoflag(accelerations_WH, accelerations_ANN_noflag, accelerations_ANN_flag, 
                flags, t, asteroids, asteroids_extra, t_end):
    """
    plot_accelerations_flagvsnoflag: plot accelerations predicted with WH, HNN flag and HNN no flag
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN flag and HNN no flag
    """
    # Add names of asteroids
    names = ['Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))
    for k in range(asteroids + asteroids_extra):
        names.append("Asteroid %i"%(k+ 1+ asteroids))

    h = 1e-1
    x = np.arange(0, len(accelerations_ANN_noflag)*h, h)

    asteroids_plot = asteroids+asteroids_extra
    fig, axes = plt.subplots(3,1+asteroids_plot, figsize=(15,8))
    fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.4, wspace= 0.25)

    for plot in range(2,3+asteroids_plot):
        a_WH = np.zeros(len(accelerations_ANN_noflag))
        a_ANN = np.zeros(len(accelerations_ANN_noflag))
        a_DNN = np.zeros(len(accelerations_ANN_flag))
        
        for item in range(len(accelerations_ANN_noflag)):
            a_WH[item] = np.linalg.norm(accelerations_WH[item, plot*3:plot*3+3])
            a_ANN[item] = np.linalg.norm(accelerations_ANN_noflag[item, plot*3:plot*3+3])
            a_DNN[item] = np.linalg.norm(accelerations_ANN_flag[item, plot*3:plot*3+3])
        
        l1, = axes[0, plot-2].plot(x, a_WH, color = color1[0], label = 'WH')
        l2, = axes[1, plot-2].plot(x, a_ANN, color = color1[1], label = 'Without flags')
        l3, = axes[2, plot-2].plot(x, a_DNN, color = color1[2], label = 'With flags')
        
        index_DNN = np.where(flags[:, plot] == 0)[0]
        x2_DNN = np.copy(x)
        x2_DNN = np.delete(x2_DNN, index_DNN)
        l4 = axes[2, plot-2].scatter(x2_DNN, np.delete(a_DNN, index_DNN), color = color1[2], label = 'Numerically')
        axes[0,plot-2].set_title(names[plot-1], fontsize = 24)
        
        axes[2,plot-2].set_xlabel('t (yr)', fontsize = 28)
        axes[2,plot-2].annotate("Flags: %i / %i"%(np.count_nonzero(flags[:, plot]), len(accelerations_ANN_flag)), \
                xy =  (x[0]+0.9, max((a_DNN))*0.9), fontsize = 20)
        
        for iterate in range(3):
            axes[iterate,plot-2].grid(alpha = 0.5)        
            ticks = -np.log10(axes[iterate,plot-2].get_yticks())
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
            axes[iterate,plot-2].set_yticklabels(np.round(trunc(axes[iterate,plot-2].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 20)
            axes[iterate,plot-2].set_xticklabels(np.round(trunc(axes[iterate,plot-2].get_xticks(), decs = 0), decimals = 0), rotation = 0, fontsize = 20)
            axes[iterate, plot-2].xaxis.set_major_formatter(FormatStrFormatter('%i'))

            axes[iterate,0].set_ylabel(r'a ($\rm au/yr^2$)', fontsize = 28)
    
    lgnd = axes[0,0].legend([l1, l2, l3, l4], ['WH', 'Without flags', 'With flags', 'Flags'], loc = 'lower left', fontsize = 23, \
                framealpha = 0.9, bbox_to_anchor=(0.1, 1.3, 3.3, 1.0),\
                ncol=4, mode="expand", borderaxespad=0., handletextpad=0.3)

    # plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_accel_%dyr_flagcomparison.png' % t_end, bbox_inches='tight')
    plt.show()

def plot_accel_flagvsR(accel_i, accelerations_baseline,\
                flags_i, t_i, asteroids, asteroids_extra, R_i):
    """
    plot_accelerations_flagvsnoflag: plot accelerations predicted with WH, HNN flag and HNN no flag
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN flag and HNN no flag
    """
    # Add names of asteroids
    names = ['Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))


    accelerations_1 = accel_i[0]
    accelerations_2 = accel_i[2]
    accelerations_3 = accel_i[4]
    R = [R_i[0], R_i[2], R_i[4]]
    t = [t_i[0], t_i[2], t_i[4]]
    flags_i = [flags_i[0],flags_i[2], flags_i[4]]

    h = 1e-1
    x = np.arange(0, len(accelerations_2)*h, h)

    asteroids_plot = asteroids+asteroids_extra

    fig = plt.figure(figsize = (15, 8), constrained_layout = True)
    gs = GridSpec(2, 3, figure = fig, height_ratios= [2, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    axes = [ax1, ax2, ax3, ax4]

    a_1 = np.zeros(len(accelerations_2))
    a_2 = np.zeros(len(accelerations_2))
    a_3 = np.zeros(len(accelerations_2))
    a_baseline = np.zeros(len(accelerations_baseline))
        
    body = 3
    for item in range(len(accelerations_2)):
        a_1[item] = np.linalg.norm(accelerations_1[item, 3*body:3*body+3])
        a_2[item] = np.linalg.norm(accelerations_2[item, 3*body:3*body+3])
        a_3[item] = np.linalg.norm(accelerations_3[item, 3*body:3*body+3])
        a_baseline[item] = np.linalg.norm(accelerations_baseline[item, 3*body:3*body+3])
    accel = [a_1, a_2, a_3]
        
    for plot in range(3):
        axes[plot].plot(x, accel[plot], color = color1[0], label = 'WH-HNN')
        flags = flags_i[plot]
        index_DNN = np.where(flags[:, 3] == 0)[0]
        x2_DNN = np.copy(x)
        x2_DNN = np.delete(x2_DNN, index_DNN)
        axes[plot].scatter(x2_DNN, np.delete(accel[plot], index_DNN), color = color1[2], label = 'Flags')
    
        axes[plot].set_title( r'$R$ = %0.1f'%R[plot], fontsize = 24)
        axes[plot].set_xlabel('$t$ (yr)', fontsize = 25)
        axes[plot].annotate("Flags: %i / %i"%(np.count_nonzero(flags[:, 3]), len(accelerations_ANN_flag)), \
            xy =  (x[len(x)//2]-5, max((accel[plot]))*0.92), fontsize = 20)
    
        axes[plot].grid(alpha = 0.5)        
        ticks = -np.log10(axes[plot].get_yticks())
        dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
        axes[plot].set_yticklabels(np.round(trunc(axes[plot].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 20)
        axes[plot].set_xticklabels(np.round(trunc(axes[plot].get_xticks(), decs = 2), decimals = dec+1), rotation = 0, fontsize = 20)
        
    axes[0].set_ylabel('a ($au/yr^2$)', fontsize = 25)
    lgnd = axes[0].legend(loc = 'lower left', fontsize = 23, \
                framealpha = 0.9, bbox_to_anchor=(0.0, 1.2, 1.0, 1.0),\
                ncol=4, mode="expand", borderaxespad=0., handletextpad=0.3)

    axes[3].plot(R_i, t_i, marker = 'o', markersize = 10, color = color1[0])
    axes[3].set_xlabel(r'$R$', fontsize = 25)
    axes[3].set_ylabel('Comput. time (s)', fontsize = 25)
    axes[3].grid(alpha = 0.5)        
    ticks = -np.log10(axes[3].get_yticks())
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes[3].set_yticklabels(np.round(trunc(axes[3].get_yticks(), decs = 6), decimals = dec+3), rotation = 0, fontsize = 20)
    axes[3].set_xticklabels(np.round(trunc(axes[3].get_xticks(), decs = 6), decimals = dec+4), rotation = 0, fontsize = 20)

    # plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_accel_%dyr_flagR.png' % t_end, bbox_inches='tight')
    plt.show()

def calculate_centralH(data, m):
    """
    calculate_centralH: calculate energy due to central body, potential + kinetic
    INPUTS: 
        data: samples and positions/velocities
        m: masses
    OUTPUTS:
        H: energy 
    """
    sr = rebound.Simulation()
    sr.units = {'yr', 'au', 'MSun'}

    T = 0
    U = 0
    G = sr.G
    r = data[:, 0:3]
    v = data[:, 3:]
    particles = np.shape(data)[0]
    for i in range(particles):
        T += m[i] * np.linalg.norm(v[i,:])**2 / 2 
    for j in range(1, particles):
        U -= G * m[0] *m[i] / np.linalg.norm(r[0, :]- r[i,:])
    H = T + U
    return H

def calculate_interH(data, m):
    """
    calculate_inteH: calculate interactive energy (energy due to mutual interactions)
    INPUTS: 
        data: samples and positions/velocities
        m: masses
    OUTPUTS:
        H: energy 
    """
    sr = rebound.Simulation()
    sr.units = {'yr', 'au', 'MSun'}

    T = 0
    U = 0
    G = sr.G
    r = data[:, 0:3]
    v = data[:, 3:]
    particles = np.shape(data)[0]
    for i in range(1, particles): # exclude central body
        for j in range(i+1, particles):
            U -= G * m[i] * m[j] / np.linalg.norm(r[j,:]- r[i,:])
    H = U
    return H

def plot_energyvsH(sim, sim2, t_end):
    """
    Plot interactive energy vs output of the HNN.   
        Only implemented for JS case 
    INPUTS:
        sim1: numerical integration
        sim2: numerical integration with HNN
    """
    total_H = np.zeros(len(sim2.H))
    time = np.linspace(0, sim.t, num = len(sim2.H))

    data_nb = sim.coord
    data_nih = sim2.coord

    # Calculate energy
    E_1 = np.zeros(len(sim2.H))
    E_2 = np.zeros(len(sim2.H))
    data = np.zeros((3, 6))
    for i in range(len(sim2.H)):
        data[:,0:3] = data_nb[i][0:3*3].reshape((3,3))
        data[:,3:] = data_nb[i][ 3*3:].reshape((3,3))
        E_1[i] = calculate_interH(data, sim.particles.masses)
        data[:,0:3] = data_nih[i][0:3*3].reshape((3,3))
        data[:,3:] = data_nih[i][3*3:].reshape((3,3))
        # E_2[i] = sim2.energy[i] - calculate_centralH(data, sim2.particles.masses) # also possible
        E_2[i] = calculate_interH(data, sim2.particles.masses)

    lw = 4
    fig, axes = plt.subplots(1,1, figsize=(8,6))
    axes.plot(time, E_1,  '-',linewidth = lw, color = color1[0], label = 'WH Energy')
    axes.plot(time, E_2, '--',  linewidth = lw, color = color1[2], label = 'WH-HNN Energy')
    axes.plot(time, np.array(sim2.H), linewidth = lw, color = color1[1], label = 'WH-HNN H')
    # plt.title('Comparison of interactive Hamiltonian \n and the predicted output of the HNN', fontsize = 13)
    plt.xlabel('$t$ (yr)', fontsize =30)
    plt.ylabel(r'Energy ($\rm {kg}\; {au}^2\; yr^{-2} \times 10^{-6}$)', fontsize =30)

    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    axes.ticklabel_format(useOffset=False, style='plain')
    ticks = -np.log10(abs(axes.get_yticks()))+1
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes.set_yticklabels(np.round(trunc(axes.get_yticks()*1e6, decs = 6), decimals = 2), fontsize = 27)
    
    plt.legend(fontsize = 25)
    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/HvsE_%dyr.png' % t_end)
    plt.show()

def plot_asteroids():
    """
    plot_asteroids: plot asteroids vs time and asteroids vs energy error
    """
    # Load data
    with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    # t_num = data['time']
    t_num = data['time_accel']
    e_energy = data['energy']
    h, t_end = data['settings']
    asteroids = data['asteroids']

    ########################################################
    #### Plots together
    ########################################################
    fig, axes = plt.subplots(1,2, figsize=(16,4), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.5, wspace= 0.55)

    axes[0].plot(asteroids, t_num[:,0], color = color1[0],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[0].plot(asteroids, t_num[:,1], color = color1[1],  linestyle='-', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axes[0].plot(asteroids, t_num[:,2], color = color1[2],  linestyle='-', linewidth = 2, marker = '^', markersize = 12,label = 'WH-HNN')
    
    axes[0].set_xlabel('Number of asteroids', fontsize = 27)
    axes[0].set_ylabel('Computation time (s)', fontsize = 27)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(alpha = 0.5)
    
    axes[0].tick_params(axis='both', which='major', labelsize=25)
    axes[0].tick_params(axis='both', which='minor', labelsize=25)
    axes[0].legend(fontsize = 22)

    axins = axes[0].inset_axes([0.72, 0.12, 0.25, 0.4])
    axins.plot(asteroids[-3:], t_num[-3:,0], color = color1[0],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axins.plot(asteroids[-3:], t_num[-3:,1], color = color1[1],  linestyle='-', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axins.plot(asteroids[-3:], t_num[-3:,2], color = color1[2],  linestyle='-', linewidth = 2, marker = '^', markersize = 12,label = 'WH-HNN')

    x1, x2, y1, y2 = (asteroids[-3]+asteroids[-2])/2, asteroids[-1]*1.1, 400, 7000
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.tick_params(axis='both', which='major', labelsize=18)
    axins.tick_params(axis='both', which='minor', labelsize=18)
    axins.set_yscale('linear')

    axes[0].indicate_inset_zoom(axins, edgecolor="black")

    e_rel = abs( ( np.sum(e_energy[:,0,-10:], axis = 1)/10 - e_energy[:,0,0])  / e_energy[:,0,0] ) *1e5
    e_rel2 = abs( ( np.sum(e_energy[:,1,-10:], axis = 1)/10 - e_energy[:,1,0])  / e_energy[:,1,0] ) *1e5
    e_rel3 = abs( ( np.sum(e_energy[:,2,-10:], axis = 1)/10 - e_energy[:,2,0])  / e_energy[:,2,0] ) *1e5

    axes[1].plot(asteroids, t_num[:,1] -t_num[:,0], color = color1[0], linewidth = 2, linestyle = '--')
    axes[1].plot(asteroids, t_num[:,2] -t_num[:,0], color = color1[0], linewidth = 2, linestyle = '-')
    axes[1].plot([], [], color = 'black', linestyle = '--', label = 'HNN')
    axes[1].plot([], [], color = 'black', linestyle = '-', label = 'WH-HNN ')
    axes[1].set_xlabel('Number of asteroids', fontsize = 27)
    axes[1].set_ylabel(r'$t - t_{WH}$ (s)', fontsize = 27, color = color1[0])
    axes[1].set_xscale('log')
    axes[1].set_yscale('symlog', linthresh = 1)
    axes[1].tick_params(axis='both', which='major', labelsize=25)
    axes[1].tick_params(axis='both', which='minor', labelsize=25)
    axes[1].tick_params(axis='y', labelcolor = color1[0])

    ax2 = axes[1].twinx()
    ax2.plot(asteroids, e_rel2-e_rel, linestyle = '--', linewidth = 2, color = color1[2])
    ax2.plot(asteroids, e_rel3-e_rel, linestyle = '-', linewidth = 2, color = color1[2])
    ax2.set_ylabel(r'$\varepsilon - \varepsilon_{WH} \; _{(\times1e5)}$ ', fontsize = 27, color = color1[2])
    ax2.tick_params(axis='y', which='major', labelsize=25)
    ax2.tick_params(axis='y', which='minor', labelsize=25)
    ax2.tick_params(axis='y', labelcolor = color1[2])
    axes[1].legend(fontsize = 22, loc = 'lower left', labelcolor = 'black')
    axes[1].grid(alpha = 0.5)

    # plt.suptitle("Time and Energy error \n $t_f$ = %0.3f and $h$ = %0.3f"%(t_end, h), fontsize = 18)
    plt.tight_layout()
    plt.savefig('./Experiments/AsteroidVsTime/timeVsError_%dyr.png' % t_end)
    plt.show()

def plot_asteroids_accel():
    """
    plot_asteroids_accel: plot asteroids vs time and asteroids vs energy error for computation time 
    only with accel calculation
    """
    # Load data
    with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    # t_num = data['time']
    t_num = data['time_accel']
    e_energy = data['energy']
    h, t_end = data['settings']
    asteroids = data['asteroids']

    ########################################################
    #### Plots together
    ########################################################
    fig, axes = plt.subplots(1,2, figsize=(16,5.5), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.5, wspace= 0.55)

    axes[0].plot(asteroids, t_num[:,0], color = color1[0],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[0].plot(asteroids, t_num[:,1], color = color1[1],  linestyle='-', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axes[0].plot(asteroids, t_num[:,2], color = color1[2],  linestyle='-', linewidth = 2, marker = '^', markersize = 12,label = 'WH-HNN')
    

    # t_1 = t_num[7, 1] / ((asteroids[7]+2) * np.log(asteroids[7]+2))  # First time is for 5 asteroids (+2 planets). time per operation
    # axes[0].plot(asteroids, t_1 *((asteroids+2) * np.log(asteroids+2)) , color = 'blue', linewidth = 3, alpha = 0.5, linestyle = '--', label = 'N log(N)' )
    t_2 = t_num[7, 0] / (asteroids[7]+2)**2 
    axes[0].plot(asteroids, t_2 *((asteroids+2)**2) , color = 'black', linewidth = 3, alpha = 0.7, linestyle = '--', label = r'$N^2$' )
    t_3 = t_num[8, 1] / ((asteroids[8]+2)) # First time is for 5 asteroids (+2 planets). time per operation
    axes[0].plot(asteroids, t_3 *(asteroids+2)  , color = 'black', linewidth = 3, alpha = 0.7, linestyle = ':', label = r'$N$' )
    

    axes[0].set_xlabel('Number of asteroids', fontsize = 27)
    axes[0].set_ylabel('Computation time (s)', fontsize = 27)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(alpha = 0.5)
    
    axes[0].tick_params(axis='both', which='major', labelsize=25)
    axes[0].tick_params(axis='both', which='minor', labelsize=25)
    # axes[0].legend(fontsize = 21, framealpha = 0.5, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    axes[0].legend(fontsize = 21, framealpha = 0.5, loc='best')

    e_rel = abs( ( np.sum(e_energy[:,0,-10:], axis = 1)/10 - e_energy[:,0,0])  / e_energy[:,0,0] ) *1e5
    e_rel2 = abs( ( np.sum(e_energy[:,1,-10:], axis = 1)/10 - e_energy[:,1,0])  / e_energy[:,1,0] ) *1e5
    e_rel3 = abs( ( np.sum(e_energy[:,2,-10:], axis = 1)/10 - e_energy[:,2,0])  / e_energy[:,2,0] ) *1e5

    axes[1].plot(asteroids, t_num[:,1] -t_num[:,0], color = color1[0], linewidth = 2, linestyle = '--')
    axes[1].plot(asteroids, t_num[:,2] -t_num[:,0], color = color1[0], linewidth = 2, linestyle = '-')
    axes[1].plot([], [], color = 'black', linestyle = '--', label = 'HNN')
    axes[1].plot([], [], color = 'black', linestyle = '-', label = 'WH-HNN ')
    axes[1].set_xlabel('Number of asteroids', fontsize = 27)
    axes[1].set_ylabel(r'$\rm tc - tc_{WH}$ (s)', fontsize = 27, color = color1[0])
    axes[1].set_xscale('log')
    axes[1].set_yscale('symlog', linthresh = 1)
    axes[1].tick_params(axis='both', which='major', labelsize=25)
    axes[1].tick_params(axis='both', which='minor', labelsize=25)
    axes[1].tick_params(axis='y', labelcolor = color1[0])

    ax2 = axes[1].twinx()
    ax2.plot(asteroids, e_rel2-e_rel, linestyle = '--', linewidth = 2, color = color1[2])
    ax2.plot(asteroids, e_rel3-e_rel, linestyle = '-', linewidth = 2, color = color1[2])
    ax2.set_ylabel(r'$\rm \varepsilon - \varepsilon_{WH} \; _{(\times10^{-5})}$ ', fontsize = 27, color = color1[2])
    ax2.tick_params(axis='y', which='major', labelsize=25)
    ax2.tick_params(axis='y', which='minor', labelsize=25)
    ax2.tick_params(axis='y', labelcolor = color1[2])
    axes[1].legend(fontsize = 22, loc = 'lower left', labelcolor = 'black')
    axes[1].grid(alpha = 0.5)

    # plt.suptitle("Time and Energy error \n $t_f$ = %0.3f and $h$ = %0.3f"%(t_end, h), fontsize = 18)
    plt.tight_layout()
    plt.savefig('./Experiments/AsteroidVsTime/timeVsError_%dyr.png' % t_end)
    plt.show()

def polifit():
    # Load data
    with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    t_num = data['time']
    e_energy = data['energy']
    h, t_end = data['settings']
    asteroids = data['asteroids']

    z1 = np.polyfit(asteroids, t_num[:, 1], 1)
    z2 = np.polyfit(asteroids, t_num[:, 1], 2)
    z3 = np.polyfit(asteroids, t_num[:, 1], 3)

    fig, axes = plt.subplots(1,2, figsize=(16,4), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.5, wspace= 0.55)

    axes[0].plot(asteroids, t_num[:,0], color = color[3],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[0].plot(asteroids, t_num[:,1], color = color[9],  linestyle=':', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axes[0].plot(asteroids, t_num[:,2], color = color[5],  linestyle='--', linewidth = 2, marker = '^', markersize = 12,label = 'WH-HNN')

    axes[0].plot(asteroids,  z1[0]*asteroids + z1[1]*asteroids**0, label = '1')
    axes[0].plot(asteroids, z2[0]*asteroids**2 + z2[1]*asteroids + z2[2]*asteroids**0, label = '2')
    axes[0].plot(asteroids, z3[0]*asteroids**3+ z3[1]*asteroids**2 + z3[2]*asteroids + z3[3]*asteroids**0, label = '3')
    axes[0].set_xscale('log')
    axes[0].set_yscale('symlog')
    axes[0].legend()
    plt.show()

def plot_errorPhaseOrbit(theta_JS, E_accel):
    E_accel_norm = np.linalg.norm(E_accel, axis = 1)
    plt.scatter(theta_JS[:, 0], theta_JS[:,1], c = E_accel_norm, cmap = 'viridis')
    plt.xlabel(r'$\theta_J$', fontsize = 22)
    plt.ylabel(r'$\theta_S$', fontsize = 22)
    plt.colorbar()
    plt.savefig('./Experiments/error_phase/phasemap.png')

    plt.show()
