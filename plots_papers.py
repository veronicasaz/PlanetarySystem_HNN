import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import h5py

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.formatter.useoffset'] = False

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


color1 = ['navy', 'dodgerblue','darkorange']
# color2 = ['dodgerblue', 'navy', 'orangered', 'green', 'olivedrab',  'saddlebrown', 'darkorange', 'red' ]
color2 = ['dodgerblue', 'navy', 'saddlebrown', 'darkorange', 'olivedrab']

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
    colors = [color[8], color[9], color[3], color[6], color[0], color[10], color[5], color[7], color[1], color[11]]
    line = ['-', '--', '-.', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
    lnwidth = 3
    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,0].set_title("Numerical integrator result: %.1f s"%(t_num), fontsize = titlesize)
        axes[0,1].set_title("Hamiltonian Neural Network result: %.1f s"%(t_nn), fontsize = titlesize)
        axes[0,2].set_title("Deep Neural Network result: %.1f s"%(t_dnn), fontsize = titlesize)

        # axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = colors[i], label=names[i])
        # axes[0,0].axis('equal')
        # axes[0,0].set_xlabel('$x$',fontsize = labelsize)
        # axes[0,0].set_ylabel('$y$',fontsize = labelsize)
        # # axes[0,0].legend(loc = 'upper right', fontsize = 16, framealpha = 0.9)
        # axes[0,0].legend(bbox_to_anchor=(-0.85, 1.05), loc = 'upper left', fontsize = 20, framealpha = 0.9)
        
        # axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = colors[i], label=names[i])
        # axes[1,0].axis('equal')
        # axes[1,0].set_ylabel('$y$',fontsize = labelsize)
        # axes[1,0].set_xlabel('$x$',fontsize = labelsize)
        # axes[2,0].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = colors[i], label=names[i])
        # axes[2,0].axis('equal')
        # axes[2,0].set_ylabel('$y$',fontsize = labelsize)
        # axes[2,0].set_xlabel('$x$',fontsize = labelsize)

        # eccentricity in the second column
        axes[0,0].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = colors[i], linewidth = lnwidth, label=names[i])
        # axes[0,0].legend(loc = 'center left', fontsize = 20, framealpha = 0.9, 
        axes[0, 0].legend(loc = 'lower left', fontsize = 22, \
                framealpha = 0.9, bbox_to_anchor=(0.5, 1.22, 2.7, 1.5),\
                ncol=5, mode="expand", borderaxespad=0.)
        # axes[0,0].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0,0].set_ylabel('$e$',fontsize = labelsize)
        axes[0,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = colors[i], linewidth = lnwidth, label=names[i])
        # axes[0,1].set_ylabel('$e$',fontsize = labelsize)
        # axes[0,1].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0,2].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = colors[i],linewidth = lnwidth,  label=names[i])
        # axes[0,2].set_ylabel('$e$',fontsize = labelsize)
        # axes[0,2].set_xlabel('$t$ [years]',fontsize = labelsize)
                
    # energy drift in the second row
    axes[1,0].set_xlabel('$t$ [years]',fontsize = labelsize)
    axes[1,0].plot(time, sim.energy, linestyle = '-',  color = color[9], alpha=1,label= 'Error with WH')
    axes[1,0].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1, 0].ticklabel_format(useOffset=False)
    axes[1, 0].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9)

    axes[1,1].plot(time, sim2.energy, linestyle = '-',  color = color[3],alpha=1, label= 'Error with WH-HNN')
    axes[1,1].plot(time, sim.energy, linestyle = '-',  color = color[9],alpha=1, label= 'Error with WH')
    axes[1,1].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9)
    # axes[1,1].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1,1].ticklabel_format(useOffset=False)
    axes[1,1].set_xlabel('$t$ [years]',fontsize = labelsize)

    axes[1,2].plot(time, sim3.energy, alpha=1, linestyle = '-',  color = color[3],label= 'Error with WH-DNN')
    axes[1,2].plot(time, sim.energy, alpha=1, linestyle = '-',  color = color[9], label= 'Error with WH')
    if typePlot == 'JS':
        axes[1,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)
    else:
        axes[1,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)
    # axes[1,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1,2].ticklabel_format(useOffset=False)
    axes[1,2].set_xlabel('$t$ [years]',fontsize = labelsize)

    for i in range(2):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.png' % t_end)
    plt.show()

def plot_NeurIPS_energyTogether(sim, sim2, sim3, t, asteroids, asteroids_extra, t_end, h, typePlot,):
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

    fig, axes = plt.subplots(1,4, figsize=(20,6))
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
    colors = [color[8], color[9], color[3], color[6], color[0], color[10], color[5], color[7], color[1], color[11]]
    line = ['-', '--', '-.', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
    lnwidth = 3
    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0].set_title("Numerical integrator result: %.1f s"%(t_num), fontsize = titlesize)
        axes[1].set_title("Hamiltonian Neural Network result: %.1f s"%(t_nn), fontsize = titlesize)
        axes[2].set_title("Deep Neural Network result: %.1f s"%(t_dnn), fontsize = titlesize)

        # axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = colors[i], label=names[i])
        # axes[0,0].axis('equal')
        # axes[0,0].set_xlabel('$x$',fontsize = labelsize)
        # axes[0,0].set_ylabel('$y$',fontsize = labelsize)
        # # axes[0,0].legend(loc = 'upper right', fontsize = 16, framealpha = 0.9)
        # axes[0,0].legend(bbox_to_anchor=(-0.85, 1.05), loc = 'upper left', fontsize = 20, framealpha = 0.9)
        
        # axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = colors[i], label=names[i])
        # axes[1,0].axis('equal')
        # axes[1,0].set_ylabel('$y$',fontsize = labelsize)
        # axes[1,0].set_xlabel('$x$',fontsize = labelsize)
        # axes[2,0].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = colors[i], label=names[i])
        # axes[2,0].axis('equal')
        # axes[2,0].set_ylabel('$y$',fontsize = labelsize)
        # axes[2,0].set_xlabel('$x$',fontsize = labelsize)

        # eccentricity in the second column
        axes[0].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = colors[i], linewidth = lnwidth, label=names[i])
        axes[0].legend(loc = 'center left', fontsize = 18, framealpha = 0.9, bbox_to_anchor=(-0.75, 0.5))
        # axes[0].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9, bbox_to_anchor=(0., 1.02, 1., .102))
        # axes[0,0].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0].set_ylabel('$e$',fontsize = labelsize)
        axes[1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = colors[i], linewidth = lnwidth, label=names[i])
        # axes[0,1].set_ylabel('$e$',fontsize = labelsize)
        # axes[0,1].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[2].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = colors[i],linewidth = lnwidth,  label=names[i])
        # axes[0,2].set_ylabel('$e$',fontsize = labelsize)
        # axes[0,2].set_xlabel('$t$ [years]',fontsize = labelsize)
                
        
    # energy drift in the second row
    axes[3].set_xlabel('$t$ [years]',fontsize = labelsize)
    axes[3].plot(time, sim.energy, linestyle = '-',  color = color[9], alpha=1)
    axes[3].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[3].ticklabel_format(useOffset=False)

    axes[3].plot(time, sim2.energy, linestyle = '-',  color = color[3],alpha=1, label= 'Error with WH-HNN')
    axes[3].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9)

    axes[3].plot(time, sim3.energy, alpha=1, linestyle = '-',  color = color[6],label= 'Error with WH-DNN')
    axes[3].set_yscale('log')

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
    # line = ['-', '--', '-.', ':', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
    line = ['-', '-',  '-', '-', '-','-', '-', '-', '-','-', '-', '-', '-']
    lnwidth = 3
    
    for i in range(1, data_nih['x'].shape[1]):
        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = color2[i-1], label=names[i])
        axes[0,1].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = color2[i-1], label=names[i])
        axes[0,2].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = color2[i-1], label=names[i])
        
        # eccentricity in the second column
        axes[1,0].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[1,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = color2[i-1], linewidth = lnwidth, label=names[i])
        axes[1,2].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = color2[i-1],linewidth = lnwidth,  label=names[i])            

    axes[0,0].set_title("Numerical integrator result", fontsize = titlesize)
    axes[0,1].set_title("Hamiltonian Neural Network result", fontsize = titlesize)
    axes[0,2].set_title("Deep Neural Network result", fontsize = titlesize)

    for col in range(3):
        axes[0,col].scatter(data_nb['x'][:,0], data_nb['y'][:,0], linestyle = line[0], s = 20, color = 'black', label=names[0])
        axes[0,col].axis('equal')
        axes[0,col].set_xlabel('$x$ (au)',fontsize = labelsize)
        axes[0,col].set_ylabel('$y$ (au)',fontsize = labelsize)
        
        axes[1,col].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[1,col].set_ylabel('$e$',fontsize = labelsize)

    axes[0,0].legend(loc = 'lower left', fontsize = 27, \
                framealpha = 0.9, bbox_to_anchor=(0.0, 1.22, 3.5, 1.5),\
                ncol=6, mode="expand", borderaxespad=0.)

    for i in range(2):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=25)
            axes[i,j].tick_params(axis='both', which='minor', labelsize=25)
            axes[i, j].locator_params(axis = 'x', nbins=6)
            axes[i, j].locator_params(axis = 'y', nbins=6)

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
    # line = ['-', '--', '-.', ':', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
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
    color_e = color2[2]
    color_e2 = color2[3]
    color_e3 = color2[4]
    lnwidth = 2

    axes[2,1].plot(time, np.array(sim2.energy)*1e3,  linestyle = '-',  linewidth = lnwidth,color = color_e2,alpha=1, label= 'Error with WH-HNN')
    axes[2,2].plot(time, np.array(sim3.energy) *1e3, alpha=1, linestyle = '-',linewidth = lnwidth, color = color_e3,label= 'Error with WH-DNN')
    # ax.get_xaxis().set_major_formatter('{x:1.2f}')

    axes[0,0].set_title("Numerical integrator result", fontsize = titlesize)
    axes[0,1].set_title("Hamiltonian Neural Network result", fontsize = titlesize)
    axes[0,2].set_title("Deep Neural Network result", fontsize = titlesize)
    
    axes[0,0].set_ylabel('$y$ (au)',fontsize = labelsize)
    axes[1,0].set_ylabel('$e$',fontsize = labelsize)  
    axes[2,0].set_ylabel(r'$dE/E_0 \;\times 10^3$',fontsize = labelsize)

    for col in range(3):
        axes[0,col].axis('equal')
        axes[0,col].set_xlabel('$x$ (au)',fontsize = labelsize)
    
        axes[1,col].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[1,col].get_yaxis().set_major_formatter('{x:1.3f}')

        axes[2,col].plot(time, np.array(sim.energy) *1e3 , linestyle = '-', linewidth = lnwidth,color = color_e, alpha=1, label = 'Error with WH')
        axes[2,col].set_xlabel('$t$ (yr)',fontsize = labelsize)
        axes[2,col].ticklabel_format(useOffset=False)
        
    axes[0,0].legend(loc = 'lower left', fontsize = 27, \
                framealpha = 0.9, bbox_to_anchor=(0.0, 1.24, 2.0, 1.5),\
                ncol=6, mode="expand", borderaxespad=0.)
    axes[2,0].legend(loc = 'lower left', fontsize = 23, framealpha = 0.9)
    axes[2,1].legend(loc = 'lower left', fontsize = 23, framealpha = 0.9)
    axes[2,2].legend(loc = 'upper left', fontsize = 23, framealpha = 0.8)

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

def plot_asteroids_NeurIPS():
    """
    plot_asteroids: plot asteroids vs time and asteroids vs energy error
    """
    # Load data
    with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    t_num = data['time']
    # t_num2 = data['time_accel']
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

    # ax2 = plt.axes([.65, .6, .2, .2])
    # ax2.plot(asteroids[-2:], t_num[-2:, 0])
    # plt.setp(ax2, xticks=[], yticks=[])
    axins = axes[0].inset_axes([0.72, 0.12, 0.25, 0.4])
    # sub region of the original image
    axins.plot(asteroids[-3:], t_num[-3:,0], color = color1[0],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axins.plot(asteroids[-3:], t_num[-3:,1], color = color1[1],  linestyle='-', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axins.plot(asteroids[-3:], t_num[-3:,2], color = color1[2],  linestyle='-', linewidth = 2, marker = '^', markersize = 12,label = 'WH-HNN')

    x1, x2, y1, y2 = (asteroids[-3]+asteroids[-2])/2, asteroids[-1]*1.1, 400, 7000
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.tick_params(axis='both', which='major', labelsize=18)
    axins.tick_params(axis='both', which='minor', labelsize=18)
    axins.set_yscale('linear')
    # axins.set_xticklabels(axes[0].get_xticks()[-4:])
    # axins.set_yticklabels(axes[0].get_yticks()[-4:])

    axes[0].indicate_inset_zoom(axins, edgecolor="black")


    # axins = zoomed_inset_axes(axes[0], zoom=0.001, loc='lower right')
    # # axins.xaxis.get_major_locator().set_params(nbins=7)
    # plt.setp(axins.get_xticklabels(), visible=True)
    # plt.setp(axins.get_yticklabels(), visible=True)


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

