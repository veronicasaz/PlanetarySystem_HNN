"""
Created: July 2021 
Last modified: July 2022 
Author: Veronica Saz Ulibarrena 
Description: Use neural networks in integration
"""
import rebound
import sys
import numpy as np
import time
import h5py

from wh_tf_flag import WisdomHolman
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

from data import load_json

from nn_tensorflow import  ANN

from plots_papers import plot_NeurIPS, plot_NeurIPS_energyTogether, \
                plot_asteroids_NeurIPS, polifit, plot_CompPhys_trajectory

import matplotlib
from test_dataset import trunc

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.formatter.useoffset'] = False

# Setup rebound 
# sr = rebound.Simulation()
# sr.units = {'yr', 'au', 'MSun'}
# sr.add('Sun')
# sr.add('Jupiter')
# sr.add('Saturn')
# sr.save('ic_sun_jupiter_saturn.bin')

color1 = ['navy', 'dodgerblue','darkorange']
color2 = ['dodgerblue', 'navy', 'orangered', 'green', 'olivedrab',  'saddlebrown', 'darkorange', 'red' ]

def load_model(multiple):
    """
    load_model: load network(s) needed
    INPUTS: 
        multiple: type of problem 
    Output:
        ANN_tf: netowork
        or 
        Nets: list of networks
    """
    settings_file_path= "./config_ANN.json"
    settings = load_json(settings_file_path)
    settings_dataset = load_json("./config_data.json")
    config = {**settings_dataset, **settings}

    config['output_dim'] = "H" # By default the output is H
    path_std = "./dataset/"

    if multiple == 'Asteroid_JS' or multiple == 'Asteroid_JS_energy': # Asteroid, Sun, Jupiter and Saturn require 2 nets
        path_model = "./ANN_tf/"
        Nets = []
        
        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS/')
        Nets.append(ANN_tf)

        config['bodies'] = 3 # Case with SJSa
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids/2_lossa/', path_std = path_std, std = True)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids/2_lossa/')
        Nets.append(ANN_tf)
        return Nets

    elif multiple == 'Asteroid_avgJS': # Asteroid, Sun, Jupiter and Saturn with 1 net
        path_model = "./ANN_tf/"
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids/')
        return ANN_tf
    else: # Sun, Jupiter and Saturn or other cases with 1 network
        path_model = "./ANN_tf/"
        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS/')
        return ANN_tf

def load_DNN(multiple):
    """
    load_model: load Deep Neural network(s) needed
    INPUTS: 
        multiple: type of problem 
    Output:
        ANN_tf: netowork
        or 
        Nets: list of networks
    """
    settings_file_path= "./config_ANN.json"
    settings = load_json(settings_file_path)
    settings_dataset = load_json("./config_data.json")
    config = {**settings_dataset, **settings}
    path_std = "./dataset/"

    if multiple == 'Asteroid_JS' or multiple == 'Asteroid_JS_energy':
        path_model = "./ANN_tf/"
        Nets = []
        config['output_dim'] = 'a'
        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS_DNN/', path_std = path_std, pred_type = 'a')
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS_DNN/')
        Nets.append(ANN_tf)
        
        config['bodies'] = 3 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids_DNN/', path_std = path_std, pred_type = 'a', std = True)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids_DNN/')
        Nets.append(ANN_tf)
        return Nets
    elif multiple == 'Asteroid_avgJS':
        path_model = "./ANN_tf/"
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids_DNN/', path_std = path_std, pred_type = 'a')
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids_DNN/')
        return ANN_tf
    else: # Sun, Jupiter and Saturn or other cases with 1 network
        path_model = "./ANN_tf/"
        config['output_dim'] = 'a'
        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS_DNN/', path_std = path_std, pred_type = 'a')
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS_DNN/')
        return ANN_tf
    
def calculate_centralH(data, m):
    """
    calculate_centralH: calculate energy due to central body, potential + kinetic
    INPUTS: 
        data: samples and positions/velocities
        m: masses
    OUTPUTS:
        H: energy 
    """
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

def add_particles(sim, asateroids, asteroids_extra):
    """
    add_particles: add particles to integrator
    INPUTS: 
        sim: initialized simulation
        asteroids: number of asteroids
        asteroids_extra: number of extrapolation asteroids
    OUTPUTS:
        sim: initialized simulation with particles
    """
    # Choose masses, semi-major axis, and true anomaly for the asteroids
    m_a = np.array([1e19, 2e19, 5e19, 1e20]) / 1.9891e30 # To mass of the sun
    a_a = np.linspace(2.2, 3.2, num= asteroids)
    f_a = np.linspace(0, 2*np.pi, num= asteroids)
    # np.random.shuffle(a_a)
    # np.random.shuffle(f_a)
    # a_a = [3.2] # for flag vs no flag

    m_a_e = np.array([1e18, 1e19, 5e19, 1e21]) / 1.9891e30
    a_a_e = [4.0, 1.0, 1.5] #AU
    

    for i, p in enumerate(sr.particles): # Planets
        print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
        sim.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))
    for j in range(asteroids):
        sim.particles.add(mass =m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0.0, f= f_a[j])
    for j in range(asteroids_extra):
        sim.particles.add(mass =m_a_e[j], a=a_a_e[j] , e=0.1, i=0.0, f= f_a[j])
    return sim

def simulate(t_end, h, asteroids, asteroids_extra, multiple, flag, name, R):
    """
    simulate: run integration using WH, HNN, and DNN
    INPUTS:
        t_end: final time of integration
        h: time-step
        asteroids: number of asteroids
        asteroids_extra: number of extrapolation asteroids
        multiple: type of simulation
        flag: True (verifying prediction of ANN), False (not verifying)
        name: name of file to save
    OUTPUTS:
        sim, sim2, sim3: simulations for WH, HNN and DNN
        [t_num, t_nn, t_dnn]: computation time for WH, HNN and DNN
    """
    ########################################################
    #### WH 
    ########################################################
    sim = WisdomHolman(CONST_G=sr.G, accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
    sim = add_particles(sim, asteroids, asteroids_extra)

    t0_num = time.time()
    sim.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nb'+name+'.hdf5'
    sim.integrator_warmup()
    sim.h = h
    sim.acceleration_method = 'numpy'
    sim.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
    sim.buf.recorder.start()
    sim.integrate(t_end, nih=False)
    sim.buf.flush()
    sim.stop()

    t_num = time.time() - t0_num
    sim.buf.recorder.data.keys()

    ########################################################
    #### WH + HNN
    ########################################################
    EXPERIMENT_DIR = '.'
    sys.path.append(EXPERIMENT_DIR)

    nih_model = load_model(multiple)

    sim2 = WisdomHolman(hnn=nih_model, CONST_G=sr.G, \
                        accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_ANN.txt",\
                        flag = flag, \
                        multiple_nets = multiple, R= R)
    sim2 = add_particles(sim2, asteroids, asteroids_extra)

    t0_nn = time.time()
    sim2.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nih'+name+'.hdf5'
    sim2.integrator_warmup()
    sim2.h = h
    sim2.acceleration_method = 'numpy'
    sim2.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
    sim2.buf.recorder.start()
    sim2.integrate(t_end, nih=True)
    sim2.buf.flush()
    sim2.stop()

    sim2.buf.recorder.data.keys()

    t_nn = time.time() - t0_nn

    print("Time WH: %.3f, Time ANN: %.3f"%(t_num, t_nn))

    ########################################################
    #### WH + DNN
    ########################################################
    nDNN_model = load_DNN(multiple)

    sim3 = WisdomHolman(hnn=nDNN_model, CONST_G=sr.G, \
                        accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_DNN.txt",\
                        flag = flag, \
                        multiple_nets = multiple, R = R)
    # sim.integrator = 'WisdomHolman'
    sim3 = add_particles(sim3, asteroids, asteroids_extra)

    t0_dnn = time.time()
    sim3.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_dnn'+name+'.hdf5'
    sim3.integrator_warmup()
    sim3.h = h
    sim3.acceleration_method = 'numpy'
    sim3.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
    sim3.buf.recorder.start()
    sim3.integrate(t_end, nih=True)
    sim3.buf.flush()
    sim3.stop()

    sim3.buf.recorder.data.keys()
    t_dnn = time.time() - t0_dnn

    np.savetxt("./Experiments/sun_jupiter_saturn/computationTime.txt", [t_num, t_nn, t_dnn])

    print("Time WH: %.3f, Time ANN: %.3f"%(t_num, t_nn))
    return sim, sim2, sim3, [t_num, t_nn, t_dnn]

def plot_general(sim, sim2, sim3, t, asteroids, asteroids_extra):
    """
    plot_general: plot simulation with WH, HNN and DNN
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

    fig, axes = plt.subplots(3,5, figsize=(18,18))
    data_nb = sim.buf.recorder.data
    data_nih = sim2.buf.recorder.data
    data_dnn = sim3.buf.recorder.data
    time = np.linspace(0, t_end, data_nih['x'].shape[0])

    # Add names of asteroids
    names = ['Sun', 'Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid%i"%(j+1))

    labelsize = 20
    titlesize = 22
    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,2].set_title("Numerical integrator result: %.3f s"%(t_num), fontsize = titlesize)
        axes[1,2].set_title("Hamiltonian Neural Network result: %.3f s"%(t_nn), fontsize = titlesize)
        axes[2,2].set_title("Deep Neural Network result: %.3f s"%(t_dnn), fontsize = titlesize)

        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[0,0].axis('equal')
        axes[0,0].set_xlabel('$x$',fontsize = labelsize)
        axes[0,0].set_ylabel('$y$',fontsize = labelsize)
        axes[0,0].legend(loc = 'upper right', fontsize = labelsize, framealpha = 0.9)
        axes[0,0].set_yticklabels(fontsize = 16)
        axes[0,0].set_xticklabels(fontsize = 16)

        axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[1,0].axis('equal')
        axes[1,0].set_ylabel('$y$',fontsize = labelsize)
        axes[1,0].set_xlabel('$x$',fontsize = labelsize)

        axes[2,0].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[2,0].axis('equal')
        axes[2,0].set_ylabel('$y$',fontsize = labelsize)
        axes[2,0].set_xlabel('$x$',fontsize = labelsize)

        # semi-major in the second column
        axes[0,1].plot(time, data_nb['a'][:,i], label=names[i], alpha=0.6)
        axes[0,1].set_xlabel('$t$ [years]',fontsize = labelsize )
        axes[0,1].set_ylabel('$a$ [au]',fontsize = labelsize)
        axes[1,1].plot(time, data_nih['a'][:,i], label=names[i], alpha=0.6)
        axes[1,1].set_xlabel('$t$ [years]',fontsize = labelsize )
        axes[1,1].set_ylabel('$a$ [au]',fontsize = labelsize)
        axes[2,1].plot(time, data_dnn['a'][:,i], label=names[i], alpha=0.6)
        axes[2,1].set_xlabel('$t$ [years]',fontsize = labelsize )
        axes[2,1].set_ylabel('$a$ [au]',fontsize = labelsize)
        
        # eccentricity in the second column
        axes[0,2].plot(time, data_nb['ecc'][:,i], label=names[i], alpha=0.6)
        axes[0,2].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0,2].set_ylabel('$e$',fontsize = labelsize)
        axes[1,2].plot(time, data_nih['ecc'][:,i], label=names[i], alpha=0.6)
        axes[1,2].set_ylabel('$e$',fontsize = labelsize)
        axes[1,2].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[2,2].plot(time, data_dnn['ecc'][:,i], label=names[i], alpha=0.6)
        axes[2,2].set_ylabel('$e$',fontsize = labelsize)
        axes[2,2].set_xlabel('$t$ [years]',fontsize = labelsize)
        
        # inclination in the second column
        axes[0,3].plot(time, np.degrees(data_nb['inc'][:,i]), label=names[i], alpha=0.6)
        axes[0,3].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0,3].set_ylabel('$i$ [deg]',fontsize = labelsize)
        axes[1,3].plot(time, np.degrees(data_nih['inc'][:,i]), label=names[i], alpha=0.6)
        axes[1,3].set_ylabel('$i$ [deg]',fontsize = labelsize)
        axes[1,3].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[2,3].plot(time, np.degrees(data_dnn['inc'][:,i]), label=names[i], alpha=0.6)
        axes[2,3].set_ylabel('$i$ [deg]',fontsize = labelsize)
        axes[2,3].set_xlabel('$t$ [years]',fontsize = labelsize)
        
    # energy drift in the second column
    axes[0,4].set_xlabel('$t$ [years]',fontsize = labelsize)
    axes[0,4].plot(time, sim.energy, alpha=0.6)
    axes[0,4].set_xlabel('$t$ [years]',fontsize = labelsize)
    axes[0,4].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[0, 4].ticklabel_format(useOffset=False)

    axes[1,4].plot(time, sim2.energy, alpha=0.6, label= 'Error with WH-HNN')
    axes[1,4].plot(time, sim.energy, alpha=0.2, color = 'red', label= 'Error with WH')
    axes[1,4].legend(loc = 'lower left', fontsize = 20)
    axes[1,4].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1, 4].ticklabel_format(useOffset=False)

    axes[2,4].plot(time, sim3.energy, alpha=0.6, label= 'Error with WH-DNN')
    axes[2,4].plot(time, sim.energy, alpha=0.2, color = 'red', label= 'Error with WH')
    axes[2,4].legend(loc = 'lower left', fontsize = 20)
    axes[2,4].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[2, 4].ticklabel_format(useOffset=False)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
    plt.show()

def plot_general_printversion(sim, sim2, sim3, t, asteroids, asteroids_extra, typePlot):
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

    fig, axes = plt.subplots(3,3, figsize=(18,10))
    data_nb = sim.buf.recorder.data
    data_nih = sim2.buf.recorder.data
    data_dnn = sim3.buf.recorder.data
    time = np.linspace(0, t_end, data_nih['x'].shape[0])

    # Add names of asteroids
    names = ['Sun', 'Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))

    labelsize = 22
    titlesize = 24
    line = ['-', '--', '-.', ':', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
    lnwidth = 3
    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,1].set_title("Numerical integrator result: %.1f s"%(t_num), fontsize = titlesize)
        axes[1,1].set_title("Hamiltonian Neural Network result: %.1f s"%(t_nn), fontsize = titlesize)
        axes[2,1].set_title("Deep Neural Network result: %.1f s"%(t_dnn), fontsize = titlesize)

        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = color2[i], label=names[i])
        axes[0,0].axis('equal')
        axes[0,0].set_xlabel('$x$',fontsize = labelsize)
        axes[0,0].set_ylabel('$y$',fontsize = labelsize)
        # axes[0,0].legend(loc = 'upper right', fontsize = 16, framealpha = 0.9)
        axes[0,0].legend(bbox_to_anchor=(-0.85, 1.05), loc = 'upper left', fontsize = 20, framealpha = 0.9)
        
        axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = color2[i], label=names[i])
        axes[1,0].axis('equal')
        axes[1,0].set_ylabel('$y$',fontsize = labelsize)
        axes[1,0].set_xlabel('$x$',fontsize = labelsize)
        axes[2,0].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = color2[i], label=names[i])
        axes[2,0].axis('equal')
        axes[2,0].set_ylabel('$y$',fontsize = labelsize)
        axes[2,0].set_xlabel('$x$',fontsize = labelsize)

        # eccentricity in the second column
        axes[0,1].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = color2[i], linewidth = lnwidth, label=names[i])
        axes[0,1].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0,1].set_ylabel('$e$',fontsize = labelsize)
        axes[1,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = color2[i], linewidth = lnwidth, label=names[i])
        axes[1,1].set_ylabel('$e$',fontsize = labelsize)
        axes[1,1].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[2,1].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = color2[i],linewidth = lnwidth,  label=names[i])
        axes[2,1].set_ylabel('$e$',fontsize = labelsize)
        axes[2,1].set_xlabel('$t$ [years]',fontsize = labelsize)
                
    # energy drift in the second column
    axes[0,2].set_xlabel('$t$ [years]',fontsize = labelsize)
    axes[0,2].plot(time, sim.energy, linestyle = '-',  color = color2[5], alpha=1, label = 'Error with WH')
    axes[0,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[0, 2].ticklabel_format(useOffset=False)
    axes[0, 2].legend(loc = 'lower left')

    axes[1,2].plot(time, sim2.energy, linestyle = '-',  color = color2[6],alpha=1, label= 'Error with WH-HNN')
    axes[1,2].plot(time, sim.energy, linestyle = '-',  color = color2[5],alpha=1, label= 'Error with WH')
    axes[1,2].legend(loc = 'lower left', fontsize = 18, framealpha = 0.9)
    axes[1,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1,2].ticklabel_format(useOffset=False)
    axes[1,2].set_xlabel('$t$ [years]',fontsize = labelsize)

    axes[2,2].plot(time, sim3.energy, alpha=1, linestyle = '-',  color = color2[7],label= 'Error with WH-DNN')
    axes[2,2].plot(time, sim.energy, alpha=1, linestyle = '-',  color = color2[5], label= 'Error with WH')
    # if typePlot == 'JS':
    axes[2,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)
    # else:
    #     axes[2,2].legend(loc = 'lower right', fontsize = 18, framealpha = 0.9)
    axes[2,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[2,2].ticklabel_format(useOffset=False)
    axes[2,2].set_xlabel('$t$ [years]',fontsize = labelsize)

    for i in range(3):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=17)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.png' % t_end)
    plt.show()

def plot_accelerations(sim, sim2, sim3, typenet = "Asteroid_JS"):
    """
    plot_accelerations: plot accelerations predicted with WH, HNN and DNN
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN and DNN
    """
    ############################################################################################
    # Figure accelerations
    accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
    accelerations_ANN = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
    accelerations_DNN = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")
    flags = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txtflag_list.txt")
    flags_DNN = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txtflag_list.txt")

    x = np.arange(0, len(accelerations_ANN), 1)

    names = ['Sun', 'Jupiter', 'Saturn']
    for j in range(3):
        names.append("Asteroid%i"%(j+1))

    asteroids_plot = 3
    fig, axes = plt.subplots(3,2+asteroids_plot, figsize=(18,6))
    for plot in range(1,3+asteroids_plot):
        a_WH = np.zeros(len(accelerations_ANN))
        a_ANN = np.zeros(len(accelerations_ANN))
        a_DNN = np.zeros(len(accelerations_ANN))
        
        for item in range(len(accelerations_ANN)):
            a_WH[item] = np.linalg.norm(accelerations_WH[item, plot*3:plot*3+3])
            a_ANN[item] = np.linalg.norm(accelerations_ANN[item, plot*3:plot*3+3])
            a_DNN[item] = np.linalg.norm(accelerations_DNN[item, plot*3:plot*3+3])
        
        axes[0, plot-1].plot(x, a_WH, color = 'blue', label = 'WH')
        axes[1, plot-1].plot(x, a_ANN, color = 'red', label = 'HNN')
        axes[2, plot-1].plot(x, a_DNN, color = 'green', label = 'DNN')
        
        if typenet == "SJSa":
            index = np.where(flags[:, plot] == 0)[0]
            x2 = np.copy(x)
            x2 = np.delete(x2, index)
            axes[1, plot-1].scatter(x2, np.delete(a_ANN, index), color = 'red', label = 'Numerically')
            index_DNN = np.where(flags_DNN[:, plot] == 0)[0]
            x2_DNN = np.copy(x)
            x2_DNN = np.delete(x2_DNN, index_DNN)
            axes[2, plot-1].scatter(x2_DNN, np.delete(a_DNN, index_DNN), color = 'green', label = 'Numerically')
        else:
            index = np.where(flags == 0)[0]
            x2 = np.copy(x)
            x2 = np.delete(x2, index)
            axes[1, plot-1].scatter(x2, np.delete(a_ANN, index), color = 'red', label = 'Numerically')
            index_DNN = np.where(flags_DNN == 0)[0]
            x2_DNN = np.copy(x)
            x2_DNN = np.delete(x2_DNN, index_DNN)
            axes[2, plot-1].scatter(x2_DNN, np.delete(a_DNN, index_DNN), color = 'green', label = 'Numerically')
        

        axes[0,plot-1].set_xlabel('$t$ [years]')
        axes[0,plot-1].set_ylabel('a')
        axes[0,plot-1].grid(alpha = 0.5)
        axes[0,plot-1].set_title(names[plot])

        axes[1,plot-1].set_xlabel('$t$ [years]')
        axes[1,plot-1].set_ylabel('a')
        axes[1,plot-1].legend(loc = 'upper right')
        axes[1,plot-1].grid(alpha = 0.5)
        axes[1,plot-1].set_title(names[plot])
        axes[1,plot-1].legend()

        axes[1,0].annotate("Flags: %i / %i"%(np.count_nonzero(flags), len(accelerations_ANN)), xy =  (x[-1]/100, max((a_ANN))))
        
        axes[2,plot-1].set_xlabel('$t$ [years]')
        axes[2,plot-1].set_ylabel('a')
        axes[2,plot-1].legend(loc = 'upper right')
        axes[2,plot-1].grid(alpha = 0.5)
        axes[2,plot-1].set_title(names[plot])
        axes[2,0].annotate("Flags: %i / %i"%(np.count_nonzero(flags_DNN), len(accelerations_DNN)), xy =  (x[-1]/100, max((a_DNN))))
        axes[2,plot-1].legend()

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_accel_%dyr.pdf' % t_end)
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

    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,2].set_title("Numerical integrator result: %.3f s"%(t_num))
        axes[1,2].set_title("Neural Network result without flag: %.3f s"%(t_nn))
        axes[2,2].set_title("Neural Network result with flag: %.3f s"%(t_nn2))

        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[0,0].axis('equal')
        axes[0,0].set_ylabel('$y$')
        axes[0,0].legend(loc = 'upper right')
        axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[1,0].axis('equal')
        axes[1,0].set_ylabel('$y$')
        axes[1,0].legend(loc = 'upper right')
        axes[2,0].plot(data_nih2['x'][:,i], data_nih2['y'][:,i], '-', label=names[i], alpha=0.6)
        axes[2,0].axis('equal')
        axes[2,0].set_ylabel('$y$')
        axes[2,0].legend(loc = 'upper right')
        axes[2,0].set_xlabel('$x$')

        # semi-major in the second column
        axes[0,1].plot(time, data_nb['a'][:,i], label=names[i], alpha=0.6)
        axes[0,1].set_ylabel('$a$ [au]')
        axes[1,1].plot(time, data_nih['a'][:,i], label=names[i], alpha=0.6)
        axes[1,1].set_ylabel('$a$ [au]')
        axes[2,1].plot(time, data_nih2['a'][:,i], label=names[i], alpha=0.6)
        axes[2,1].set_ylabel('$a$ [au]')
        axes[2,1].set_xlabel('$t$ [years]')
        
        # eccentricity in the second column
        axes[0,2].plot(time, data_nb['ecc'][:,i], label=names[i], alpha=0.6)
        axes[0,2].set_ylabel('$e$')
        axes[1,2].plot(time, data_nih['ecc'][:,i], label=names[i], alpha=0.6)
        axes[1,2].set_ylabel('$e$')
        axes[2,2].plot(time, data_nih2['ecc'][:,i], label=names[i], alpha=0.6)
        axes[2,2].set_ylabel('$e$')
        axes[2,2].set_xlabel('$t$ [years]')
        
        axes[0,3].plot(time, np.degrees(data_nb['inc'][:,i]), label=names[i], alpha=0.6)
        axes[0,3].set_ylabel('$i$ [deg]')
        axes[1,3].plot(time, np.degrees(data_nih['inc'][:,i]), label=names[i], alpha=0.6)
        axes[1,3].set_ylabel('$i$ [deg]')
        axes[2,3].plot(time, np.degrees(data_nih2['inc'][:,i]), label=names[i], alpha=0.6)
        axes[2,3].set_ylabel('$i$ [deg]')
        axes[2,3].set_xlabel('$t$ [years]')
        
    # energy drift in the second column
    axes[0,4].plot(time, sim.energy, alpha=0.6)
    axes[0,4].set_ylabel('$dE/E_0$')
    axes[1,4].plot(time, sim2.energy, alpha=0.6)
    axes[1,4].set_ylabel('$dE/E_0$')
    axes[2,4].plot(time, sim3.energy, alpha=0.6)
    axes[2,4].set_ylabel('$dE/E_0$')
    axes[2,4].set_xlabel('$t$ [years]')
        
    plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_flagcomparison%dyr.pdf' % t_end)
    plt.show()

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def plot_accel_flagvsnoflag(accelerations_WH, accelerations_ANN_noflag, accelerations_ANN_flag, 
                flags, t, asteroids, asteroids_extra):
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
    # x2 = np.copy(x)

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
        
        axes[2,plot-2].set_xlabel('$t$ (yr)', fontsize = 28)
        axes[2,plot-2].annotate("Flags: %i / %i"%(np.count_nonzero(flags[:, plot]), len(accelerations_ANN_flag)), \
                xy =  (x[0]+0.9, max((a_DNN))*0.9), fontsize = 20)
        
        for iterate in range(3):
            axes[iterate,plot-2].grid(alpha = 0.5)        
            ticks = -np.log10(axes[iterate,plot-2].get_yticks())
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
            axes[iterate,plot-2].set_yticklabels(np.round(trunc(axes[iterate,plot-2].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 20)
            axes[iterate,plot-2].set_xticklabels(np.round(trunc(axes[iterate,plot-2].get_xticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 20)
            axes[iterate,0].set_ylabel('a ($au/yr^2$)', fontsize = 28)
            # axes[iterate,0].legend(loc = 'upper left', framealpha = 0.5, fontsize = 24)
    
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
    # x2 = np.copy(x)

    asteroids_plot = asteroids+asteroids_extra

    
    # fig, axes = plt.subplots(1,3, figsize=(15,6))
    # fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.4, wspace= 0.25)
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
        # axes[plot].plot(x, a_baseline, color = color1[1], linewidth = 3, linestyle = ':', label = 'Numerical result')
        axes[plot].plot(x, accel[plot], color = color1[0], label = 'WH-HNN')
        flags = flags_i[plot]
        index_DNN = np.where(flags[:, 3] == 0)[0]
        x2_DNN = np.copy(x)
        x2_DNN = np.delete(x2_DNN, index_DNN)
        axes[plot].scatter(x2_DNN, np.delete(accel[plot], index_DNN), color = color1[2], label = 'Flags')
        # axes[plot].set_title(names[plot-1], fontsize = 24)
    
        axes[plot].set_title( 'R = %0.1f'%R[plot], fontsize = 24)
        axes[plot].set_xlabel('$t$ (yr)', fontsize = 25)
        axes[plot].annotate("Flags: %i / %i"%(np.count_nonzero(flags[:, 3]), len(accelerations_ANN_flag)), \
            xy =  (x[len(x)//2]-5, max((accel[plot]))*0.92), fontsize = 20)
    
        axes[plot].grid(alpha = 0.5)        
        ticks = -np.log10(axes[plot].get_yticks())
        dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
        axes[plot].set_yticklabels(np.round(trunc(axes[plot].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 20)
        axes[plot].set_xticklabels(np.round(trunc(axes[plot].get_xticks(), decs = 2), decimals = dec+1), rotation = 0, fontsize = 20)
        
        # axes[plot].legend(loc = 'upper left', fontsize = 16)

    axes[0].set_ylabel('a ($au/yr^2$)', fontsize = 25)
    lgnd = axes[0].legend(loc = 'lower left', fontsize = 23, \
                framealpha = 0.9, bbox_to_anchor=(0.0, 1.2, 1.0, 1.0),\
                ncol=4, mode="expand", borderaxespad=0., handletextpad=0.3)

    axes[3].plot(R_i, t_i, marker = 'o', color = color1[0])
    axes[3].set_xlabel('R', fontsize = 25)
    axes[3].set_ylabel('Comput. time (s)', fontsize = 25)
    axes[3].grid(alpha = 0.5)        
    ticks = -np.log10(axes[3].get_yticks())
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes[3].set_yticklabels(np.round(trunc(axes[3].get_yticks(), decs = 6), decimals = dec+3), rotation = 0, fontsize = 20)
    axes[3].set_xticklabels(np.round(trunc(axes[3].get_xticks(), decs = 6), decimals = dec+4), rotation = 0, fontsize = 20)

    # plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_accel_%dyr_flagR.png' % t_end, bbox_inches='tight')
    plt.show()


def load_model_asteroids():
    """
    load_model_asateroids: load network(s) needed for study of number of asteroids
    """
    settings_file_path= "./config_ANN.json"
    settings = load_json(settings_file_path)
    settings_dataset = load_json("./config_data.json")
    config = {**settings_dataset, **settings}

    path_std = "./dataset/"

    if multiple == 'Asteroid_JS':
        path_model = "./ANN_tf/"
        Nets = []

        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS/')
        Nets.append(ANN_tf)

        config['bodies'] = 3 # Case with SJSa
        # ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids_DNN/', path_std = path_std, std = True)
        # ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids_DNN/')
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids/2_lossa/', path_std = path_std, std = True)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids/2_lossa/')
        Nets.append(ANN_tf)
        return Nets

    elif multiple != False:
        path_model = "./ANN_tf/trained_nets/multiple/"
        Nets = []
        for net in range(multiple):
            ANN_tf = ANN(config, path_model = path_model+str(net+1)+'/', path_std = path_std)
            ANN_tf.load_model_fromFile(path_model+str(net+1)+'/')
            Nets.append(ANN_tf)
        return Nets
    else:
        path_model = "./ANN_tf/"
        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS/')
        return ANN_tf

def run_asteroids(R):
    """
    run_asteroids: simulate with different numbers of asteroids
    """
    # sr = rebound.Simulation()
    # sr.units = {'yr', 'au', 'MSun'}

    # sr.add('Sun')
    # sr.add('Jupiter')
    # sr.add('Saturn')
    # sr.save('ic_sun_jupiter_saturn.bin')

    multiple = 'Asteroid_JS' # False: one network for all, Asteroid_JS: one for asteroids, one for JS, 1,2, 3... number to average
    nih_model = load_model_asteroids()

    t_end = 20
    h = 1.0e-1

    savedata = dict()

    # asteroids = [5, 10, 20, 30, 50, 70, 100, 150, 200, 500, 1000, 2000]
    asteroids = [5, 10, 20, 30, 50, 70, 100, 200, 500, 1000]
    # asteroids = [5, 10, 20, 30, 50, 70, 100, 200]
    # asteroids = [5, 10, 20, 30, 50, 70, 90, 100, 150]
    # asteroids = [40, 50, 60, 100, 150, 200, 300]
    # asteroids = [5, 10, 20, 50, 60, 80, 100]
    t_num = np.zeros((len(asteroids), 3))
    t_num2 = np.zeros((len(asteroids), 3))
    e_energy = np.zeros((len(asteroids), 3, int(t_end//h)+1))
    
    for test in range(len(asteroids)):
        m_a = np.random.uniform(low = 1e19, high = 1e20, size = (asteroids[test],)) /1.9891e30 # To mass of the sun
        a_a = np.linspace(1.6, 4.0, num= asteroids[test])
        f_a = np.linspace(0, 2*np.pi, num= asteroids[test])
        # np.random.shuffle(m_a)
        np.random.shuffle(a_a)
        np.random.shuffle(f_a)

        ########################################################
        #### WH 
        ########################################################
        sim = WisdomHolman(CONST_G=sr.G, accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_WH.txt")

        for i, p in enumerate(sr.particles):
            print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
            sim.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))

        for j in range(asteroids[test]):
            sim.particles.add(mass =m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0, f= f_a[j])

        sim.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nb.hdf5'
        sim.integrator_warmup()
        sim.h = h
        sim.acceleration_method = 'numpy'
        sim.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
        sim.buf.recorder.start()
        t0_num = time.process_time()
        sim.integrate(t_end, nih=False)
        t_num[test, 0] = time.process_time() - t0_num
        sim.buf.flush()
        e_energy[test, 0,:] = sim.energy

        sim.stop()

        sim.buf.recorder.data.keys()

        ########################################################
        #### WH + HNN false flag
        ########################################################
        EXPERIMENT_DIR = '.'
        sys.path.append(EXPERIMENT_DIR)

        sim2 = WisdomHolman(hnn=nih_model, CONST_G=sr.G, \
                            accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_ANN.txt",\
                            flag = False, \
                            multiple_nets = multiple, R= R)
        # sim.integrator = 'WisdomHolman'

        for i, p in enumerate(sr.particles):
            print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
            sim2.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))

        for j in range(asteroids[test]):
            sim2.particles.add(mass = m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0, f= f_a[j])


        print("Flag", sim2.flag)
        sim2.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nih.hdf5'
        sim2.integrator_warmup()
        sim2.h = h
        sim2.acceleration_method = 'numpy'
        sim2.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
        sim2.buf.recorder.start()
        t0_nn = time.process_time()
        sim2.integrate(t_end, nih=True)
        t_num[test, 1] = time.process_time() - t0_nn
        sim2.buf.flush()
        print("n flags", np.count_nonzero(sim2.flag_list))
        e_energy[test, 1, :] = sim2.energy
        sim2.stop()


        ########################################################
        #### WH + HNN true flag
        ########################################################
        sim3 = WisdomHolman(hnn=nih_model, CONST_G=sr.G, \
                            accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_ANN.txt",\
                            flag = True, \
                            multiple_nets = multiple, R= R)

        for i, p in enumerate(sr.particles):
            print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
            sim3.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))

        for j in range(asteroids[test]):
            sim3.particles.add(mass = m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0, f= f_a[j])

        print("Flag", sim3.flag)
        sim3.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nih.hdf5'
        sim3.integrator_warmup()
        sim3.h = h
        sim3.acceleration_method = 'numpy'
        sim3.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
        sim3.buf.recorder.start()
        t0_nn = time.process_time()
        sim3.integrate(t_end, nih=True)
        t_num[test, 2] = time.process_time() - t0_nn
        sim3.buf.flush()
        print("n flags", np.count_nonzero(sim3.flag_list))
        e_energy[test, 2, :] = sim3.energy
        sim3.stop()

        t_num2[test,:] = np.array([sim.T_total, sim2.T_total, sim3.T_total])
        savedata['time'] = t_num
        savedata['time_accel'] = t_num2
        savedata['energy'] = e_energy
        savedata['settings'] = [h, t_end]
        savedata['asteroids'] = asteroids

        with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'w') as h5f:
            for dset in savedata.keys():
                h5f.create_dataset(dset, data=savedata[dset], compression="gzip")

        print("process", sim.T_total, sim2.T_total, sim3.T_total)
        

def plot_asteroids():
    """
    plot_asteroids: plot asteroids vs time and asteroids vs energy error
    """
    # Load data
    with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    # with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy_2.h5", 'r') as h5f:
    #     data2 = {}
    #     for dset in h5f.keys():
    #         data2[dset] = h5f[dset][()]

    # t_num = np.vstack((data['time'], data2['time']))
    # e_energy = np.vstack((data['energy'], data2['energy']))
    # # h, t_end = np.vstack((data['settings'], data2['settings']), axis = 0)
    # asteroids = np.append(data['asteroids'], data2['asteroids'] )

    t_num = data['time']
    e_energy = data['energy']
    h, t_end = data['settings']
    asteroids = data['asteroids']

    ########################################################
    #### Plots together
    ########################################################
    fig, axes = plt.subplots(1,2, figsize=(16,4), gridspec_kw={'width_ratios': [1.7, 1]})
    fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.5, wspace= 0.55)

    axes[0].plot(asteroids, t_num[:,0], color = color[3],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[0].plot(asteroids, t_num[:,1], color = color[9],  linestyle='-', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axes[0].plot(asteroids, t_num[:,2], color = color[5],  linestyle='--', linewidth = 2, marker = '^', markersize = 10,label = 'WH-HNN')
    
    # t_1 = t_num[7, 0] / ((asteroids[7]+2) * np.log(asteroids[7]+2))  # First time is for 5 asteroids (+2 planets). time per operation
    # axes[0].plot(asteroids, t_1 *((asteroids+2) * np.log(asteroids+2)) , color = 'blue', linewidth = 3, alpha = 0.5, linestyle = '--', label = 'N log(N)' )
    # t_2 = t_num[7, 0] / (asteroids[7]+2)**2 
    # axes[0].plot(asteroids, t_2 *((asteroids+2)**2) , color = 'black', linewidth = 3, alpha = 0.5, linestyle = '--', label = r'$N^2$' )
    axes[0].set_xlabel('Number of asteroids', fontsize = 22)
    axes[0].set_ylabel('Computation time (s)', fontsize = 22)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize = 22)
    axes[0].grid(alpha = 0.5)

    e_rel = abs( ( np.sum(e_energy[:,0,-10:], axis = 1)/10 - e_energy[:,0,0])  / e_energy[:,0,0] )
    e_rel2 = abs( ( np.sum(e_energy[:,1,-10:], axis = 1)/10 - e_energy[:,1,0])  / e_energy[:,1,0] )
    e_rel3 = abs( ( np.sum(e_energy[:,2,-10:], axis = 1)/10 - e_energy[:,2,0])  / e_energy[:,2,0] )
    axes[1].plot(asteroids, e_rel, color = color[3], linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[1].plot(asteroids, e_rel2, color = color[9],  linestyle='-',  linewidth = 2, marker = 'x', markersize = 10,label = 'HNN')
    axes[1].plot(asteroids, e_rel3, color = color[5], linestyle='--',   linewidth = 2, marker = '^', markersize = 10,label = 'WH-HNN')
    # axes[1].plot(asteroids, abs((e_energy[:,0,-1]-e_energy[:,0,0])/ e_energy[:,0,0]), color = color[3], linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    # axes[1].plot(asteroids, abs((e_energy[:,1,-1]-e_energy[:,0,0])/ e_energy[:,0,0]), color = color[9],  linestyle='-',  linewidth = 2, marker = 'x', markersize = 10,label = 'HNN')
    # axes[1].plot(asteroids, abs((e_energy[:,2,-1]-e_energy[:,0,0])/ e_energy[:,0,0]), color = color[5], linestyle='--',   linewidth = 2, marker = '^', markersize = 10,label = 'WH-HNN')
    # axes[1].plot(asteroids, e_energy[:,0,-1], color = color[3], linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    # axes[1].plot(asteroids, e_energy[:,1,-1], color = color[9],  linestyle='-',  linewidth = 2, marker = 'x', markersize = 10,label = 'HNN')
    # axes[1].plot(asteroids, e_energy[:,2,-1], color = color[5], linestyle='--',   linewidth = 2, marker = '^', markersize = 10,label = 'WH-HNN')
    axes[1].set_xlabel('Number of asteroids', fontsize = 22)
    # axes[1].set_ylabel('Relative mean energy error of last 10 steps', fontsize = 22)
    axes[1].set_ylabel('Relative final energy error', fontsize = 22)
    axes[1].set_yscale('symlog', linthresh = 0.000004)
    axes[1].set_xscale('log')
    axes[1].legend(fontsize = 22)
    axes[1].grid(alpha = 0.5)

    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[0].tick_params(axis='both', which='minor', labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    axes[1].tick_params(axis='both', which='minor', labelsize=20)

    # plt.suptitle("Time and Energy error \n $t_f$ = %0.3f and $h$ = %0.3f"%(t_end, h), fontsize = 18)
    plt.tight_layout()
    plt.savefig('./Experiments/AsteroidVsTime/timeVsError_%dyr.png' % t_end)
    plt.show()

def plot_accel_flagvsR_energy(accel_i, energy_i, time_i, flags_i, asteroids, asteroids_extra, flags_R):
    """
    plot_accelerations_flagvsnoflag: plot accelerations predicted with WH, HNN flag and HNN no flag
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN flag and HNN no flag
    """
    # Add names of asteroids
    names = ['Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))

    h = 1e-1
    # x2 = np.copy(x)

    colors = [color[8], color[9], color[3], color[6], color[0], color[10], color[5], color[7], color[1], color[11]]
    asteroids_plot = asteroids+asteroids_extra
    
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    fig.subplots_adjust(top=0.9,left = 0.09, right = 0.98, hspace = 0.4, wspace= 0.25)
        
    for plot in range(len(energy_i)):
        n_flags = np.count_nonzero(flags_i[plot][3, :])
        axes[0].scatter(n_flags, abs((energy_i[plot][-1]-energy_i[plot][0])/energy_i[plot][0]), color = color[plot], label = 'R = %0.1f'%flags_R[plot])
        axes[1].scatter(n_flags, time_i[plot][1], color = color[plot], label = 'R = %0.1f'%flags_R[plot])
    
    axes[0].set_xlabel('Flags', fontsize = 22)
    axes[0].set_ylabel('Relative energy error', fontsize = 22)    
    axes[0].grid(alpha = 0.5)        
    ticks = -np.log10(axes[0].get_yticks())
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes[0].set_yticklabels(np.round(trunc(axes[0].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 16)
    axes[0].set_xticklabels(np.round(trunc(axes[0].get_xticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 16)
    axes[0].legend(loc = 'upper left', framealpha = 0.5, fontsize = 16)

    axes[1].set_xlabel('Flags', fontsize = 22)
    axes[1].set_ylabel('Computation time (s)', fontsize = 22)    
    axes[1].grid(alpha = 0.5)        
    ticks = -np.log10(axes[1].get_yticks())
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes[1].set_yticklabels(np.round(trunc(axes[1].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 16)
    axes[1].set_xticklabels(np.round(trunc(axes[1].get_xticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 16)
    # axes[1].legend(loc = 'upper left', framealpha = 0.5, fontsize = 16)

    # plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_accel_%dyr_flagR_energy.png' % t_end)
    plt.show()    

def plot_accel_flagvsR_energy2(accel_i, energy_i, time_i, flags_i, asteroids, asteroids_extra, flags_R):
    """
    plot_accelerations_flagvsnoflag: plot accelerations predicted with WH, HNN flag and HNN no flag
    INPUTS:
        sim, sim2, sim3: simulations for WH, HNN flag and HNN no flag
    """
    # Add names of asteroids
    names = ['Jupiter', 'Saturn']
    for j in range(asteroids):
        names.append("Asteroid %i"%(j+1))

    h = 1e-1
    # x2 = np.copy(x)

    colors = [color[8], color[9], color[3], color[6], color[0], color[10], color[5], color[7], color[1], color[11]]
    asteroids_plot = asteroids+asteroids_extra
    
    fig, axes = plt.subplots(1,1, figsize=(10,4))
    fig.subplots_adjust(top=0.9,left = 0.12, right = 0.98, hspace = 0.4, wspace= 0.25)
        
    n_flags = np.zeros(len(time_i))
    time = np.zeros(len(time_i))
    e_error = np.zeros(len(time_i))
    for i in range(len(time_i)):
        n_flags[i] = np.count_nonzero(flags_i[i])
        time[i] = time_i[i][1] # Choose HNN
        e_error[i] = (np.mean(energy_i[i][-10:])-np.mean(energy_i[-1][-10]) - energy_i[i][0])
    # e_error -= e_error[-1]
    print(e_error)

    sc = axes.scatter(time, e_error, c = n_flags, s = 50)
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel('Number of flags', rotation=90, size = 24)
    cbar.ax.tick_params(labelsize=16)
    
    axes.set_xlabel('Computation time (s)', fontsize = 22)
    axes.set_ylabel(r'Relative energy error $\times 10^{6}$', fontsize = 22)    
    axes.grid(alpha = 0.5)        
    
    # ticks = -np.log10(axes.get_yticks())
    # dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )+1
    # axes.set_yticklabels(np.round(trunc(axes.get_yticks(), decs = 8), decimals = dec+1), rotation = 0, fontsize = 16)
    
    ticks = -np.log10(axes.get_xticks())
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes.set_xticklabels(np.round(trunc(axes.get_xticks(), decs = 2), decimals = dec+1), rotation = 0, fontsize = 16)
    # axes.legend(loc = 'upper left', framealpha = 0.5, fontsize = 16)


    # plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_accel_%dyr_flagR_energy.png' % t_end)
    plt.show()    

def plot_energyvsH(sim, sim2):
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
        # E_1[i] = sim.energy[i]- calculate_centralH(data, sim.particles.masses)
        E_1[i] = calculate_interH(data, sim.particles.masses)
        data[:,0:3] = data_nih[i][0:3*3].reshape((3,3))
        data[:,3:] = data_nih[i][3*3:].reshape((3,3))
        # E_2[i] = sim2.energy[i] - calculate_centralH(data, sim2.particles.masses)
        E_2[i] = calculate_interH(data, sim2.particles.masses)

    lw = 3
    fig, axes = plt.subplots(1,1, figsize=(8,6))
    axes.plot(time, E_1,  linewidth = lw, color = color1[0], label = 'WH Energy')
    axes.plot(time, E_2, ':',  linewidth = lw, color = color1[2], label = 'WH-HNN Energy')
    axes.plot(time, np.array(sim2.H), linewidth = lw, color = color1[1], label = 'WH-HNN H')
    # plt.title('Comparison of interactive Hamiltonian \n and the predicted output of the HNN', fontsize = 13)
    plt.xlabel('Time ($yr$)', fontsize =30)
    plt.ylabel(r'Energy $\times 10^6$ (${kg}\; {au}^2\; yr^{-2}$)', fontsize =30)

    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    axes.ticklabel_format(useOffset=False, style='plain')
    # axes.set_yticklabels(axes.get_yticks()*1e6, fontsize = 16)
    ticks = -np.log10(abs(axes.get_yticks()))+1
    dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
    axes.set_yticklabels(np.round(trunc(axes.get_yticks()*1e6, decs = 6), decimals = 2), fontsize = 27)
    
    plt.legend(fontsize = 25)
    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/HvsE_%dyr.png' % t_end)
    plt.show()

def compute_predError(t_end, h, asteroids, asteroids_extra):
    """
    compute_predError
    """
    # Run numerically
    sim = WisdomHolman(CONST_G=sr.G, accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
    sim = add_particles(sim, asteroids, asteroids_extra)

    sim.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nb'+'_phaseError'+'.hdf5'
    sim.integrator_warmup()
    sim.h = h
    sim.acceleration_method = 'numpy'
    sim.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
    sim.buf.recorder.start()
    sim.integrate(t_end, nih=False)
    sim.buf.flush()
    sim.stop()

    sim.buf.recorder.data.keys()

    hnn = load_model(multiple)

    # For each time-step, predict accelerations
    iters = np.shape(sim.coord)[0]
    accel_pred = np.zeros((iters, 9))
    masses = sim.particles.masses
    nbodies = len(masses)

    input_JS = np.zeros((iters, 8))
    theta_JS = np.zeros((iters, 2))
    for iterstep in range(iters):

        # Predict acceleration with HNN
        q = sim.coord[iterstep][0:3*nbodies].reshape(nbodies, 3)
        jacobi_input = np.hstack((np.reshape(masses[1:], (-1,1)), q[1:,:]))
        input_JS[iterstep, :] = jacobi_input[0:2].flatten()

        # Calculate phase
        theta_JS[iterstep, 0] = np.arctan2(q[1, 1], q[1, 0])
        theta_JS[iterstep, 1] = np.arctan2(q[2, 1], q[2, 0])

    
    accel_pred[:, 3:] = hnn.predict(input_JS)


    # Calculate prediction error
    E_accel = sim.dcoord_a - accel_pred

    return theta_JS, E_accel


def plot_errorPhaseOrbit(theta_JS, E_accel):
    E_accel_norm = np.linalg.norm(E_accel, axis = 1)
    plt.scatter(theta_JS[:, 0], theta_JS[:,1], c = E_accel_norm, cmap = 'viridis')
    plt.xlabel(r'$\theta_J$', fontsize = 22)
    plt.ylabel(r'$\theta_S$', fontsize = 22)
    plt.colorbar()
    plt.savefig('./Experiments/error_phase/phasemap.png')

    plt.show()

if __name__ == "__main__":
    h = 1e-1
    # multiple = 'JS'
    multiple = 'Asteroid_JS'
    
    run = 4
    if run == 1:
        if multiple == 'JS':
            t_end = 5000
            asteroids = 0
            asteroids_extra = 0
        else:
            t_end = 100
            asteroids = 3
            asteroids_extra = 0
        ##########################################
        # General
        ##########################################
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '', 0.3)            
        plot_CompPhys_trajectory(sim, sim2, sim3, t, t_end, asteroids, asteroids_extra, typePlot = multiple)
        # plot_NeurIPS_energyTogether(sim, sim2, sim3, t, asteroids, asteroids_extra, t_end, h, typePlot = multiple)
    elif run == 2.5:
        multiple = 'JS'
        t_end = 25
        asteroids = 0
        asteroids_extra = 0
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, False, '', 0.3)            
        plot_energyvsH(sim, sim2)
    elif run == 2: 
        """
        Flag vs no flag
        """
        t_end = 30
        asteroids = 1
        asteroids_extra = 1
        R = 0.3

        ##########################################
        # Test flag vs no flag
        ##########################################
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, False, '', R)

        accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
        accelerations_ANN_noflag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
        accelerations_DNN_noflag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")

        sim, sim2_f, sim3_f, t_f = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '2', R)
        print("time", t, t_f)

        # plot_general_flagvsnoflag(sim, sim2, sim2_f, [t, t_f], asteroids, asteroids_extra)
        
        accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
        accelerations_ANN_flag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
        accelerations_DNN_flag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")
        flags = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txtflag_list.txt")
        flags_DNN = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txtflag_list.txt")
        # plot_accel_flagvsnoflag(accelerations_WH, accelerations_DNN_noflag, accelerations_DNN_flag, \
        #             flags_DNN, [t, t_f], asteroids, asteroids_extra)
        plot_accel_flagvsnoflag(accelerations_WH, accelerations_ANN_noflag, accelerations_ANN_flag, \
                    flags, [t, t_f], asteroids, asteroids_extra)

    elif run == 3:
        """
        R vs N flags and computation time
        """
        multiple = 'Asteroid_JS'
        t_end = 50
        asteroids = 50
        asteroids_extra = 0
        ##########################################
        # Test flag vs no flag
        ##########################################
        # sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, False, '', 0.3)

        accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")

        flags_R = [ 0.7, 0.6, 0.5, 0.4,0.3]
        flag_n = np.zeros((len(flags_R),1+asteroids +3))
        flag_dnn = np.zeros((len(flags_R),1+asteroids +3))
        flag_n[:, 0] = flags_R
        flag_dnn[:, 0] = flags_R

        accel_i = list()
        flags_i = list()
        accel_i_DNN = list()
        flags_i_DNN = list()
        time_i = list()
        for flag_R in range(len(flags_R)):
            sim, sim2_f, sim3_f, t_f = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '2', flags_R[flag_R])

            accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
            accelerations_ANN_flag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
            accelerations_DNN_flag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")
            flags = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txtflag_list.txt")
            flags_DNN = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txtflag_list.txt")
            t_f = np.loadtxt("./Experiments/sun_jupiter_saturn/computationTime.txt")

            flag_n[flag_R, 1:] = np.sum(flags, axis = 0 )
            flag_dnn[flag_R, 1:] = np.sum(flags_DNN, axis = 0 )

            accel_i.append(accelerations_ANN_flag)
            flags_i.append(flags)

            accel_i_DNN.append(accelerations_DNN_flag)
            flags_i_DNN.append(flags_DNN)
            time_i.append(t_f[1])
            # plot_accel_flagvsnoflag(accelerations_WH, accelerations_ANN_noflag, accelerations_ANN_flag, \
            #         flags, [t, t_f], asteroids, asteroids_extra)


        np.savetxt("./Experiments/flagvsnoflag/flagvsR.txt", flag_n)
        np.savetxt("./Experiments/flagvsnoflag/flagvsR_DNN.txt", flag_dnn)
        
        plot_accel_flagvsR(accel_i, accelerations_WH, \
                    flags_i, time_i, asteroids, asteroids_extra, flags_R)

        # plot_accel_flagvsR(accel_i_DNN[0], accel_i_DNN[1], accel_i_DNN[2], \
        #             flags_i_DNN, [t, t_f], asteroids, asteroids_extra, flags_R)

    elif run == 4:
        multiple = 'Asteroid_JS'
        ##########################################
        # Asteroids vs time and energy
        ##########################################
        # run_asteroids(0.25)
        # plot_asteroids()
        plot_asteroids_NeurIPS()
        # polifit()

    elif run == 5:
        t_end = 1000
        asteroids = 0
        asteroids_extra = 0
        theta_JS, E_accel = compute_predError(t_end, h, asteroids, asteroids_extra)
        plot_errorPhaseOrbit(theta_JS, E_accel)

    elif run == 6:
        multiple = 'Asteroid_JS'
        t_end = 50
        asteroids = 2
        asteroids_extra = 0
        ##########################################
        # Test flag vs no flag
        ##########################################
        # sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, False, '', 0.3)

        # accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
        # accelerations_ANN_noflag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
        # accelerations_DNN_noflag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")

        flags_R = [0.1, 0.3, 0.7, 1.2]
        # flags_R = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
        flag_n = np.zeros((len(flags_R),1+asteroids +3))
        flag_dnn = np.zeros((len(flags_R),1+asteroids +3))
        flag_n[:, 0] = flags_R
        flag_dnn[:, 0] = flags_R

        accel_i = list()
        flags_i = list()
        accel_i_DNN = list()
        flags_i_DNN = list()
        energy_i = list()
        time_i = list()
        for flag_R in range(len(flags_R)):
            sim, sim2_f, sim3_f, t_f = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '2', flags_R[flag_R])

            accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
            accelerations_ANN_flag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
            accelerations_DNN_flag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")
            flags = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txtflag_list.txt")
            flags_DNN = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txtflag_list.txt")

            print(sim2_f.energy[-1])
            energy_i.append(sim2_f.energy)
            time_i.append(t_f)
            flag_n[flag_R, 1:] = np.sum(flags, axis = 0 )
            flag_dnn[flag_R, 1:] = np.sum(flags_DNN, axis = 0 )

            accel_i.append(accelerations_ANN_flag)
            flags_i.append(flags)

            accel_i_DNN.append(accelerations_DNN_flag)
            flags_i_DNN.append(flags_DNN)
            # plot_accel_flagvsnoflag(accelerations_WH, accelerations_ANN_noflag, accelerations_ANN_flag, \
            #         flags, [t, t_f], asteroids, asteroids_extra)
        energy_i.append(sim.energy)

        
        np.savetxt("./Experiments/flagvsnoflag/flagvsR.txt", flag_n)
        np.savetxt("./Experiments/flagvsnoflag/flagvsR_DNN.txt", flag_dnn)
        
        plot_accel_flagvsR_energy2(accel_i, energy_i, time_i, \
                    flags_i, asteroids, asteroids_extra, flags_R)

