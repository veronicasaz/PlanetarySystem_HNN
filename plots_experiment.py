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
from data import load_json

from nn_tensorflow import  ANN

# Setup rebound 
sr = rebound.Simulation()
sr.units = {'yr', 'au', 'MSun'}
sr.add('Sun')
sr.add('Jupiter')
sr.add('Saturn')
sr.save('ic_sun_jupiter_saturn.bin')

color = [ 'skyblue','royalblue', 'blue', 'navy','slateblue', 'coral', 'salmon',\
    'orange', 'burlywood', 'lightgreen', 'olivedrab','darkcyan' ]

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
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids/', path_std = path_std, std = True)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids/')
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
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS_DNN/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'JS/Model_JS_DNN/')
        Nets.append(ANN_tf)
        
        config['bodies'] = 3 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids_DNN/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids_DNN/')
        Nets.append(ANN_tf)
        return Nets
    elif multiple == 'Asteroid_avgJS':
        path_model = "./ANN_tf/"
        ANN_tf = ANN(config, path_model = path_model+'asteroid/Model_asteroids_DNN/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'asteroid/Model_asteroids_DNN/')
        return ANN_tf
    else: # Sun, Jupiter and Saturn or other cases with 1 network
        path_model = "./ANN_tf/"
        config['output_dim'] = 'a'
        config['bodies'] = 2 # Case with SJS
        ANN_tf = ANN(config, path_model = path_model+'JS/Model_JS_DNN/', path_std = path_std)
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
        U += G * m[0] *m[i] / np.linalg.norm(r[0, :]- r[i,:])
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
    m_a = np.array([1.5e-3, 3e-3, 5e-3, 6e-3, 4e-3]) / 1.9891e30 # To mass of the sun
    a_a = np.linspace(2.2, 3.2, num= asteroids)
    f_a = np.linspace(0, 2*np.pi, num= asteroids)
    # np.random.shuffle(a_a)
    # np.random.shuffle(f_a)

    m_a_e = np.array([1.5e-3, 3e-3, 5e-3]) / 1.9891e30
    a_a_e = [1.0, 1.5, 4.0] #AU

    for i, p in enumerate(sr.particles): # Planets
        print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
        sim.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))
    for j in range(asteroids):
        sim.particles.add(mass =m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0.0, f= f_a[j])
    for j in range(asteroids_extra):
        sim.particles.add(mass =m_a_e[j], a=a_a_e[j] , e=0.1, i=0.0, f= f_a[j])
    return sim

def simulate(t_end, h, asteroids, asteroids_extra, multiple, flag, name):
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
                        multiple_nets = multiple)
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
                        multiple_nets = multiple)
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

def plot_general_printversion(sim, sim2, sim3, t, asteroids, asteroids_extra):
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

    fig, axes = plt.subplots(3,3, figsize=(18,15))
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
    colors = [color[8], color[9], color[3], color[6], color[0], color[10], color[5], color[7], color[1], color[11]]
    line = ['-', '--', '-.', ':', '-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':']
    lnwidth = 3
    for i in range(1, data_nih['x'].shape[1]):
        # coordinates in the first column
        axes[0,1].set_title("Numerical integrator result: %.3f s"%(t_num), fontsize = titlesize)
        axes[1,1].set_title("Hamiltonian Neural Network result: %.3f s"%(t_nn), fontsize = titlesize)
        axes[2,1].set_title("Deep Neural Network result: %.3f s"%(t_dnn), fontsize = titlesize)

        axes[0,0].plot(data_nb['x'][:,i], data_nb['y'][:,i], linestyle = line[i], linewidth = lnwidth, color = colors[i], label=names[i])
        axes[0,0].axis('equal')
        axes[0,0].set_xlabel('$x$',fontsize = labelsize)
        axes[0,0].set_ylabel('$y$',fontsize = labelsize)
        axes[0,0].legend(loc = 'upper right', fontsize = 16, framealpha = 0.9)
        axes[1,0].plot(data_nih['x'][:,i], data_nih['y'][:,i], linestyle = line[i],linewidth = lnwidth, color = colors[i], label=names[i])
        axes[1,0].axis('equal')
        axes[1,0].set_ylabel('$y$',fontsize = labelsize)
        axes[1,0].set_xlabel('$x$',fontsize = labelsize)
        axes[2,0].plot(data_dnn['x'][:,i], data_dnn['y'][:,i], linestyle = line[i], linewidth = lnwidth,color = colors[i], label=names[i])
        axes[2,0].axis('equal')
        axes[2,0].set_ylabel('$y$',fontsize = labelsize)
        axes[2,0].set_xlabel('$x$',fontsize = labelsize)

        # eccentricity in the second column
        axes[0,1].plot(time, data_nb['ecc'][:,i], linestyle = line[i],  color = colors[i], linewidth = lnwidth, label=names[i])
        axes[0,1].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[0,1].set_ylabel('$e$',fontsize = labelsize)
        axes[1,1].plot(time, data_nih['ecc'][:,i], linestyle = line[i],  color = colors[i], linewidth = lnwidth, label=names[i])
        axes[1,1].set_ylabel('$e$',fontsize = labelsize)
        axes[1,1].set_xlabel('$t$ [years]',fontsize = labelsize)
        axes[2,1].plot(time, data_dnn['ecc'][:,i], linestyle = line[i],  color = colors[i],linewidth = lnwidth,  label=names[i])
        axes[2,1].set_ylabel('$e$',fontsize = labelsize)
        axes[2,1].set_xlabel('$t$ [years]',fontsize = labelsize)
                
    # energy drift in the second column
    axes[0,2].set_xlabel('$t$ [years]',fontsize = labelsize)
    axes[0,2].plot(time, sim.energy, linestyle = '-',  color = color[9], alpha=1)
    axes[0,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[0, 2].ticklabel_format(useOffset=False)

    axes[1,2].plot(time, sim2.energy, linestyle = '-',  color = color[3],alpha=1, label= 'Error with WH-HNN')
    axes[1,2].plot(time, sim.energy, linestyle = '-',  color = color[9],alpha=1, label= 'Error with WH')
    axes[1,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)
    axes[1,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[1,2].ticklabel_format(useOffset=False)
    axes[1,2].set_xlabel('$t$ [years]',fontsize = labelsize)

    axes[2,2].plot(time, sim3.energy, alpha=1, linestyle = '-',  color = color[3],label= 'Error with WH-DNN')
    axes[2,2].plot(time, sim.energy, alpha=1, linestyle = '-',  color = color[9], label= 'Error with WH')
    axes[2,2].legend(loc = 'upper left', fontsize = 18, framealpha = 0.9)
    axes[2,2].set_ylabel('$dE/E_0$',fontsize = labelsize)
    axes[2,2].ticklabel_format(useOffset=False)
    axes[2,2].set_xlabel('$t$ [years]',fontsize = labelsize)

    for i in range(3):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.png' % t_end)
    plt.savefig('./Experiments/sun_jupiter_saturn/sun_jupiter_saturn_%dyr.pdf' % t_end)
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

    h = 1e-1
    x = np.arange(0, len(accelerations_ANN_noflag)*h, h)
    # x2 = np.copy(x)

    asteroids_plot = asteroids+asteroids_extra
    fig, axes = plt.subplots(3,2+asteroids_plot, figsize=(20,8))
    fig.subplots_adjust(top=0.9,hspace = 0.4, wspace= 0.5)

    for plot in range(1,3+asteroids_plot):
        a_WH = np.zeros(len(accelerations_ANN_noflag))
        a_ANN = np.zeros(len(accelerations_ANN_noflag))
        a_DNN = np.zeros(len(accelerations_ANN_flag))
        
        for item in range(len(accelerations_ANN_noflag)):
            a_WH[item] = np.linalg.norm(accelerations_WH[item, plot*3:plot*3+3])
            a_ANN[item] = np.linalg.norm(accelerations_ANN_noflag[item, plot*3:plot*3+3])
            a_DNN[item] = np.linalg.norm(accelerations_ANN_flag[item, plot*3:plot*3+3])
        
        axes[0, plot-1].plot(x, a_WH, color = color[3], label = 'WH')
        axes[1, plot-1].plot(x, a_ANN, color = color[9], label = 'Without flags')
        axes[2, plot-1].plot(x, a_DNN, color = color[5], label = 'With flags')
        
        index_DNN = np.where(flags[:, plot] == 0)[0]
        x2_DNN = np.copy(x)
        x2_DNN = np.delete(x2_DNN, index_DNN)
        axes[2, plot-1].scatter(x2_DNN, np.delete(a_DNN, index_DNN), color = color[5], label = 'Numerically')

        axes[0,plot-1].set_ylabel('a ($au/yr^2$)', fontsize = 18)
        axes[0,plot-1].grid(alpha = 0.5)
        axes[0,plot-1].set_title(names[plot-1], fontsize = 20)
        
        # axes[0,plot-1].set_xticklabels(axes[0,plot-1].get_xticks() , fontsize = 15)
        # axes[0,plot-1].set_yticklabels(axes[0,plot-1].get_yticks(), fontsize = 15)
        ticks = -np.log10(axes[0,plot-1].get_yticks())
        dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
        axes[0,plot-1].set_yticklabels(np.round(trunc(axes[0,plot-1].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 14)
        # axes[0,plot-1].set_yticklabels(axes[0,plot-1].get_yticks(), rotation = 0, fontsize = 14)
        axes[0,plot-1].set_xticklabels(np.round(trunc(axes[0,plot-1].get_xticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 14)

        axes[1,plot-1].set_ylabel('a ($au/yr^2$)', fontsize = 18)
        axes[1,plot-1].grid(alpha = 0.5)
        # axes[1,plot-1].set_title(names[plot-1])
        ticks = -np.log10(axes[1,plot-1].get_yticks())
        dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
        axes[1,plot-1].set_yticklabels(np.round(trunc(axes[1,plot-1].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 14)
        axes[1,plot-1].set_xticklabels(np.round(trunc(axes[1,plot-1].get_xticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 14)
        
        axes[2,plot-1].set_xlabel('$t$ (yr)', fontsize = 18)
        axes[2,plot-1].set_ylabel('a ($au/yr^2$)', fontsize = 18)
        axes[2,plot-1].grid(alpha = 0.5)
        # axes[2,plot-1].set_title(names[plot-1])
        axes[2,plot-1].annotate("Flags: %i / %i"%(np.count_nonzero(flags[:, plot]), len(accelerations_ANN_flag)), \
                xy =  (x[int(len(x)//2)]*1.2, max((a_DNN))*0.9), fontsize = 12)
        ticks = -np.log10(axes[2,plot-1].get_yticks())
        dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
        axes[2,plot-1].set_yticklabels(np.round(trunc(axes[2,plot-1].get_yticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 14)
        axes[2,plot-1].set_xticklabels(np.round(trunc(axes[2,plot-1].get_xticks(), decs = 5), decimals = dec+1), rotation = 0, fontsize = 14)
        

    axes[0,1].legend(loc = 'center left', framealpha = 0.5, fontsize = 14)
    axes[1,1].legend(loc = 'center left', framealpha = 0.5, fontsize = 14)
    axes[2,1].legend(loc = 'center left', framealpha = 0.5, fontsize = 14)

    # plt.tight_layout()
    plt.savefig('./Experiments/flagvsnoflag/sun_jupiter_saturn_accel_%dyr_flagcomparison.png' % t_end)
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
        path_model = "./ANN_tf/trained_nets/"
        Nets = []
        ANN_tf = ANN_JS(config, path_model = path_model+'Model_JS/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'Model_JS/')
        Nets.append(ANN_tf)
        ANN_tf = ANN(config, path_model = path_model+'Model_asteroids/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'Model_asteroids/')
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
        path_model = "./ANN_tf/trained_nets/"
        ANN_tf = ANN(config, path_model = path_model+'Model_JS/', path_std = path_std)
        ANN_tf.load_model_fromFile(path_model+'Model_JS/')
        return ANN_tf

def run_asteroids():
    """
    run_asteroids: simulate with different numbers of asteroids
    """
    sr = rebound.Simulation()
    sr.units = {'yr', 'au', 'MSun'}

    sr.add('Sun')
    sr.add('Jupiter')
    sr.add('Saturn')
    sr.save('ic_sun_jupiter_saturn.bin')

    multiple = 'Asteroid_JS' # False: one network for all, Asteroid_JS: one for asteroids, one for JS, 1,2, 3... number to average
    nih_model = load_model_asteroids()

    t_end = 20
    h = 1.0e-1

    # asteroids = [5, 10, 20, 30, 50, 70, 90, 100, 150, 200]
    asteroids = [1000, 10000]
    t_num = np.zeros((len(asteroids), 3))
    e_energy = np.zeros((len(asteroids), 3, int(t_end//h)+1))
    for test in range(len(asteroids)):
        m_a = (np.random.random_sample(size=asteroids[test]) * (4e-3 - 1.0e-3) + 1e-3 ) /1.9891e30 # To mass of the sun
        a_a = np.linspace(2.2, 3.2, num= asteroids[test])
        f_a = np.linspace(0, 2*np.pi, num= asteroids[test])
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

        t0_num = time.time()
        sim.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nb.hdf5'
        sim.integrator_warmup()
        sim.h = h
        sim.acceleration_method = 'numpy'
        sim.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
        sim.buf.recorder.start()
        sim.integrate(t_end, nih=False)
        sim.buf.flush()
        e_energy[test, 0,:] = sim.energy

        sim.stop()

        t_num[test, 0] = time.time() - t0_num
        sim.buf.recorder.data.keys()

        ########################################################
        #### WH + HNN false flag
        ########################################################
        EXPERIMENT_DIR = '.'
        sys.path.append(EXPERIMENT_DIR)

        sim2 = WisdomHolman(hnn=nih_model, CONST_G=sr.G, \
                            accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_ANN.txt",\
                            flag = False, \
                            multiple_nets = multiple)
        # sim.integrator = 'WisdomHolman'

        for i, p in enumerate(sr.particles):
            print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
            sim2.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))

        for j in range(asteroids[test]):
            sim2.particles.add(mass = m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0, f= f_a[j])


        t0_nn = time.time()
        sim2.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nih.hdf5'
        sim2.integrator_warmup()
        sim2.h = h
        sim2.acceleration_method = 'numpy'
        sim2.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
        sim2.buf.recorder.start()
        sim2.integrate(t_end, nih=True)
        sim2.buf.flush()
        e_energy[test, 1, :] = sim2.energy
        sim2.stop()

        t_num[test, 1] = time.time() - t0_nn

        ########################################################
        #### WH + HNN true flag
        ########################################################
        sim2 = WisdomHolman(hnn=nih_model, CONST_G=sr.G, \
                            accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_ANN.txt",\
                            flag = True, \
                            multiple_nets = multiple)

        for i, p in enumerate(sr.particles):
            print(p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
            sim2.particles.add(mass=p.m, pos=(p.x, p.y, p.z), vel=(p.vx, p.vy, p.vz))

        for j in range(asteroids[test]):
            sim2.particles.add(mass = m_a[j%len(m_a)], a=a_a[j] , e=0.1, i=0, f= f_a[j])

        t0_nn = time.time()
        sim2.output_file = './Experiments/sun_jupiter_saturn/sun_jupiter_saturn_nih.hdf5'
        sim2.integrator_warmup()
        sim2.h = h
        sim2.acceleration_method = 'numpy'
        sim2.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
        sim2.buf.recorder.start()
        sim2.integrate(t_end, nih=True)
        sim2.buf.flush()
        e_energy[test, 2, :] = sim2.energy
        sim2.stop()

        t_num[test, 2] = time.time() - t0_nn
    
    savedata = dict()
    savedata['time'] = t_num
    savedata['energy'] = e_energy
    savedata['settings'] = [h, t_end]
    savedata['asteroids'] = asteroids

    with h5py.File("./Experiments/AsteroidVsTime/asteroids_timeEnergy.h5", 'w') as h5f:
        for dset in savedata.keys():
            h5f.create_dataset(dset, data=savedata[dset], compression="gzip")

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
    # t_num = np.stack((data['time'], data2['time']), axis = 0)
    # e_energy = np.stack((data['energy'], data2['energy']), axis = 0)
    # h, t_end = np.stack((data['settings'], data2['settings']), axis = 0)
    # asteroids = np.stack((data['asteroids'], data2['asteroids']), axis = 0)

    t_num = data['time']
    e_energy = data['energy']
    h, t_end = data['settings']
    asteroids = data['asteroids']

    ########################################################
    #### Plots together
    ########################################################
    fig, axes = plt.subplots(1,2, figsize=(18,6), gridspec_kw={'width_ratios': [3, 1]})
    axes[0].plot(asteroids, t_num[:,0], color = color[3],  linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[0].plot(asteroids, t_num[:,1], color = color[9],  linestyle='-', linewidth = 2, marker = 'x',markersize = 10, label = 'HNN')
    axes[0].plot(asteroids, t_num[:,2], color = color[5],  linestyle='--', linewidth = 2, marker = '^', markersize = 10,label = 'WH-HNN')
    axes[0].set_xlabel('Number of asteroids', fontsize = 20)
    axes[0].set_ylabel('Computation time', fontsize = 20)
    axes[0].legend(fontsize = 20)
    axes[0].grid(alpha = 0.5)

    axes[1].plot(asteroids, e_energy[:,0,-1]-e_energy[:,0,0], color = color[3], linestyle='-', linewidth = 2, marker = 'o', markersize = 10,label = 'WH')
    axes[1].plot(asteroids, e_energy[:,1,-1]-e_energy[:,1,0], color = color[9],  linestyle='-',  linewidth = 2, marker = 'x', markersize = 10,label = 'HNN')
    axes[1].plot(asteroids, e_energy[:,2,-1]-e_energy[:,2,0], color = color[5], linestyle='--',   linewidth = 2, marker = '^', markersize = 10,label = 'WH-HNN')
    axes[1].set_xlabel('Number of asteroids', fontsize = 20)
    axes[1].set_ylabel('Mean energy error of last 10 steps', fontsize = 20)
    axes[1].legend(fontsize = 20)
    axes[1].grid(alpha = 0.5)

    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].tick_params(axis='both', which='minor', labelsize=8)
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].tick_params(axis='both', which='minor', labelsize=8)

    # plt.suptitle("Time and Energy error \n $t_f$ = %0.3f and $h$ = %0.3f"%(t_end, h), fontsize = 18)
    plt.tight_layout()
    plt.savefig('./Experiments/AsteroidVsTime/timeVsError_%dyr.png' % t_end)
    plt.show()

if __name__ == "__main__":
    h = 1e-1
    multiple = 'JS'
    # multiple = 'Asteroid_JS'
    
    run = 1
    if run == 1:
        t_end = 5000
        if multiple == 'JS':
            asteroids = 0
            asteroids_extra = 0
        else:
            asteroids = 2
            asteroids_extra = 1
        ##########################################
        # General
        ##########################################
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '')
        # plot_general(sim, sim2, sim3, t, asteroids, asteroids_extra)
        plot_general_printversion(sim, sim2, sim3, t, asteroids, asteroids_extra)
        plot_accelerations(sim, sim2, sim3, typenet = multiple)
    
    elif run == 2: 
        t_end = 30
        asteroids = 2
        asteroids_extra = 0
        ##########################################
        # Test flag vs no flag
        ##########################################
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, False, '')

        accelerations_WH = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
        accelerations_ANN_noflag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_ANN.txt")
        accelerations_DNN_noflag = np.loadtxt("./Experiments/sun_jupiter_saturn/accelerations_DNN.txt")

        sim, sim2_f, sim3_f, t_f = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '2')

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
        ##########################################
        # Asteroids vs time and energy
        ##########################################
        # run_asteroids()
        plot_asteroids()