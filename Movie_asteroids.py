import rebound
import sys
import numpy as np
import time
import h5py

from wh_tf_flag import WisdomHolman
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation 

import subprocess

from data import load_json
from nn_tensorflow import  ANN

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.formatter.useoffset'] = False



color = [ 'skyblue','royalblue', 'blue', 'navy','slateblue', 'coral', 'salmon',\
    'orange', 'burlywood', 'lightgreen', 'olivedrab','darkcyan' ]

def load_model():
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

def add_particles(sr, sim, asteroids):
    """
    add_particles: add particles to integrator
    INPUTS: 
        sim: initialized simulation
        asteroids: number of asteroids
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
    return sim

def simulate(t_end, h, asteroids, flag, name, R):
    """
    simulate: run integration using WH, HNN, and DNN
    INPUTS:
        t_end: final time of integration
        h: time-step
        asteroids: number of asteroids
        flag: True (verifying prediction of ANN), False (not verifying)
        name: name of file to save
    OUTPUTS:
        sim2: simulation for HNN
        t_nn: computation time for HNN
    """
    # Setup rebound 
    sr = rebound.Simulation()
    sr.units = {'yr', 'au', 'MSun'}
    sr.add('Sun')
    sr.add('Jupiter')
    sr.add('Saturn')
    sr.save('ic_sun_jupiter_saturn.bin')
    ########################################################
    #### WH + HNN
    ########################################################
    EXPERIMENT_DIR = '.'
    sys.path.append(EXPERIMENT_DIR)

    nih_model = load_model()

    sim2 = WisdomHolman(hnn=nih_model, CONST_G=sr.G, \
                        accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_ANN.txt",\
                        flag = flag, \
                        multiple_nets = 'Asteroid_JS', R= R)
    sim2 = add_particles(sr, sim2, asteroids)

    t0_nn = time.time()
    sim2.output_file = './Experiments/movie/sun_jupiter_saturn_nih'+name+'.hdf5'
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
    print("Time", t_nn)

    return sim2, t_nn

def saveData(sim, t, t_end, h):
    data_nb = sim.buf.recorder.data

    timesteps = np.shape(data_nb['x'])[0]
    bodies = np.shape(data_nb['x'])[1]
    r = np.zeros((timesteps, bodies, 3))
    r[:,:, 0] = data_nb['x']
    r[:,:, 1] = data_nb['y']
    r[:,:, 2] = data_nb['z']
    data = dict()
    data['E'] = sim.energy
    data['r'] = r
    data['t'] = np.arange(0, t_end, h)
    # data['h'] = h
    # data['t_comp'] = t

    with h5py.File("./Experiments/movie/movieAsteroidsData.h5", 'w') as h5f:
        for dset in data.keys():
            h5f.create_dataset(dset, data=data[dset], compression="gzip")
    

def plot():
    with h5py.File("./Experiments/movie/movieAsteroidsData.h5", 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ko')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        # ax.plot(0,0)
        return ln,
    
    def update(frame):
        # xdata.append(frame)
        # ydata.append(np.sin(frame))
        xdata = data['r'][frame, :, 0]
        ydata = data['r'][frame, :, 1]
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames = np.arange(0, len(data['t']),1),\
                                    init_func= init, blit = True)
    plt.show()

if __name__ == "__main__":
    h = 1e-1
    multiple = 'Asteroid_JS'
    t_end = 5000

    asteroids = 5
    ##########################################
    # General
    ##########################################
    # sim, t = simulate(t_end, h, asteroids, True, '', 0.3)
    # saveData(sim, t, t_end, h)
    ani = plot()