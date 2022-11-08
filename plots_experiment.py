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


from data import load_json
from nn_tensorflow import  ANN
from wh_tf_flag import WisdomHolman
from plots_papers import plot_NeurIPS,  \
                plot_CompPhys_trajectory, plot_CompPhys_trajectory_JS,\
                plot_general_flagvsnoflag,\
                plot_energyvsH,\
                plot_accel_flagvsnoflag, plot_accel_flagvsR,\
                plot_asteroids, plot_asteroids_accel,\
                polifit, plot_errorPhaseOrbit

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
    

def add_particles(sim, sr, asteroids, asteroids_extra):
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
    # a_a = np.linspace(2.2, 3.2, num= asteroids)
    a_a = np.array([2.8, 3.2])
    f_a = np.linspace(0, 2*np.pi, num= asteroids)
    # np.random.shuffle(a_a)
    # np.random.shuffle(f_a)

    m_a_e = np.array([1e18, 1e19, 5e19, 1e21]) / 1.9891e30
    a_a_e = [1.8, 1.0, 4.0] #AU
    

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
    sr = rebound.Simulation()
    sr.units = {'yr', 'au', 'MSun'}

    sr.add('Sun')
    sr.add('Jupiter')
    sr.add('Saturn')
    sr.save('ic_sun_jupiter_saturn.bin')

    sim = WisdomHolman(CONST_G=sr.G, accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
    sim = add_particles(sim, sr, asteroids, asteroids_extra)

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
    sim2 = add_particles(sim2, sr, asteroids, asteroids_extra)

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
    sim3 = add_particles(sim3, sr, asteroids, asteroids_extra)

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

    savedata = dict()

    asteroids = [5, 10, 20, 30, 50, 70, 100, 200, 500, 1000, 2000]
    t_num = np.zeros((len(asteroids), 3))
    t_num2 = np.zeros((len(asteroids), 3))
    e_energy = np.zeros((len(asteroids), 3, int(t_end//h)+1))
    
    for test in range(len(asteroids)):
        m_a = np.random.uniform(low = 1e19, high = 1e20, size = (asteroids[test],)) /1.9891e30 # To mass of the sun
        a_a = np.linspace(2.0, 3.5, num= asteroids[test])
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
        
def compute_predError(t_end, h, asteroids, asteroids_extra):
    """
    compute_predError for errorPhaseOrbit
    """
    sr = rebound.Simulation()
    sr.units = {'yr', 'au', 'MSun'}

    sr.add('Sun')
    sr.add('Jupiter')
    sr.add('Saturn')
    sr.save('ic_sun_jupiter_saturn.bin')

    # Run numerically
    sim = WisdomHolman(CONST_G=sr.G, accel_file_path = "./Experiments/sun_jupiter_saturn/accelerations_WH.txt")
    sim = add_particles(sim, sr,  asteroids, asteroids_extra)

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


if __name__ == "__main__":
    # Choose case:
    # multiple = 'JS'
    multiple = 'Asteroid_JS'

    h = 1e-1
    
    # Choose experiment from 1 to 6
    run = 4
    if run == 1:
        """
        Run simulation and plot trajectories
        """
        if multiple == 'JS':
            t_end = 5000
            asteroids = 0
            asteroids_extra = 0
        else:
            t_end = 1000
            asteroids = 2
            asteroids_extra = 1

        ##########################################
        # General
        ##########################################
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, True, '', 0.3)   
        if multiple == 'JS':
            plot_CompPhys_trajectory_JS(sim, sim2, sim3, t, t_end, asteroids, asteroids_extra, typePlot = multiple)
        else:         
            plot_CompPhys_trajectory(sim, sim2, sim3, t, t_end, asteroids, asteroids_extra, typePlot = multiple)

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
        plot_accel_flagvsnoflag(accelerations_WH, accelerations_ANN_noflag, accelerations_ANN_flag, \
                    flags, [t, t_f], asteroids, asteroids_extra, t_end)

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

        np.savetxt("./Experiments/flagvsnoflag/flagvsR.txt", flag_n)
        np.savetxt("./Experiments/flagvsnoflag/flagvsR_DNN.txt", flag_dnn)
        
        plot_accel_flagvsR(accel_i, accelerations_WH, \
                    flags_i, time_i, asteroids, asteroids_extra, flags_R)

    elif run == 4:
        multiple = 'Asteroid_JS'
        ##########################################
        # Asteroids vs time and energy
        ##########################################
        run_asteroids(0.25)
        # plot_asteroids()
        plot_asteroids_accel()
        # polifit()

    elif run == 5:
        """
        Plot interactive energy vs output of HNN
        """
        multiple = 'JS'
        t_end = 25
        asteroids = 0
        asteroids_extra = 0
        sim, sim2, sim3, t = simulate(t_end, h, asteroids, asteroids_extra, multiple, False, '', 0.3)            
        plot_energyvsH(sim, sim2, t_end)

    elif run == 6:
        t_end = 1000
        asteroids = 0
        asteroids_extra = 0
        theta_JS, E_accel = compute_predError(t_end, h, asteroids, asteroids_extra)
        plot_errorPhaseOrbit(theta_JS, E_accel)
