"""
Created: July 2021 
Last modified: November 2022 
Author: Veronica Saz Ulibarrena 
Description: creation of a training and test dataset. Inputs taken from config_data.json. Integration done using wh_generate_database
"""
import sys
import os
import numpy as np
import time

import h5py
import json
from pyDOE import lhs

import time
from wh_generate_database import WisdomHolman

if __name__ == '__main__':

    # Load settings for the database
    with open("./config_data.json") as jsonFile:
        config_data = json.load(jsonFile)
        jsonFile.close()

    h = config_data['time_step'] # time step parameter 
    x_limits = np.array(list(config_data['ranges'].values()))

    # Save masses of different bodies
    m_JSb = np.array([1, x_limits[0,0], x_limits[0,1], 1]) # mass of Jup and saturn

    # Run N_exp integrations with a timestep h
    data = {}
    time_sim0 = time.time()
    for run in range(2): # 2 runs: Train and test dataset
        if run == 0: # 
            N_exp = config_data['N_exp']
        else:
            N_exp = config_data['N_exp_test']

        coords = None
        dcoords = None
        mass = None 
        
        # Latin hypercube for orbit parameters
        params = lhs(len(x_limits)-1, samples = N_exp) * (x_limits[1:, 1]- x_limits[1:, 0]) + x_limits[1:, 0] 
        time_0 = time.time()
        for trail in range(N_exp):
            print('Trail #%d/%d (train)' % (trail+1, N_exp))
            wh = WisdomHolman()

            # Add big bodies
            wh.particles.add(mass=m_JSb[0], pos=[0., 0., 0.,], vel=[0., 0., 0.,], name='Sun')
            wh.particles.add(mass=m_JSb[1], a=params[trail, 2] , e=0.1*np.random.rand(), \
                    i=np.pi/30*np.random.rand(), primary='Sun',f=2*np.pi*np.random.rand(), name='Jupiter')
            wh.particles.add(mass=m_JSb[2],  a=params[trail, 3] , e=0.1*np.random.rand(), \
                    i=np.pi/30*np.random.rand(), primary='Sun',f=2*np.pi*np.random.rand(), name='Saturn')

            # Add 1 asteroid if we are not in the SJS case
            if config_data['asteroids'] != 0:
                m_JSb[3] = params[trail, 0] / 1.9891e30 # Move to mass of the sun
                wh.particles.add(mass=m_JSb[3], a=params[trail, 1] , e=0.1*np.random.rand(), \
                    i=np.pi/30*np.random.rand(), name='asteroid', primary='Sun',f=2*np.pi*np.random.rand())
            else:
                m_JSb = np.delete(m_JSb, 3) # delete mass of asteroid to not include it
            
            print(wh.particles)
            wh.h = h
            wh.acceleration_method = 'numpy'
            wh.integrate(config_data['t_final'])
            # Save coordinates of every experiment
            if coords is None and dcoords is None:
                if np.isnan(wh.coord).sum() == 0 and np.isnan(wh.dcoord).sum() == 0:
                    coords = np.array(wh.coord)
                    dcoords = np.array(wh.dcoord)
                    m_i = np.tile(m_JSb, (np.shape(wh.coord)[0],1) )
                    mass = m_i
            else:
                if np.isnan(wh.coord).sum() == 0 and np.isnan(wh.dcoord).sum() == 0:
                    coords = np.append(coords, np.array(wh.coord), axis=0)
                    dcoords = np.append(dcoords, np.array(wh.dcoord), axis=0)
                    m_i = np.tile(m_JSb, (np.shape(wh.coord)[0],1) )
                    mass = np.vstack((mass, m_i))

            if run == 0: # tRaining dataset
                data['coords'] = coords
                data['dcoords'] = dcoords
                data['mass'] = mass
            else: # test dataset
                data['test_coords'] = coords
                data['test_dcoords'] = dcoords
                data['test_mass'] = mass

            # Save to file
            path_dataset = config_data['data_dir']
            with h5py.File(os.path.join(path_dataset, 'train_test.h5'), 'w') as h5f:
                for dset in data.keys():
                    h5f.create_dataset(dset, data=data[dset], compression="gzip")

            print("Time: ", time_sim0 - time.time())

    # Save to file
    path_dataset = config_data['data_dir']
    with h5py.File(os.path.join(path_dataset, 'train_test.h5'), 'w') as h5f:
        for dset in data.keys():
            h5f.create_dataset(dset, data=data[dset], compression="gzip")
    print('Training data generated.')
    print("Number of samples: ", np.shape(coords)[0])