"""
Created: July 2021 
Last modified: July 2022 
Author: Veronica Saz Ulibarrena 
Description: Process of dataset, plot of dataset, training of neural network, plot results
"""
import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from nn_tensorflow import  ANN, predict_multiple #TODO: get back to this
from data import get_dataset, get_traintest_data, load_json
from plots_database import plot_pairplot, plot_distribution, plot_correlation,\
             plot_distance, plot_distance3D, plot_covariance_samples, plot_covariance,\
             plot_trajectory
from test_dataset import plot_prediction_error, plot_prediction_error_H, \
            load_dataset, plot_prediction_error_HNNvsDNN, plot_prediction_error_HNNvsDNN_JS


def process_data(config):
    """
    process data: load the dataset generated and divide it into inputs/outputs, normalize...
    Inputs: 
        config: configuration file
    Outputs:
        data: dictionary with training coords, dcoords, test coords, dcoords
    """
    # Load data 
    data_0 = get_dataset(config['data_dir'], 'train_test.h5',  verbose=True)
    data, config['input_dim'] = get_traintest_data(data_0, config)
    print("Training samples:", np.shape(data['coords']))

    return data
    
def get_data(config, name = None, plot = False):
    """
    get_data: take the processed data, reduce the number of samples taken and plot it
    Inputs: 
        config: configuration file
        plot: if true, plot results
    Outputs: 
        data2: new dataset
    """
    if config['rotate'] == False:
        data = get_dataset(config['data_dir'], name +'train_test_processed.h5')
    else:
        data = get_dataset(config['data_dir'], name+'train_test_rotated.h5')
    
    # Reduce size of dataset to chosen in config file
    data2 = dict()
    data2['coords'] = data['coords'][0:config['train_samples']]
    data2['dcoords'] = data['dcoords'][0:config['train_samples']]
    data2['test_coords'] = data['test_coords'][0:config['test_samples']]
    data2['test_dcoords'] = data['test_dcoords'][0:config['test_samples']]
    
    print("Training samples:", np.shape(data2['coords']))

    # Plot
    if plot == True: # choose plots to make
        plot_distribution(data2['coords'], data2['dcoords'], '', name)
        # plot_distribution(data2['test_coords'], data2['test_dcoords'], '_test', name)
        # plot_pairplot(data2, name)
        # plot_correlation(data2, name)
        # plot_distance(data2, name)
        # plot_distance3D(data2, name)
        # plot_covariance_samples(data2, name)
        # plot_covariance(data2, name)
        # plot_trajectory(data2, name)

    return data2

def train_DNN(config, data, path_model = None):
    """
    train_DNN: train deep neural network
    Inputs: 
        config: configuration file
        data: dataset
    """
    # Initialize network
    ANN_tf = ANN(config, path_model = path_model)
    print(ANN_tf.model)

    # Load from trained network 
    # ANN_tf.load_weights()
    # TRAIN
    ANN_tf.train(data)
    ANN_tf.saveTraining()
    ANN_tf.plotTraining(ANN_tf.path_model, path_training_process = ANN_tf.path_model)

def autokeras(config, data):
    """
    autokeras: use autokeras package to find best hyperparameters. 
    NOTE: can't work with custom HNN
    Inputs: 
        config: configuration file
        data: dataset
    """
    ANN_tf = ANN(config)
    ANN_tf.train_autokeras(data)
    ANN_tf.load_model_ak(ANN_tf.path_model)

def train_multiple(config, data, number):
    """
    train_multiple: train many neural networks with different seed and save them in folders
    Inputs: 
        config: configuration file
        data: dataset
        number: number of networks to train
    """
    s = np.arange(1,1000,1) # seed
    np.random.shuffle(s)

    nets = list()
    for net in range(number):
        print("=============================================================")
        print("Network number: ", net)
        print("=============================================================")
        ANN_tf = ANN(config, seed = s[net], \
                            path_model = "./ANN_tf/trained_nets/multiple/"+str(net+1)+'/',\
                            type_network = "DNN")
        ANN_tf.load_model_fromFile("./ANN_tf/trained_nets/multiple/"+str(net+1)+'/')
        nets.append(ANN_tf)

        ANN_tf.train(data)
        ANN_tf.saveTraining()
        # Plot training: make sure plt.show() does not appear
        ANN_tf.plotTraining("./ANN_tf/trained_nets/multiple/"+str(net+1)+'/', \
                            path_training_process = "./ANN_tf/trained_nets/multiple/"+str(net+1)+'/')
        # Evaluate training results in test dataset
        x, y_pred, y_real = load_dataset(config, "./config_ANN.json", data, \
                            path_model =  "./ANN_tf/trained_nets/multiple/"+str(net+1)+'/')
        # Plot test results: make sure plt.show() does not appear
        plot_prediction_error("./ANN_tf/trained_nets/multiple/"+str(net+1)+'/', x, y_pred, y_real)
    
    # Find prediction error when multiple networks are combined
    y_pred, y_std = predict_multiple(nets, x, divide_by_mass = False)
    plot_prediction_error("./ANN_tf/trained_nets/multiple/", x, y_pred, y_real)

def predict(path_pic, config, settings_file_path, data):
    """
    predict: evaluate trained network on test dataset
    Inputs: 
        path_pic: path to save figure
        config: configuration file
        settings_file_path: path to config file
        data: dataset
    """
    x, y_pred, y_real = load_dataset(config, settings_file_path, data, path_model = path_pic)

    if config['loss_variable'] == 'dI': # loss is with accelerations
        plot_prediction_error(path_pic, x, y_pred, y_real)
    
    elif config['loss_variable'] == 'H': # If loss is h
        plot_prediction_error_H(path_pic, x, y_pred, y_real)
        x, y_pred, y_real = load_dataset(config, settings_file_path, drdv = True)
        plot_prediction_error(path_pic, x, y_pred, y_real)

def predict_HNNvsDNN(path_pic, settings, settings_file_path, data, name):
    """
    predict_HNNvsDNN: evaluate trained networks on test dataset and plot together
    Inputs: 
        path_pic: path to save figure
        config: configuration file
        settings_file_path: path to config file
        data: dataset
    """
    if name == 'asteroid':
        # predict
        settings['output_dim'] = 'H'
        x, y_pred, y_real = load_dataset(settings, settings_file_path, data , path_model = path_pic + "Model_asteroids/1/")
        settings['output_dim'] = 'H'
        x3, y_pred3, y_real3 = load_dataset(settings, settings_file_path, data, path_model = path_pic + "Model_asteroids/2/")
        settings['output_dim'] = 'a'
        x2, y_pred2, y_real2 = load_dataset(settings, settings_file_path, data, path_model = path_pic + "Model_asteroids_DNN/")

        # plot together    
        plot_prediction_error_HNNvsDNN(path_pic, x, y_pred, y_real, x2, y_pred2, y_real2, x3, y_pred3, y_real3)
    else:
        settings['output_dim'] = 'H'
        x, y_pred, y_real = load_dataset(settings, settings_file_path, data , path_model = path_pic + "Model_JS/")
        settings['output_dim'] = 'a'
        x2, y_pred2, y_real2 = load_dataset(settings, settings_file_path, data, path_model = path_pic + "Model_JS_DNN/")

        # plot together    
        plot_prediction_error_HNNvsDNN_JS(path_pic, x, y_pred, y_real, x2, y_pred2, y_real2)


if __name__ == "__main__":

    ####### GET CONFIG ###########
    settings_file_path= "./config_ANN.json"
    settings = load_json(settings_file_path)
    settings_dataset = load_json("./config_data.json")
    settings = {**settings_dataset, **settings}

    # Choose case
    # name = 'JS'
    name = 'asteroid'

    ####### GET DATA ###########
    # Only 1 of the 2 necessary
    # data = process_data(settings) # comment if not necessary. Only needed once
    data = get_data(settings, plot = False, name = name+'/') # Case with asteroids
    
    ####### TRAIN ##########
    if name == 'JS':
        settings['bodies'] = 2 # Case with SJS
    else:
        settings['bodies'] = 3 # Case with asteroid
        
    # train_DNN(settings, data, path_model ="./ANN_tf/"+ name +'/')
    # train_multiple(settings, data, 6)
    # autokeras(settings, data) # Check best parameters
    
    ####### PREDICT ########
    path_pic = "./ANN_tf/" + name +'/'
    # predict(path_pic, settings, settings_file_path, data)
    predict_HNNvsDNN(path_pic, settings, settings_file_path, data, name)


    

