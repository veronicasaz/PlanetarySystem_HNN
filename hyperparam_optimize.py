"""
Created: July 2021 
Last modified: October 2022 
Author: Veronica Saz Ulibarrena 
Description: Hyperparameter optimization, loss function weight optimization
"""
import numpy as np
import pandas as pd
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as ke
from IPython import embed
import time
from matplotlib import ticker
import matplotlib.colors as plc

# from plots import plot_predicted_orbit
import datetime
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from keras.utils.generic_utils import get_custom_objects
import sklearn.metrics as skl

import autokeras as ak

from train_tf import get_data
from nn_tensorflow import tanh_log, loss_mse
from data import get_dataset, get_traintest_data, load_json
from test_dataset import plot_prediction_error, load_dataset

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float64')

from plot_tools import color1, color2

# Activation function
def loss_weight_mse(x, y, y_pred, w):
    """
    loss_weight_mse: loss function to test different weights
    INPUTS:
        x: inputs
        y: real output
        y_pred: predicted output
        w: weights [w1, w2]. w3 is 1
    OUTPUTS:
    """
    w1, w2 = w
    w3 = 1

    samples = y.get_shape().as_list()[0]
    a1 = tf.slice(y, [0,0], [samples, 3]) 
    a1_pred = tf.slice(y_pred, [0,0], [samples, 3]) 
    a2 = tf.slice(y, [0,3], [samples, 3]) 
    a2_pred = tf.slice(y_pred, [0,3], [samples, 3]) 
    a3 = tf.slice(y, [0,6], [samples, 3]) 
    a3_pred = tf.slice(y_pred, [0,6], [samples, 3]) 
    return tf.reduce_mean((w1*tf.square(a1-a1_pred), w2*tf.square(a2-a2_pred), w3*tf.square(a3-a3_pred)))

class ANN(tf.keras.Model):
    def __init__(self, settings, params_net, w, path_model = None, path_std = None, restart = False, seed = None):
        """
        ANN: class for the neural network
        INPUTS:
            settings: configuration file 
            path_model: path to save/load trained model
            path_std: path for the standardization file
            seed: seed for initial random weight initialization
            restart: restart training using a created model as starting point
        """

        super(ANN, self).__init__()
                
        # Settings
        self.settings = settings
        self.params_net = params_net
        
        if path_model == None:
            self.path_model = self.settings['model_path']+'/'
        else:
            self.path_model = path_model

        self.path_std = path_std
        if self.path_std == None:
            self.path_std = './dataset/'

        if ('input_dim' in self.settings) == False:
            self.settings['input_dim'] = 4* self.settings['bodies'] 

        if self != None:
            self.seed = seed
        else: 
            self.seed = 5

        if restart == False:
            self.model = self.create_model()
        else:
            self.model = self.load_model_fromFile(path = self.path_model)

        # Learning rate and optimizer
        if self.settings['lr_constant'] == False:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                    params_net[4],
                                    decay_steps=self.params_net[5],
                                    decay_rate=self.params_net[6]
                                                )
            self.optimizer = ke.optimizers.Adam(learning_rate = lr_schedule, clipnorm=1.0)
        else:
            self.optimizer = ke.optimizers.Adam(learning_rate = self.settings['learning_rate'], clipnorm=1.0)

        # Loss function
        self.w = w
        self.loss = loss_weight_mse

    def create_model(self):
        """
        create_model: create neurons and weights
        OUTPUTS:
            model: created model
        """
        param_layer = int(self.params_net[1])
        param_neurons = int(self.params_net[2])
        param_neurdecay = self.params_net[3]

        n_inputs = self.settings['input_dim']
        if self.settings['output_dim'] =='a':
            n_outputs = n_inputs // 4 * 3
        elif self.settings['output_dim'] =='H':
            n_outputs = 1 

        inputs = ke.Input(shape=(n_inputs,), name = 'input')
        hidden_layer = inputs
        get_custom_objects().update({'tanh_log': tanh_log})

        if not isinstance(param_neurons, list):
            neurons = [param_neurons] * self.settings['layer']
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * param_layer
            #TODO: test of mixed activations
            if self.settings['activations2'] != False:
                self.settings['activations'][0:param_layer//2] = [self.settings['activations2']]*  (param_layer //2)
        if not isinstance(self.settings['weights'], list):
            self.settings['weights'] = [tf.keras.initializers.glorot_normal(seed = self.seed) ] * param_layer
            
        for i, (neuron, activation, weights) in enumerate(zip(neurons, self.settings['activations'], self.settings['weights']), start = 1):

            neuron_updated = neuron * param_neurdecay **(i-1)
            hidden_layer = ke.layers.Dense(neuron_updated, activation = activation,
                                           name = 'hidden_' + str(i),
                                           kernel_initializer =  weights)(hidden_layer)
            
            hidden_layer = ke.layers.Dropout(rate = self.settings['dropout'])(hidden_layer)

        output_layer = ke.layers.Dense(n_outputs, activation = None,
                                       name = 'output')(hidden_layer)
        model = ke.Model(inputs = inputs, outputs = output_layer)
        
        self.model = model

        self.model.summary()

        return model

    @tf.function
    def time_derivative(self, dH, x):       
        """
        time_derivative: time_derivative for HNN
        INPUTS: 
            dH: derivative of output with respect to inputs
            x: inputs
        OUTPUTS:
            y: accelerations
        """    
        # flip first 3 rows with last 3 rows
        n_inputs = dH.get_shape().as_list()[1]
        n_samples = dH.get_shape().as_list()[0]
        I_np = np.eye(n_inputs)

        mass_idx = np.arange(0, n_inputs, 4) # If mass of Jup and Saturn included
        I_np = np.delete(I_np, mass_idx, 0)

        I_m_np = np.zeros((n_inputs//4*3, n_inputs))
        for i in range(n_inputs//4):
            I_m_np[i*3:i*3+3, i*4] = np.ones(3)

        I_np = - I_np

        I = tf.convert_to_tensor(I_np, dtype=tf.dtypes.float64, dtype_hint=None, name=None)
        I_m = tf.convert_to_tensor(I_m_np, dtype=tf.dtypes.float64, dtype_hint=None, name=None)

        # Select the terms from dH that correspond to the acceleration
        y = tf.linalg.matvec(I, dH, transpose_a=False, adjoint_a=False, a_is_sparse=False, b_is_sparse=False,\
            name=None)

        # Get vector [m1, m1, m1, m2, m2, m2, m3, m3, m3]
        M = tf.linalg.matvec(I_m, x, transpose_a=False, adjoint_a=False, a_is_sparse=False, b_is_sparse=False,\
            name=None)
        y = tf.divide(y, M)

        return y

    @tf.function
    def train_step(self, x, y):
        """
        train_step: evaluate loss and upgrade weights
        INPUTS: 
            x: inputs
            y: real output
        OUTPUTS:
            loss_value: loss at this step
        https://stackoverflow.com/questions/65058699/using-gradients-in-a-custom-loss-function-tensorflowkeras
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            H = self.model(x, training=True)

            if self.settings['output_dim'] == "H" and self.settings['loss_variable'] == 'dI':
                dH = g.gradient(H, x)
                y_pred = self.time_derivative(dH, x)
            else:
                y_pred = H

            loss_value = self.loss(x, y, y_pred, self.w)
        
        # Train network
        grads = g.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        del g # delete g after used

        return loss_value
    
    @tf.function
    def test_step(self, x, y):
        """
        test_step: get validation loss for a time step
        INPUTS: 
            x: inputs of the network
            y: real output
        OUTPUTS:
            loss_value: loss at this step 
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            H = self.model(x, training=False)

            if self.settings['output_dim'] == "H":
                dH = g.gradient(H, x)
                y_pred = self.time_derivative(dH, x)
            else:
                y_pred = H

            loss_value = self.loss(x, y, y_pred, self.w)
        del g
        return loss_value

            
    def train(self, data):
        """
        train: main training function. Training loop
        INPUTS: 
            data: training dataset
        """
        features = data['coords']
        labels = data['dcoords']
        # Save history
        self.history = dict()
        self.history['loss'] = np.zeros(self.settings['max_epochs'])
        self.history['val_loss'] = np.zeros(self.settings['max_epochs'])

        # Separate into train and validation and create dataset
        size_val = int( (1-self.settings['validation_split']) * features.shape[0] )

        features_train = tf.convert_to_tensor(features[0:size_val,:], dtype= 'float64')
        labels_train = tf.convert_to_tensor(labels[0:size_val,:], dtype= 'float64')
        features_val = tf.convert_to_tensor(features[size_val:,:], dtype= 'float64')
        labels_val = tf.convert_to_tensor(labels[size_val:,:], dtype= 'float64')

        if not self.settings['batch_size']:
            self.settings['batch_size'] = int( (1-self.settings['validation_split']) * features.shape[0])
        elif self.settings['batch_size'] == -1:
            self.settings['batch_size'] = int((1-self.settings['validation_split']) * features.shape[0])

        time_0 = time.time()
        prev_time = time_0
        for ep in range(1, self.settings['max_epochs']+1):
            
            train_data = tf.data.Dataset.from_tensor_slices((features_train, labels_train)).shuffle(buffer_size=np.shape(features_train)[0]).batch(self.settings['batch_size'])
            val_data = tf.data.Dataset.from_tensor_slices((features_val, labels_val)).shuffle(buffer_size=np.shape(features_val)[0]).batch(np.shape(features_val)[0])

            loss_b = 0
            for batch, (X, y) in enumerate(train_data):
                loss_b += self.train_step(X, y)
            self.history['loss'][ep-1] = loss_b / (batch+1)# average of losses of batches
                
            # Validation round
            loss_val = 0
            for batch, (X, y) in enumerate(val_data):
                loss_val += self.test_step(X, y)

            self.history['val_loss'][ep-1] = self.test_step(X, y)

            time_i = time.time()
            # Print epoch result
            print("Epoch", ep,
                " Time/epoch: %0.3E, total time: %0.3E. Loss: %0.4E, val_loss: %0.4E"%(
                        time_i - prev_time,
                        time_i - time_0,
                        self.history['loss'][ep-1],
                        self.history['val_loss'][ep-1]
                ))
            prev_time = time_i

            if ep % 20 == 0:
                self.model.save(self.path_model + "model.h5") # Save model every 100 iterations in case I need to stop early
                self.saveTraining()

            if ep > 100 and ep % 20 == 0 :
                if np.mean(self.history['val_loss'][ep-20:ep]) >= 1.2*np.mean(self.history['val_loss'][ep-40:ep-20]): # overfitting
                    print("Overfitting")
                    break
                elif abs(np.mean(self.history['loss'][ep-40:ep]) - np.mean(self.history['loss'][ep-90:ep-40]))/ np.mean(self.history['loss'][ep-90:ep-40]) < 0.01: # no improvement
                    print("No improvement")
                    break
                
        self.model.save(self.path_model + "model.h5")


    def saveTraining(self, path= None):
        """
        saveTraining: save training progress
        INPUTS:
            path: path where to save
        """
        if path == None:
            path = self.path_model
        path = path +"training.txt"
        trainloss = self.history['loss']
        valloss = self.history['val_loss']
        vector = np.array([trainloss, valloss])

        np.savetxt(path, vector)

    def plotTraining(self, path_figure, path_training_process = None):
        """
        plotTraining: plot training progress
        INPUTS:
            path_figure: path where to save figure
            path_training_process: path where training data is saved
        """
        colors = ['r-.','g-.','k-.','b-.','r-.','g-.','k-.','b-.','r-','g-','k-','b-','r--','g--','k.-','b.-']
        
        if path_training_process:
            data = np.loadtxt(path_training_process +"training.txt")
            train_loss = data[0,:]
            val_loss = data[1,:]
        else:
            train_loss = self.history['loss']
            val_loss = self.history['val_loss']

        f = plt.figure()
        ax = f.add_subplot(111)
        plt.plot(train_loss)
        plt.plot(val_loss)

        text = "Train loss = %e\n Validation loss = %e"%(train_loss[-1], val_loss[-1]) 
        plt.text(0.5, 0.7, text, horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)

        plt.title('Model loss')
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.grid(alpha = 0.5)
        plt.tight_layout()
        plt.savefig(path_figure+"trainingloss.png", dpi = 100)
        # plt.show()

    def load_model_fromFile(self, path= None):
        """
        load_model_fromFile: load saved model
        INPUTS: 
            path: if given, path where model is stored
        OUTPUTS:
            model: loaded model
        """
        if path == None:
            path = self.path_model
        print(path +'model.h5')
        model = ke.models.load_model(path +'model.h5', custom_objects={'tanh_log': tf.keras.layers.Activation(tanh_log), 'sin': tf.keras.layers.Activation(sin)})

        self.model = model
        return model

    def predict(self, inp, path_std = None, std = True):
        """
        predict: Predict accelerations when output is H
        INPUTS:
            inp: inputs to the network
            path_std: path of standardization file
            std: if True, apply standardization
        OUTPUTS:
            predictions: output of the network, accelerations
        """
        if inp.ndim == 1:
            inp = np.expand_dims(inp, axis=0)
            
        x = tf.convert_to_tensor(inp, dtype= 'float64')   

        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            H = self.model(x, training=False)
        
        if self.settings['output_dim'] == "H" and self.settings['loss_variable'] == 'dI':
            dH = g.gradient(H, x)
            y_pred = self.time_derivative(dH, x)
        else:
            y_pred = H
        del g

        return y_pred


def optimize(data, config, params, N):
    """
    optimize: start training for each of the parameter configurations and save
    INPUTS: 
        data: training and test dataset
        config: settings
        params: hyperparameters to optimize
        N: number of experiments to carry out
    """
    # Create combinations of parameters
    combi = np.zeros((N, len(params)))
    for j in range(len(params)):
        idx = np.random.randint(0, len(params[j]), size = N)
        for r in range(N):
            combi[r, j] = params[j][idx[r]]

    np.savetxt("./ANN_tf/asteroid/optim/params.txt", combi)

    for net in range(N):
        params_net = combi[net, :]
        print(params_net)

        data['coords'] = data['coords'][0:int(params_net[0]), :] # select number of samples
        data['dcoords'] = data['dcoords'][0:int(params_net[0]), :] # select number of samples

        print("=============================================================")
        print("Network number: ", net)
        print("=============================================================")
        ANN_tf = ANN(config, params_net, [1,1], path_model = "./ANN_tf/asteroid/optim/net_"+str(net+1)+"_")

        ANN_tf.train(data)
        ANN_tf.saveTraining()
        ANN_tf.plotTraining("./ANN_tf/asteroid/optim/net_"+str(net+1)+"_", \
                            path_training_process = "./ANN_tf/asteroid/optim/net_"+str(net+1)+'_')

        x, y_pred, y_real = load_dataset(config, "./config_ANN.json", data, \
                            path_model = "./ANN_tf/asteroid/optim/net_"+str(net+1)+"_")
        plot_prediction_error("./ANN_tf/asteroid/optim/net_"+str(net+1)+'_', x, y_pred, y_real)

def optimize_w(data, config, w, N):
    """
    optimize: start training for each of the combinations of the weight for the loss function
    INPUTS: 
        data: training and test dataset
        config: settings
        w: weights to optimize
        N: number of experiments to carry out
    OUTPUTS:
        mse_test: mean squared error of the test dataset
    """
    # Create combinations of w
    combi = np.zeros((N, len(w)))
    for j in range(len(w)):
        idx = np.random.randint(0, len(w[j]), size = N)
        for r in range(N):
            combi[r, j] = w[j][idx[r]]

    np.savetxt("./ANN_tf/trained_nets/optim_w/weights.txt", combi)
    params_net = [50000, 2, 100, 0.6, 0.001, 50000, 0.9]
    mse_test = np.zeros(N)
    for net in range(N):
        w = combi[net, :]

        data['coords'] = data['coords'][0:int(params_net[0]), :] # select number of samples
        data['dcoords'] = data['dcoords'][0:int(params_net[0]), :] # select number of samples

        print("=============================================================")
        print("Network number: ", net)
        print("=============================================================")
        ANN_tf = ANN(config, params_net, w, path_model = "./ANN_tf/asteroid/optim_w/net_"+str(net+1)+"_")

        ANN_tf.train(data)
        ANN_tf.saveTraining()
        ANN_tf.plotTraining("./ANN_tf/trained_nets/optim_w/net_"+str(net+1)+"_", \
                            path_training_process = "./ANN_tf/trained_nets/optim_w/net_"+str(net+1)+'_')
        x, y_pred, y_real = load_dataset(config, "./config_ANN.json", data, \
                            path_model = "./ANN_tf/asteroid/optim_w/net_"+str(net+1)+"_")
        plot_prediction_error("./ANN_tf/trained_nets/optim_w/net_"+str(net+1)+'_', x, y_pred, y_real)
        mse_test[net] = skl.mean_squared_error(y_real/y_real, y_pred/y_real)
    return mse_test

def plot_optim(data, N):
    """
    plot_optim: plot results of hyperparameter optimization
    INPUTS: 
        data: training and test dataset
        N: number of experiments to carry out
    """
    params = np.loadtxt("./ANN_tf/asteroid/optim/params.txt")
    loss = np.zeros((N, 2))

    for net in range(N):
        data_t = np.loadtxt("./ANN_tf/asteroid/optim/net_"+str(net+1)+"_" +"training.txt")
        loss[net, 0] = data_t[0, -1]
        loss[net, 1] = data_t[1, -1]

    D_samples = np.hstack((loss, params[0:N,:]))
    
    subplot2 = 1
    subplot1 = 1
    fig, ax = plt.subplots(subplot1, subplot2, figsize=(12,6))
    
    plt.subplot(subplot1, subplot2, 1)
    index = np.where(D_samples[:, 0] == np.min(D_samples[:, 0]))[0]
    index_val = np.where(D_samples[:, 1] == np.min(D_samples[:, 1]))[0]
    plt.scatter(D_samples[:, 0], D_samples[:, 1], s = 150, marker = 'o' , color = color2[1])
    
    label1 = "Samples: %i \nLayers: %i \nNeurons: %i \nRatio of neurons: %0.1f\n"%(D_samples[index, 2],\
        D_samples[index, 3], D_samples[index, 4], D_samples[index, 5])
    label2 = "Samples: %i \nLayers: %i \nNeurons: %i \nRatio of neurons: %0.1f"%(D_samples[index_val, 2],\
        D_samples[index_val, 3], D_samples[index_val, 4], D_samples[index_val, 5])
    plt.scatter(D_samples[index, 0], D_samples[index, 1], s = 250, color = color2[0], marker = 's' , label = label1)
    plt.scatter(D_samples[index_val, 0], D_samples[index_val, 1], s = 250, color =color2[2], marker = 's' , label = label2)

    # plt.title('Results of hyperparameter optimization', fontsize = 22)
    legend = plt.legend(fontsize = 26, bbox_to_anchor=(1.0, 1.0))
    plt.grid(alpha = 0.5)
    # plt.axis('equal')

    plt.xlabel("Training loss", fontsize = 30)
    plt.ylabel("Validation loss", fontsize = 30)
    plt.xticks(fontsize = 28)
    plt.yticks(fontsize = 28)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./ANN_tf/asteroid/optim/"+'Optim.png', dpi = 100)
    plt.show()

def plot_optim_w(data, N):
    """
    plot_optim_w: plot results of weight optimization
    INPUTS: 
        data: training and test dataset
        N: number of experiments to carry out
    """
    w = np.loadtxt("./ANN_tf/asteroid/optim_w/weights.txt")
    loss = np.zeros((N, 2))
    for net in range(N):
        data_t = np.loadtxt("./ANN_tf/asteroid/optim_w/net_"+str(net+1)+"_" +"training.txt")
        loss[net, 0] = data_t[0, -1]
        loss[net, 1] = data_t[1, -1]

    D_samples = np.hstack((loss, w))
    
    subplot2 = 1
    subplot1 = 1
    fig, ax = plt.subplots(subplot1, subplot2, figsize=(15,15))
    
    plt.subplot(subplot1, subplot2, 1)
    index = np.where(D_samples[:, 0] == np.min(D_samples[:, 0]))[0]
    index_val = np.where(D_samples[:, 1] == np.min(D_samples[:, 1]))[0]
    plt.scatter(D_samples[:, 0], D_samples[:, 1], s = 100, marker = 'o' )
    plt.scatter(D_samples[index, 0], D_samples[index, 1], s = 150, color = 'red', marker = 's' , label = str(index[0]+1) +str(w[index, :]))
    plt.scatter(D_samples[index_val, 0], D_samples[index_val, 1], s = 150, color = 'blue', marker = 's' , label = str(index_val[0]+1) +str(w[index_val, :]))

    for i in range(N):
        plt.annotate("%i, %i"%(D_samples[i, 2], D_samples[i, 3]), xy=(D_samples[i, 0]*1.02, D_samples[i, 1]*0.98))

    plt.legend()
    plt.grid(alpha = 0.5)

    plt.xlabel("Train loss")
    plt.ylabel("Validation loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./ANN_tf/asteroid/optim_w/"+'Optim_w.png', dpi = 100)
    plt.show()

def plot_mse(data, N, mse):
    """
    plot_mse: plot mse for each weight combination
    INPUTS: 
        data: training and test dataset
        N: number of experiments to carry out
        mse: mean squared error for each combination
    """
    w = np.loadtxt("./ANN_tf/asteroid/optim_w/weights.txt")
    subplot2 = 1
    subplot1 = 1
    fig, ax = plt.subplots(subplot1, subplot2, figsize=(15,15))
    
    plt.subplot(subplot1, subplot2, 1)
    cm = plt.cm.get_cmap('RdYlBu')

    sc = plt.scatter(w[:,0], w[:,1], s = 100, marker = 'o', c = mse, cmap=cm, norm=plc.LogNorm())
    cbar = plt.colorbar(sc)
    cbar.set_label('mean((y_pred - y_real) / y_real)^2', rotation=90)
    plt.legend()
    plt.grid(alpha = 0.5)

    plt.xlabel("W1")
    plt.ylabel("W2")
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./ANN_tf/asteroid/optim_w/"+'Optim_w_mse.png', dpi = 100)
    plt.show()

if __name__ == "__main__":
    ####### GET CONFIG ###########
    settings_file_path= "./config_ANN.json"
    settings = load_json(settings_file_path)
    settings_dataset = load_json("./config_data.json")
    settings = {**settings_dataset, **settings}

    settings['max_epochs'] = 2000
    data = get_data(settings, plot = False, name = 'asteroid/')

    layers = [2, 3, 4, 5]
    samples = [10000, 50000, 10000, 200000, 250000]
    neurons = [100, 200, 300, 500, 700]
    ratio_neurons = [0.6]
    lr_0 = [1e-3]
    lr_steps = [5e5]
    lr_decay = [0.9]

    N = 33

    params = [samples, layers, neurons, ratio_neurons, lr_0, lr_steps,\
        lr_decay]

    ###################################
    # Hyperparameter optimization
    ###################################
    optimize(data, settings, params, N)
    plot_optim(data, N)

    ###################################
    # Loss weight optimization
    ###################################
    w1 = [1, 10, 20, 50, 100, 1000]
    w2 = [1, 10, 20, 50, 100, 1000]
    w = [w1, w2]

    mse = optimize_w(data, settings, w, N)
    plot_optim_w(data, N)
    plot_mse(data, N, mse)