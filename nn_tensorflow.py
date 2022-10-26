"""
Created: July 2021 
Last modified: July 2022 
Author: Veronica Saz Ulibarrena 
Description: Create, train ANNs and predict with them
"""
import numpy as np
import pandas as pd
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as ke
import time
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects
import autokeras as ak

from data import standardize, rot_invariance, rot_invariance_inverse

tf.random.set_seed(12) # keep track of seed
tf.keras.backend.set_floatx('float64')

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

color1 = ['navy', 'dodgerblue','darkorange']
color2 = ['dodgerblue', 'navy', 'orangered', 'green', 'olivedrab',  'saddlebrown', 'darkorange', 'red' ]

# Activation functions
def tanh_log(x):
    """tanh_log: Simlog activation. Keras activation function. Created by Maxwell X. Cai """
    return tf.keras.activations.tanh(x) * tf.keras.backend.log(tf.keras.activations.tanh(x) * x + 1)

def sin(x):
    """sin: sin keras activation function """
    return tf.keras.backend.sin(x) 

def plot_tanh_log():
    """
    plot_tanh_log: plot comparison activation functions
    """
    xi = np.logspace(-3, 3, 100)
    x = np.concatenate((np.flip(-xi), xi))
    y1 = np.tanh(x) * np.log(np.tanh(x) *x +1)
    y2 = np.tanh(x)
    y3 = np.copy(x)
    y3[0:len(y3)//2-1] *= 0
    y4 = np.tanh(x) * x
    y5 = x + np.sin(x)**2
    f, ax = plt.subplots(figsize = (14,4),nrows=1, ncols=1)
    lw = 4
    plt.plot(x, y1, '-.', linewidth = lw,  label =r'SymmetricLog: f(x) = $tanh(x)\; \cdot log(tanh(x) \cdot x+1)$', color = color2[1])
    plt.plot(x, y2, linewidth = lw, label ='tanh: f(x) = $tanh(x)$', color = color2[0])
    plt.plot(x, y3, '--', linewidth = lw, label ='ReLU: f(x) = $max(0, z$)', color = color2[2])

    plt.xlabel('x', fontsize= 23)
    plt.ylabel('f(x)', fontsize= 23)
    # plt.title("Activation functions", fontsize= 20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    # plt.legend(fontsize = 18, loc= 'upper left')
    plt.legend(fontsize = 20, bbox_to_anchor=(1.0, 1.0) )
    
    plt.tight_layout()
    # plt.grid(alpha = 0.5)
    plt.savefig("./ANN_tf/tanh_log.png", dpi = 100)
    plt.show()

# plot_tanh_log()

# Loss functions
def loss_mse(x, y, y_pred):
    """
    loss_mse: MSE loss function
    INPUTS: 
        x: inputs
        y: real output
        y_pred: predicted output
    OUTPUT: loss function
    """
    return tf.reduce_mean( tf.square(y - y_pred) ) # 1e5 to avoid numerical problems with small numbers

def loss_asteroid(x, y, y_pred):
    """
    loss_asteroid: MSE loss function only for the asteroid
    INPUTS: 
        x: inputs
        y: real output
        y_pred: predicted output
    OUTPUT: loss function
    """
    samples = y.get_shape().as_list()[0]
    a3 = tf.slice(y, [0,6], [samples, 3]) 
    a3_pred = tf.slice(y_pred, [0,6], [samples, 3]) 
    return tf.reduce_mean( tf.square(a3 - a3_pred) )

def loss_weight_mse(x, y, y_pred):
    """
    loss_weight_mse: weighted MSE loss function
    INPUTS: 
        x: inputs
        y: real output
        y_pred: predicted output
    OUTPUT: loss function
    """
    w1 = 1
    w2 = 1
    w3 = 1

    samples = y.get_shape().as_list()[0]
    a1 = tf.slice(y, [0,0], [samples, 3]) 
    a1_pred = tf.slice(y_pred, [0,0], [samples, 3]) 
    a2 = tf.slice(y, [0,3], [samples, 3]) 
    a2_pred = tf.slice(y_pred, [0,3], [samples, 3]) 
    a3 = tf.slice(y, [0,6], [samples, 3]) 
    a3_pred = tf.slice(y_pred, [0,6], [samples, 3]) 
    return tf.reduce_mean((w1*tf.square(a1-a1_pred), w2*tf.square(a2-a2_pred), w3*tf.square(a3-a3_pred)))

def loss_mse_loga(x, y, y_pred):
    """
    loss_mse_loga: log of difference 
    INPUTS: 
        x: inputs
        y: real output
        y_pred: predicted output
    OUTPUT: loss function
    """
    acc_loss = tf.keras.backend.log(tf.math.abs(y_pred - y))
    return tf.reduce_mean(acc_loss)


def plot_loss():
    """
    plot_loss: plot loss functions for a certain range
    """
    x1 = np.logspace(-7, 2, num=50)
    x2 = np.logspace(-7, 2, num=50)

    y = np.log(np.tanh(x1) * x1 + 1)
    y2 = np.log(np.tanh(-x1) * (-x1) + 1)

    fig,ax=plt.subplots(1,1)
    ax.plot(x1, y, marker = 'x')
    ax.plot(-x1, y2, marker = 'x')
    ax.set_title('Filled Contours Plot')
    ax.set_ylabel('y (cm)')
    plt.show()

# plot_loss()

class ANN(tf.keras.Model):
    def __init__(self, settings, path_model = None, path_std = None, restart = False, seed = None, std = False, pred_type = 'dI'):
        """
        ANN: class for the neural network
        INPUTS:
            settings: configuration file 
            path_model: path to save/load trained model
            path_std: path for the standardization file
            seed: seed for initial random weight initialization
        """
        super(ANN, self).__init__()

        # Settings
        self.settings = settings
        
        if path_model == None:
            self.path_model = self.settings['model_path']+'/'
        else:
            self.path_model = path_model

        self.std = std
        self.path_std = path_std
        if self.path_std == None:
            self.path_std = './dataset/'

        self.pred_type = pred_type

        # Input dimension
        if ('input_dim' in self.settings) == False:
            self.settings['input_dim'] =  self.settings['bodies']*4 # mass + x, y, z

        if self != None:
            self.seed = seed
        else: 
            self.seed = 50
        
        self.model = self.create_model()

        # Learning rate and optimizer
        if self.settings['lr_constant'] == False:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                    self.settings['learning_rate'],
                                    decay_steps=self.settings['learning_rate_steps'],
                                    decay_rate=self.settings['learning_rate_decay']
                                                )
            self.optimizer = ke.optimizers.Adam(learning_rate = lr_schedule, clipnorm=1.0)
        else:
            self.optimizer = ke.optimizers.Adam(learning_rate = self.settings['learning_rate'], clipnorm=1.0)

        # Loss function
        if self.settings['loss'] == 'loss_mse':
            self.loss = loss_mse
        elif self.settings['loss'] == 'log':
            self.loss = tf.keras.losses.MeanSquaredLogarithmicError()
        elif self.settings['loss'] == 'loss_mse_log':
            self.loss = loss_mse_log
        elif self.settings['loss'] == 'loss_weight_mse':
            self.loss = loss_weight_mse
        elif self.settings['loss'] == 'loss_asteroid':
            self.loss = loss_asteroid

    def create_model(self):
        """
        create_model: create neurons and weights
        OUTPUTS:
            model: created model
        """
        n_inputs = self.settings['input_dim']
        if self.settings['output_dim'] =='a':
            n_outputs = n_inputs // 4 * 3
        elif self.settings['output_dim'] =='H':
            n_outputs = 1 
            
        inputs = ke.Input(shape=(n_inputs,), name = 'input')
        hidden_layer = inputs
        get_custom_objects().update({'tanh_log': tanh_log, 'sin':sin})

        if not isinstance(self.settings['neurons'], list):
            self.settings['neurons'] = [self.settings['neurons']] * self.settings['layer']
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * self.settings['layer']
            # Mix different activation functions
            if self.settings['activations2'] != False:
                self.settings['activations'][0:self.settings['layer']//2] = [self.settings['activations2']]*  (self.settings['layer'] //2)
        if not isinstance(self.settings['weights'], list):
            self.settings['weights'] = [tf.keras.initializers.glorot_normal(seed = self.seed) ] * self.settings['layer']
            
        # hidden_layer = ke.layers.BatchNormalization(momentum=0.99)(hidden_layer)
        for i, (neuron, activation, weights) in enumerate(zip(self.settings['neurons'], self.settings['activations'], self.settings['weights']), start = 1):

            neuron_updated = neuron * self.settings['neurons_ratio'] **(i-1)
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
        n_inputs = dH.get_shape().as_list()[1]
        n_samples = dH.get_shape().as_list()[0]
        I_np = np.eye(n_inputs)

        mass_idx = np.arange(0, n_inputs, 4) # If mass of Jup and Saturn included
        I_np = np.delete(I_np, mass_idx, 0)
        I_np = - I_np 

        # Matrix to divide by masses
        I_m_np = np.zeros((n_inputs//4*3, n_inputs))
        for i in range(n_inputs//4):
            I_m_np[i*3:i*3+3, i*4] = np.ones(3)

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
            loss_value = self.loss(x, y, y_pred)

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
            loss_value = self.loss(x, y, y_pred)
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

        # Save training history
        self.history = dict()
        self.history['loss'] = np.zeros(self.settings['max_epochs'])
        self.history['val_loss'] = np.zeros(self.settings['max_epochs'])

        # Separate into train and validation and create dataset
        size_val = int( (1-self.settings['validation_split']) * features.shape[0] )
        features_train = tf.convert_to_tensor(features[0:size_val,:], dtype= 'float64')
        labels_train = tf.convert_to_tensor(labels[0:size_val,:], dtype= 'float64')
        features_val = tf.convert_to_tensor(features[size_val:,:], dtype= 'float64')
        labels_val = tf.convert_to_tensor(labels[size_val:,:], dtype= 'float64')

        # Define batch size. If not given, use the full dataset
        if not self.settings['batch_size']:
            self.settings['batch_size'] = int( (1-self.settings['validation_split']) * features.shape[0])
        elif self.settings['batch_size'] == -1:
            self.settings['batch_size'] = int((1-self.settings['validation_split']) * features.shape[0])

        time_0 = time.time() # Evaluate training time
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
                        self.history['val_loss'][ep-1] ))
            
            prev_time = time_i
            if ep % 20 == 0: # Save training information every 20 steps
                self.model.save_weights(self.path_model +'my_model_weights.h5')
                self.model.save(self.path_model + "model.h5") # Save model every 100 iterations in case I need to stop early
                self.saveTraining()
                
        self.model.save(self.path_model + "model.h5")
        self.model.save_weights(self.path_model +'my_model_weights.h5')

    def saveTraining(self, path= None):
        """
        saveTraining: save training progress
        INPUTS:
            path: path where to save
        """
        if path == None:
            path = self.path_model
        # path = path +"training_tanh.txt"
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
        plt.xlabel('Epoch')
        plt.legend(['Train loss', 'Validation loss'], loc='upper right')
        plt.grid(alpha = 0.5)
        plt.tight_layout()
        plt.savefig(path_figure+"trainingloss.png", dpi = 100)
        plt.show()

    def plotWeights(self):
        """
        plot_weights: plot evolution of small part of the weights to verify they change
        """
        self.model.load_weights(self.path_model + './weights/my_checkpoint_1_0')
        batches = int(self.settings['train_samples']*(1-self.settings['validation_split'])//self.settings['batch_size'])
        W = np.zeros((self.settings['max_epochs']*batches, 4380))
        for ep in range(1, self.settings['max_epochs']+1):
            for batch in range(batches):
                self.model.load_weights(self.path_model + 'weights/my_checkpoint_%i_%i'%(ep, batch))
                # Load weights and append for each layer
                w = np.array([0])
                for l in np.arange(1, 6, 2):
                    w = np.append(w, self.model.layers[l].get_weights()[0].flatten()) # to not take bias
                W[(ep-1)*batches+batch] = w[1:]

        eps = np.arange(0, np.shape(W)[0], 1)
        
        f = plt.figure()
        ax = f.add_subplot(111)

        samples = np.random.randint(low=0.0, high= np.shape(W)[1], size=20)
        for sample in samples:
            plt.plot(eps, W[:, sample])
        
        plt.title('Weights')
        plt.ylabel("Weights")
        plt.xlabel('Step')
        plt.grid(alpha = 0.5)
        plt.tight_layout()
        plt.savefig(self.path_model+"weights/W.png", dpi = 100)
        plt.show()
        
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
        self.model = ke.models.load_model(path +'model.h5', custom_objects={'tanh_log': tf.keras.layers.Activation(tanh_log), 'sin': tf.keras.layers.Activation(sin)})
        return self.model

    def load_weights(self, path = None):
        """
        load_weights: load weights an put them into the model
        INPUTS: 
            path: if given, path where model is stored
        OUTPUTS:
            model: loaded model
        """
        if path == None:
            path = self.path_model
        print(path +'model.h5')
        self.model = self.create_model()
        self.model.load_weights(self.path_model +'my_model_weights.h5')
        return self.model

    def predict(self, inp, path_std = None, std = False):
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
            
        if path_std == None:
            path_std = self.path_std

        # Standardize if necessary
        if self.settings['standardize']  == True:
            inp_std = standardize(inp, path_std, inverse = False, typeI = "I")
        else:
            inp_std = inp

        if std == True: # alternative standardization (simple case)
            inp[:, 8] *= 1e8

        x = tf.convert_to_tensor(inp, dtype= 'float64')   

        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            H = self.model(x, training=False)
        
        if self.pred_type == 'dI':
            dH = g.gradient(H, x)
            y_pred = self.time_derivative(dH, x)
        elif self.pred_type == 'H' or self.pred_type == 'a':
            y_pred = H
        elif self.settings['output_dim'] == "H" and self.settings['loss_variable'] == 'dI':
            dH = g.gradient(H, x)
            y_pred = self.time_derivative(dH, x)
        else:
            y_pred = H
        del g

        if self.settings['standardize']== True:        
            # unstandardize
            predictions = standardize(y_pred, path_std, inverse = True, typeI = "O")
        else:
            predictions = y_pred

        return predictions

    def drdv(self, inp, path_std = None):        
        """
        predict: Predict accelerations when output is drdv
        INPUTS:
            inp: inputs to the network
            path_std: path of standardization file
        OUTPUTS:
            predictions: output of the network, accelerations
        """
        # self.model = self.load_model_fromFile(self.path_model)
        if path_std == None:
            path_std = self.path_std

        # Standardize if necessary
        if inp.ndim == 1:
            inp = np.expand_dims(inp, axis=0)

        if self.settings['standardize']  == True:
            inp_std = standardize(inp, path_std, inverse = False, typeI = "I")
        else:
            inp_std = inp

        # Use model to predict output
        x = tf.convert_to_tensor(inp_std, dtype= 'float64')        
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            H = self.model(x, training=True)

        if self.settings['output_dim'] == "H":
            dH = g.gradient(H, x)
            y_pred, M = self.time_derivative(dH)
        else:
            y_pred = H
        del g

        if self.settings['standardize']== True:        
            # unstandardize
            predictions = standardize(y_pred, path_std, inverse = True, typeI = "O")
        else:
            predictions = y_pred

        return predictions

    def train_autokeras(self, data):
        """
        train_autokeras: use autokeras to find best architecture
        INPUTS: 
            data: training dataset
        """
        features = data['coords']
        labels = data['dcoords']
        train_set = tf.data.Dataset.from_tensor_slices((data['coords'], data['dcoords']))
        test_set = tf.data.Dataset.from_tensor_slices((data['test_coords'], data['test_dcoords']))

        reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
        # Feed the tensorflow Dataset to the regressor.
        reg.fit(train_set, epochs=10)
        # Predict with the best model.
        predicted_y = reg.predict(test_set)
        # Evaluate the best model with testing data.
        print(reg.evaluate(test_set))

        self.model = reg.export_model()
        self.model.save(self.path_model + "model_autokeras", save_format="tf")
        self.model.summary()
        
    def load_model_ak(self, path_model):
        """
        load_model_ak: load model created by autokeras
        INPUTS: 
            path_model: path where the model is stored
        """
        self.model = tf.keras.models.load_model(path_model+"model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
        self.model.summary()
        
def predict_multiple(nets, inp):
    """
    predict_multiple: create mixed prediction using many networks
    INPUTS:
        nets: list with networks
        inp: input to the network
    OUTPUTS:
        accel_mean: output result
        IQR: evaluation of dispersion 
    """
    if inp.ndim == 1:
        inp = np.expand_dims(inp, axis=0)

    bodies = np.shape(inp)[1]//4

    accel = np.zeros(( len(nets), np.shape(inp)[0], bodies*3 ))
    for net in range(len(nets)):
        accel[net, :, :] = nets[net].predict(inp)


    # These two don't deal well with outliers    
    # accel_mean = np.mean(accel, axis = 0)
    # accel_std = np.std(accel, axis = 0)

    # Quantile deals better with outliers
    accel_mean = np.quantile(accel, 0.5, axis = 0)
    IQR = 1.0 * np.quantile(accel, 0.75, axis = 0) - np.quantile(accel, 0.25, axis = 0)
    # accel_max = np.quantile(accel, 0.75, axis = 0) + IQR
    # accel_min = np.quantile(accel, 0.25, axis = 0) - IQR
        
    return accel_mean, IQR
