# PlanetarySystem_HNN
## Hybrid method for solving the gravitational N-body problem with Artificial Neural Networks

**Description**

The repository contains the code to generate a dataset of planetary system trajectories that is fed into the trianing of a Hamiltonian Neural Network (HNN) and a regular Deep Neural Network (DNN).
The trained networks are then used for the integration of those planetary systems. 

Project in collaboration with Philipp Horn, Simon Portegies Zwart, Elena Sellentin, Barry Koren and, Maxwell X. Cai.

**Installation requirements**

The code in this repository requires the following packages: `abie`, `tensorflow`, and `matplotlib`. They can be installed easily with the following command:
    pip install abie tensorflow matplotlib


## Getting started 

Run: python generate_training_data.py: to generate the dataset with the parameters specified in config_data.json.

Run: python train_tf: to plot the training database, train the neural network, and plot the training results. 

Run: python hyperparameter_optimize: to perform hyperparameter optimization and plot the results.

Run: python plots_experiment: to run different integration experiments. 

## Summary of scripts

**Configuration files**
* config_data.json: settings for the generation of the dataset
* config_ANN.json: settings for the configuration of the ANNs and the training parameters

**Scripts**
* wh_generate_database.py: Wisdom-Holman integrator adapted for the generation of the dataset. 
* wh_tf_flag.py: Wisdom-Holman integrator adapted for the use of neural networks to substitute the calculation of the accelerations.
* generate_training_data.py: generate dataset of trajectories for Jupiter and Saturn, and asteroids (0 asteroids possible). 
* train_tf.py: 
    * process data for the creation of training and test datasets.
    * plot training and test datasets.
    * train neural network.
    * plot trained-network performance on test dataset.
* data.py: process data into train/test data
* nn_tensorflow.py: creation of tensorflow neural network (both for DNN and HNN), functions for training of the neural network.
* test_dataset.py: evaluate trained neural network on test dataset. Plot results comparing HNN and DNN.
* hyperparameter_optimize.py: 
    * hyperparameter optimization to find the best neural network architecture and learning parameters. 
    * hyperparameter optimization to find the value for the weights of the different variables in the loss function.
* plots_database.py: functions to plot different aspects of the training/test dataset
* plots_experiment.py: 
    1. Run evolution of planetary system using the Wisdom-Holman integrator, the DNN, and the HNN.    
    2. Test the difference between a pure integrator, a hybrid integrator, and an integrator with a HNN.
    3. Compare different R parameters for the hybrid integrator.
    4. Plot real / predicted accelerations of the different bodies in a simulation
    5. Compare interactive energy and the output of the HNN
    6. Compute and plot the prediction error
* plots_papers.py: functions to create plots for publication. 
* plot_tools.py: additional functions for plots
