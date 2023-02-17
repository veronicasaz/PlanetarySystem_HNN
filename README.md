# PlanetarySystem_HNN
## Hybrid method for solving the gravitational N-body problem with Artificial Neural Networks

**Description**

The repository contains the code to generate a dataset of planetary system trajectories that is fed into the trianing of a Hamiltonian Neural Network (HNN) and a regular Deep Neural Network (DNN).
The trained networks are then used for the integration of those planetary systems. 

**Installation requirements**
The code in this repository requires the following packages: `abie`, `tensorflow`, and `matplotlib`. They can be installed easily with the following command:
    pip install abie tensorflow matplotlib


## Getting started 

Run: python generate_training_data.py: to generate the dataset with the parameters specified in config_data.json.

Run: python train_tf: to plot the training database, train the neural network, and plot the training results. 
## Summary of scripts
**Configuration files**
* config_data.json: settings for the generation of the dataset
* config_ANN.json: settings for the configuration of the ANNs and the training parameters

**Scripts**
* wh_generate_database.py: Wisdom-Holman integrator adapted for the generation of the dataset. 
* wh_tf_flag.py: Wisdom-Holman integrator adapted for the use of neural networks to substitute the calculation of the accelerations.
* generate_training_data.py: 
* data.py
* train_tf.py
* nn_tensorflow.py
* test_dataset.py
* hyperparameter_optimize.py
* plots_database.py
* plots_experiment.py
* plots_papers.py
* plot_tools.py
