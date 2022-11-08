"""
Created: July 2021 
Last modified: October 2022 
Author: Veronica Saz Ulibarrena 
Description: Use trained network to predict test dataset. Plot results
"""
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

import matplotlib
import ast
from nn_tensorflow import  ANN 
import tensorflow as tf

from plot_tools import trunc, color1, color2, CustomTicker

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def load_dataset(config, settings_file_path, data, path_model = None, drdv = False):
    """
    load_dataset: load dataset, use network to predict outputs
    INPUTS:
        config: configuration from config_data.json and config_ANN.json
        settings_file_path: path to config file
        data: dictionary with training and testing inputs/outputs
        path_model: if given, path to trained model. Otherwise take from config file
    OUTPUTS:
        test_x: inputs of test dataset
        test_dxdt_pred: predicted output of the test dataset
        test_dxdt: real output of the test dataset
    """
    if path_model == None:
        path_model = config['model_path']+"/"
    device = 'cpu'
    
    x = data['coords']
    dxdt = data['dcoords']
    test_x = data['test_coords']
    test_dxdt = data['test_dcoords']
    
    # Reduce the amound of data to plot
    points = 1000
    test_x = test_x[0:points, :]
    test_dxdt = test_dxdt[0:points, :]

    # Tensorflow network
    ANN_tf = ANN(config, path_model = path_model)
    ANN_tf.load_model_fromFile(path_model)
    # ANN_tf.load_weights(path_model) # sometimes this command is needed instead of load_model_fromFile

    # Predict feature standardizes data
    if drdv == False:
        test_dxdt_pred = ANN_tf.predict(test_x, std = False) #TODO eliminate masses
    else:
        ANN_tf.pred_type = 'a'
        test_dxdt_pred = ANN_tf.predict(test_x, std = False)
    return test_x, test_dxdt_pred , test_dxdt

def plot_prediction_error(path_figure, x, y_pred, y_real):
    """
    plot_prediction_error: plot error in real vs predicted plot
    INPUTS:
        path_figure: path to save figure
        x: inputs of the network
        y_pred: predicted outputs
        y_real: real outputs (from dataset)
    """
    subplot2 = 3
    subplot1 = np.shape(y_pred)[1]//subplot2
    fig = plt.figure(figsize = (15,15))
    subfigs = fig.subfigures( nrows=subplot1, ncols=1)
    plt.subplots_adjust(wspace = 0.3, hspace=0.9)

    # Create 1 row per body, 3 columns for ax, ay, az
    xlabel = [r'$a_x$', r'$a_y$', r"$a_z$"]
    title = ['Jupiter', 'Saturn', 'Asteroids']
    labelsize = 15
    for sbu in range(subplot1):
        subfigs[sbu].suptitle(title[sbu], fontsize = 18)
        subfigs[sbu].subplots_adjust(wspace = 0.5)

        ax = subfigs[sbu].subplots(nrows=1, ncols=subplot2)
        for sbu2 in range(subplot2):
            var = sbu*3+ sbu2
            RMSE = np.square(np.subtract(y_real[:, var], y_pred[:, var])).mean() 
            ax[sbu2].scatter(y_real[:, var], y_pred[:, var])
            ax[sbu2].set_title("RMSE = %0.2E"%RMSE, fontsize = 12) # write RMSE in plot

            xline = np.linspace(min(y_real[:, var]), max(y_real[:,var]), num= 2)
            ax[sbu2].plot(xline, xline, color = 'red')
            
            ax[sbu2].set_xlabel(xlabel[int(var%3)]+" real", fontsize = labelsize)
            ax[sbu2].set_ylabel(xlabel[int(var%3)]+" predicted", fontsize = labelsize)
            ax[sbu2].set_xticklabels(ax[sbu2].get_xticks(), rotation = 5)
            ax[sbu2].xaxis.set_major_formatter(FormatStrFormatter('%.2E'))
            ax[sbu2].yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
            ax[sbu2].set_xlim(left = min(y_real[:, var]), right = max(y_real[:, var]))
            ax[sbu2].set_ylim(bottom = min(y_pred[:, var]), top = max(y_pred[:, var]))
            ax[sbu2].grid(alpha = 0.2)

    # plt.tight_layout()
    plt.savefig(path_figure+'Error.png', dpi = 100)
    plt.show() 

def plot_prediction_error_HNNvsDNN(path_figure, x, y_pred, y_real, x2, y_pred2, y_real2, x3, y_pred3, y_real3):
    """
    plot_prediction_error_HNNvsDNN: plot error in real vs predicted plot for HNN and DNN
    INPUTS:
        path_figure: path to save figure
        For HNN with all in loss:
        x: inputs of the network
        y_pred: predicted outputs
        y_real: real outputs (from dataset)
        For DNN with all in loss:
        x2: inputs of the network
        y_pred2: predicted outputs
        y_real2: real outputs (from dataset)
        For HNN with asteroid in loss:
        x3: inputs of the network
        y_pred3: predicted outputs
        y_real3: real outputs (from dataset)
    """
    subplot2 = 3
    subplot1 = np.shape(y_pred)[1]//subplot2
    fig = plt.figure(figsize = (23,19))
    subfigs = fig.subfigures( nrows=subplot1, ncols=1)
    plt.subplots_adjust(wspace = 0.5, hspace=2.2)

    xlabel = [r'\rm x', r'\rm y', r"\rm z"]
    title = ['Jupiter', 'Saturn', 'Asteroids']
    for sbu in range(subplot1):
        subfigs[sbu].subplots_adjust(left = 0.07, right = 0.82, wspace = 0.4, hspace = 2.0, top = 0.89, bottom = 0.25)
        ax = subfigs[sbu].subplots(nrows=1, ncols=subplot2)
        subfigs[sbu].suptitle(title[sbu], fontsize = 35, x=0.41, y=.97, horizontalalignment='left')

        for sbu2 in range(subplot2):
            var = sbu*3+ sbu2
            xline = np.linspace(min(y_real[:, var])*1.2, max(y_real[:,var])*1.2, num= 2)
            ax[sbu2].plot(xline, xline, color = 'black', linewidth = 4, zorder = 1, alpha = 0.5, label = 'Zero-error line')

            RMSE = np.square(np.subtract(y_real[:, var], y_pred[:, var])).mean()
            RMSE2 = np.square(np.subtract(y_real2[:, var], y_pred2[:, var])).mean() 
            ax[sbu2].scatter(y_real[:, var], y_pred[:, var], label = 'HNN with all bodies', color = color1[0], marker = 'o', s = 10, zorder = 2)
            ax[sbu2].scatter(y_real3[:, var], y_pred3[:, var], label = 'HNN asteroid in loss', color = color1[2], marker = 'o', s = 10, zorder = 3)
            ax[sbu2].scatter(y_real2[:, var], y_pred2[:, var], label = 'DNN', color = color1[1], marker = 'o', s = 10, zorder =4)
            # ax[sbu2].set_title("HNN RMSE = %0.2E, DNN RMSE = %0.2E, "%(RMSE, RMSE2), fontsize = 12)

            ax[sbu2].set_xlim(left = min(y_real[:, var]), right = max(y_real[:, var]))
            ax[sbu2].set_ylim(bottom = min(y_real[:, var]), top = max(y_real[:, var]))
            
            # Normalize axis
            if sbu == 0:
                norm = 1e4
                # ax[sbu2].set_xlabel(xlabel[int(var%3)]+r" real $\times10^4$ ($au/yr^2$)", fontsize = 32)
                # ax[sbu2].set_ylabel(xlabel[int(var%3)]+r" predicted  $\times10^4$ ($au/yr^2$)", fontsize = 32)

                ax[sbu2].set_xlabel(r"$a_{%s}^{\rm real}  \;(\rm au/\rm yr^2 \; \times10^{-4})$"%xlabel[int(var%3)], fontsize = 32)
                ax[sbu2].set_ylabel(r"$a_{%s}^{\rm pred}  \;(\rm au/\rm yr^2 \;\times10^{-4})$"%xlabel[int(var%3)], fontsize = 32)
            elif sbu == 1:
                norm = 1e3
                # ax[sbu2].set_xlabel(xlabel[int(var%3)]+r" real $\times10^3$ ($au/yr^2$)", fontsize = 32)
                # ax[sbu2].set_ylabel(xlabel[int(var%3)]+r" predicted  $\times10^3$ ($au/yr^2$)", fontsize = 32)
                ax[sbu2].set_xlabel(r"$a_{%s}^{\rm real}  \;(\rm au/\rm yr^2 \; \times10^{-3})$"%xlabel[int(var%3)], fontsize = 32)
                ax[sbu2].set_ylabel(r"$a_{%s}^{\rm pred}  \;(\rm au/\rm yr^2 \;\times10^{-3})$"%xlabel[int(var%3)], fontsize = 32)
            elif sbu == 2:
                norm = 1
                # ax[sbu2].set_xlabel(xlabel[int(var%3)]+r" real ($au/yr^2$)", fontsize = 32)
                # ax[sbu2].set_ylabel(xlabel[int(var%3)]+r" predicted ($au/yr^2$)", fontsize = 32)
                ax[sbu2].set_xlabel(r"$a_{%s}^{\rm real}  \;(\rm au/\rm yr^2)$"%xlabel[int(var%3)], fontsize = 32)
                ax[sbu2].set_ylabel(r"$a_{%s}^{\rm pred}  \;(\rm au/\rm yr^2)$"%xlabel[int(var%3)], fontsize = 32)
            

            ticks = -np.log10(ax[sbu2].get_xticks())
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
            ax[sbu2].set_xticklabels(trunc(np.round(ax[sbu2].get_xticks()* norm, decimals = dec), decs = 4) , rotation = 40, fontsize = 27)
            ticks = -np.log10(ax[sbu2].get_yticks())
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
            ax[sbu2].set_yticklabels(trunc(np.round(ax[sbu2].get_yticks()* norm, decimals = dec), decs = 4) , rotation = 0, fontsize = 27)

            ax[sbu2].grid(alpha = 0.2)

        if sbu == 0:
            lgnd = ax[0].legend(loc = 'lower left', fontsize = 28, \
                framealpha = 0.9, bbox_to_anchor=(-0.0, 1.15, 3.7, 1.0),\
                ncol=4, mode="expand", borderaxespad=0., handletextpad=0.)

            # Change sizes of markers in legend
            lgnd.legendHandles[0]._sizes = [100]
            lgnd.legendHandles[1]._sizes = [100]
            lgnd.legendHandles[2]._sizes = [100]
            lgnd.legendHandles[3]._sizes = [100]            
            
    plt.savefig(path_figure+'Error_HDN.png', dpi = 100, bbox_inches='tight')
    plt.show() 

def plot_prediction_error_HNNvsDNN_JS(path_figure, x, y_pred, y_real, x2, y_pred2, y_real2):
    """
    plot_prediction_error_HNNvsDNN_JS: plot error in real vs predicted plot for HNN and DNN
    INPUTS:
        path_figure: path to save figure
        path_figure: path to save figure
        For HNN with all in loss:
        x: inputs of the network
        y_pred: predicted outputs
        y_real: real outputs (from dataset)
        For DNN with all in loss:
        x2: inputs of the network
        y_pred2: predicted outputs
        y_real2: real outputs (from dataset)
    """
    subplot2 = 3
    subplot1 = np.shape(y_pred)[1]//subplot2
    fig = plt.figure(figsize = (23,15))
    subfigs = fig.subfigures( nrows=subplot1, ncols=1)
    plt.subplots_adjust(left = 0.15, right = 0.98, wspace = 0.2, hspace=1.2, top = 0.9, bottom = 0.3)

    xlabel = [r'$a_x$', r'$a_y$', r"$a_z$"]
    title = ['Jupiter', 'Saturn']
    for sbu in range(subplot1):
        # subfigs[sbu].suptitle(title[sbu], fontsize = 17)
        subfigs[sbu].subplots_adjust(wspace = 0.5)

        ax = subfigs[sbu].subplots(nrows=1, ncols=subplot2)

        for sbu2 in range(subplot2):
            var = sbu*3+ sbu2
            xline = np.linspace(min(y_real[:, var])*1.2, max(y_real[:,var])*1.2, num= 2)
            ax[sbu2].plot(xline, xline, color = 'black', linewidth = 2, zorder = 1, alpha = 0.5)

            RMSE = np.square(np.subtract(y_real[:, var], y_pred[:, var])).mean()
            RMSE2 = np.square(np.subtract(y_real2[:, var], y_pred2[:, var])).mean() 
            ax[sbu2].scatter(y_real[:, var], y_pred[:, var], label = 'HNN', color = color1[0], marker = 'o', s = 15, zorder = 3)
            ax[sbu2].scatter(y_real2[:, var], y_pred2[:, var], label = 'DNN', color = color1[1], marker = 'x', s = 15, zorder =2)

            ax[sbu2].set_xlim(left = min(y_real[:, var]), right = max(y_real[:, var]))
            ax[sbu2].set_ylim(bottom = min(y_pred[:, var]), top = max(y_pred[:, var]))
            
            ax[sbu2].set_xlabel(xlabel[int(var%3)]+" real  ($au/yr^2$)", fontsize = 20)
            ax[sbu2].set_ylabel(xlabel[int(var%3)]+" predicted  ($au/yr^2$)", fontsize = 20)
            
            ticks = -np.log10(abs(ax[sbu2].get_xticks()))
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )+1
            ax[sbu2].set_xticklabels(np.round(trunc(ax[sbu2].get_xticks(), decs = 5), decimals = dec), rotation = 30, fontsize = 16)
            ticks = -np.log10(abs(ax[sbu2].get_yticks()))
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
            ax[sbu2].set_yticklabels(np.round(trunc(ax[sbu2].get_yticks(), decs = 5), decimals = dec), fontsize = 16)

            ax[sbu2].grid(alpha = 0.2)
        
        if sbu == 0:
            lgnd = ax[0].legend(fontsize = 20)
            lgnd.legendHandles[0]._sizes = [100]
            lgnd.legendHandles[1]._sizes = [100]

    # plt.tight_layout()
    plt.savefig(path_figure+'Error_SJS.png', dpi = 100)
    plt.show() 

def plot_prediction_error_HNNvsDNN_JS_dif(path_figure, x, y_pred, y_real, x2, y_pred2, y_real2):
    """
    plot_prediction_error_HNNvsDNN_JS: plot error in real vs predicted plot for HNN and DNN
    INPUTS:
        path_figure: path to save figure
        path_figure: path to save figure
        For HNN with all in loss:
        x: inputs of the network
        y_pred: predicted outputs
        y_real: real outputs (from dataset)
        For DNN with all in loss:
        x2: inputs of the network
        y_pred2: predicted outputs
        y_real2: real outputs (from dataset)
    """
    subplot2 = 3
    subplot1 = np.shape(y_pred)[1]//subplot2
    fig = plt.figure(figsize = (23, 14))
    subfigs = fig.subfigures( nrows=subplot1, ncols=1)
    plt.subplots_adjust(wspace = 0.5, hspace=2.2)

    xlabel = [r'\rm x', r'\rm y', r"\rm z"]
    title = ['Jupiter', 'Saturn']
    for sbu in range(subplot1):
        subfigs[sbu].suptitle(title[sbu], fontsize = 35, x=0.41, y=.97, horizontalalignment='left')
        # subfigs[sbu].subplots_adjust(wspace = 0.3, top = 0.89, bottom = 0.25)
        subfigs[sbu].subplots_adjust(left = 0.07, right = 0.82, wspace = 0.4, hspace = 2.0, top = 0.89, bottom = 0.25)
        ax = subfigs[sbu].subplots(nrows=1, ncols=subplot2)

        for sbu2 in range(subplot2):
            var = sbu*3+ sbu2
            ax[sbu2].scatter(y_real[:, var], y_pred[:, var], label = 'HNN', color = color1[0], marker = 'o', s = 50, zorder = 3)
            ax[sbu2].scatter(y_real2[:, var], y_pred2[:, var], label = 'DNN', color = color1[1], marker = 's', s = 50, zorder =2)

            ax[sbu2].set_xlabel(r"$a_{%s}^{\rm real}  \;(\rm au/\rm yr^2)$"%xlabel[int(var%3)], fontsize = 35)
            ax[sbu2].set_ylabel(r"$a_{%s}^{\rm pred}\;(\rm au/yr^2)$"%(xlabel[int(var%3)]), fontsize = 35)
            if sbu2 == 2:
                ax[sbu2].set_xscale('symlog', linthresh = 1e-5)
                ax[sbu2].set_yscale('symlog', linthresh = 1e-5)
            else: 
                ax[sbu2].set_xscale('symlog', linthresh = 1e-5)
                ax[sbu2].set_yscale('symlog', linthresh = 1e-5)
            
            ticks = -np.log10(ax[sbu2].get_xticks())
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )           
            ax[sbu2].set_xticklabels(trunc(np.round(ax[sbu2].get_xticks(), decimals = dec), decs = 4) , rotation = 20, fontsize = 27)
            
            ticks = -np.log10(abs(ax[sbu2].get_yticks()))+1
            dec = int(np.round(np.nanmax(ticks[ticks!= np.inf]), 0) )
            ax[sbu2].set_yticklabels(np.round(trunc(ax[sbu2].get_yticks(), decs = 6), decimals = dec), fontsize = 27)

            ax[sbu2].xaxis.set_major_formatter(CustomTicker())
            ax[sbu2].yaxis.set_major_formatter(CustomTicker())
            ax[sbu2].grid(alpha = 0.2)
        
        if sbu == 0:
            lgnd = ax[0].legend(fontsize = 28)
            lgnd.legendHandles[0]._sizes = [100]
            lgnd.legendHandles[1]._sizes = [100]

    # plt.tight_layout()
    plt.savefig(path_figure+'Error_SJS_dif.png', dpi = 100, bbox_inches='tight')
    plt.show() 


def plot_prediction_error_H(path_figure, x, y_pred, y_real):
    """
    plot_prediction_error_H: plot error when the H is in the loss function
    INPUTS: 
        path_figure: path to save the figure
        x: inputs of the network
        y_pred: predicted outputs
        y_real: real outputs (from dataset)
    """
    fig, ax = plt.subplots(1)
    plt.scatter(y_real, y_pred)

    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.grid(alpha = 0.5)
    plt.axis('equal')
    # plt.tight_layout()
    plt.savefig(path_figure+'Error_H.png', dpi = 1000)
    plt.show() 
