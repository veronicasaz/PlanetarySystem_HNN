import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from matplotlib import rcParams
# rc('text.latex', preamble=r'\usepackage{charter}')
import seaborn as sns
import pandas as pd 
from data import get_dataset

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['axes.formatter.useoffset'] = False

color1 = ['navy', 'dodgerblue','darkorange']
color2 = ['dodgerblue', 'navy', 'orangered', 'green', 'olivedrab',  'saddlebrown', 'darkorange', 'red' ]
    
def plot_pairplot(data, name):
    """
    plot_pairplot: plot scatter of every input and output
    INPUTS: 
        data: data dictionary
    """
    data_np = np.hstack((data['coords'], data['dcoords']))
    # Create labels for pandas dataframe
    l_c = ['m', 'x', 'y', 'z']
    l_d = ['ax', 'ay', 'az']
    labels_c = list()
    for i in range(np.shape(data['coords'])[1]//4):
        for j in range(4):
            labels_c.append(l_c[j] +str(i+1))
    labels_d = list()
    for i in range(np.shape(data['dcoords'])[1]//3):
        for j in range(3):
            labels_d.append(l_d[j] +str(i+1))
    labels = labels_c + labels_d     
    # Pandas dataframe to input in seaborn   
    data_pd = pd.DataFrame(data=data_np[0:1000,:], columns = labels)

    g = sns.pairplot(data_pd,\
        x_vars=labels_c,
        y_vars=labels_d
    )
    plt.savefig( "./dataset/"+name+"pairplot.png", dpi = 100)
    plt.show()

def plot_correlation(data, name):
    """
    plot_correlation: plot correlation of inputs and outputs
    INPUTS: 
        data: data dictionary
    """
    # Create pandas dataset to input in seaborn
    data_np = np.hstack((data['coords'], data['dcoords']))
    l_c = ['m', 'x', 'y', 'z']
    l_d = ['ax', 'ay', 'az']
    labels_c = list()
    for i in range(np.shape(data['coords'])[1]//4):
        for j in range(4):
            labels_c.append(l_c[j] +str(i+1))
    labels_d = list()
    for i in range(np.shape(data['dcoords'])[1]//3):
        for j in range(3):
            labels_d.append(l_d[j] +str(i+1))
    labels = labels_c + labels_d
    data_pd = pd.DataFrame(data=data_np[0:1000,:], columns = labels)

    # Calculate correlation
    corr = data_pd.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap,center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Correlation matrix input-output", fontsize = 20)
    plt.savefig( "./dataset/"+name+"correlation.png", dpi = 100)
    plt.show()


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def plot_distribution(coords, dcoords, namefile, name):
    """
    plot_distribution: plot bars with frequency of inputs and outputs
    INPUTS: 
        coords: inputs (mass and position of each body)
        dcoords: outputs (acceleration of each body)
        name: addition to name of figure to be saved
    """
    # Plot new distribution inputs
    n_inputs = np.shape(coords)[1]
    n_outputs = np.shape(dcoords)[1]
    n_cols = 4 # 3 coordinates for position + mass
    n_rows = n_inputs // n_cols 

    # PLOT INPUTS
    fig, axes = plt.subplots(figsize = (12,8),nrows=n_rows, ncols=n_cols)
    fig.subplots_adjust(top=0.9,hspace = 0.3, wspace= 0.1)
    if name == 'asteroid/':
        # title = [r'Jupiter mass ($m_{Sun}$)',r'Jupiter x ($au$)', r'Jupiter y ($au$)', r'Jupiter z ($au$)',\
        #      r'Saturn mass ($m_{Sun}$)',r'Saturn x ($au$)', r'Saturn y ($au$)', r'Saturn z ($au$)', \
        #      r'Asteroid mass ($kg$)', r'Asteroid x ($au$)', r'Asteroid y ($au$)', r'Asteroid z ($au$)']
        title = [r'Jupiter mass ($m_{Sun}$)',r'Jupiter x ($au$)', r'Jupiter y ($au$)', r'Jupiter z ($au$)',\
             r'Saturn mass ($m_{Sun}$)',r'Saturn x ($au$)', r'Saturn y ($au$)', r'Saturn z ($au$)', \
             r'Asteroid mass ($\times 10^{8}$) ($m_{Sun}$)', r'Asteroid x ($au$)', r'Asteroid y ($au$)', r'Asteroid z ($au$)']
        title_o = [r'Jupiter $a_x$ ($au\;/\;  yr^2$)', r'Jupiter $a_y$ ($au\;/\;    yr^2$)', r'Jupiter $a_z$ ($au\;/\;    yr^2$)',\
             r'Saturn $a_x$ ($au\;/\;    yr^2$)', r'Saturn $a_y$ ($au\;/\;    yr^2$)', r'Saturn $a_z$ ($au\;/\;    yr^2$)', \
             r'Asteroid $a_x$ ($au\;/\;    yr^2$)', r'Asteroid $a_y$ ($au\;/\;    yr^2$)', r'Asteroid $a_z$ ($au\;/\;    yr^2$)']
    else:
        title = [r'Jupiter mass ($m_{Sun}$)',r'Jupiter x ($au$)', r'Jupiter y ($au$)', r'Jupiter z ($au$)',\
             r'Saturn mass ($m_{Sun}$)',r'Saturn x ($au$)', r'Saturn y ($au$)', r'Saturn z ($au$)']
        title_o = [r'Jupiter $a_x$ ($au\;/\;  yr^2$)', r'Jupiter $a_y$ ($au\;/\;    yr^2$)', r'Jupiter $a_z$ ($au\;/\;    yr^2$)',\
             r'Saturn $a_x$ ($au\;/\;    yr^2$)', r'Saturn $a_y$ ($au\;/\;    yr^2$)', r'Saturn $a_z$ ($au\;/\;    yr^2$)']

    # coords[:, -4] *= 1e10
    for j in range(n_inputs):
        ax = axes[int(j //(n_cols)), int(j%(n_cols) )] 
        ax.hist(coords[:, j], bins = 20, histtype = 'bar', color = color1[0], edgecolor="white")
        # ax.set_title(title[j], fontsize = 13)
        ax.set_xlabel(title[j], fontsize = 18)
        ax.set_ylabel("Frequency", fontsize = 18)
        ax.set_xticklabels(trunc(ax.get_xticks(), decs = 3),  fontsize=15)
        ax.get_xaxis().set_major_formatter('{x:1.2f}')
        ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
    # plt.suptitle('Distribution of inputs', y=0.98, fontsize = 20)
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"input_distribution"+namefile+".png", dpi = 100)
    plt.show()

    # PLOT OUTPUTS
    
    fig, axes = plt.subplots(figsize = (12,8),nrows=n_rows, ncols=n_cols-1)
    fig.subplots_adjust(top=0.9,hspace = 0.3, wspace= 0.4)
    for j in range(n_outputs):
        ax = axes[int(j //(n_cols-1)), int(j%(n_cols-1) )]         
        hist, bins, _ = ax.hist(dcoords[:, j], bins =100)
        ax.clear()
        # Correct negative values to put in log scale
        idx_neg = np.where(bins <0)[0]
        idx_pos = np.where(bins>0)[0]
        bins_pos = bins[idx_pos]
        bins_neg = -bins[idx_neg]
        bins_abs = np.concatenate((bins_neg, bins_pos))

        logbins_pos = np.logspace(np.log10(bins_pos[0]/100),np.log10(bins_pos[-1]),50)
        logbins_neg = np.flip(-logbins_pos)
        logbins = np.concatenate((logbins_neg, logbins_pos ))
        
        ax.hist(dcoords[:, j], bins = logbins, histtype = 'bar', color = color1[0], density=False, edgecolor="white")
        ax.set_xscale("symlog", linthresh = 10**(np.trunc(min(np.log10(bins_abs)))))
        ax.set_xticklabels(ax.get_xticks() ,rotation = 45, fontsize = 15)
        ax.get_xaxis().set_major_formatter(plt.LogFormatter(10,  labelOnlyBase=False))
        ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
        # ax.set_yscale("symlog",linthresh=1.e-6)
        ax.set_xlabel(title_o[j], fontsize = 18)
        ax.set_ylabel("Frequency", fontsize = 18)
    # plt.suptitle('Distribution of outputs', y=0.98, fontsize = 20)
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"output_distribution"+namefile+".png", dpi = 100)
    plt.show()

def plot_distance(data, name):
    """
    plot_distance: plot acceleration value for each distance combination. 
    Each combination is a subplot
    INPUTS: 
        data: dataset with inputs and outputs
    """
    coords = data['coords']
    dcoords = data['dcoords']
    n_inputs = np.shape(coords)[1]
    n_outputs = np.shape(dcoords)[1]
    n_samples = min(np.shape(coords)[0], 2000)

    # Only plot a certain number of samples
    coords = coords[0:n_samples, :]
    dcoords = dcoords[0:n_samples, :]

    # All combinations of planet 1, planet 2, asteroid
    index_combi = [[0, 1], [0,2], [1,2]]

    fig, ax = plt.subplots(figsize = (12,8),nrows=3, ncols=3)
    fig.subplots_adjust(top=0.9,hspace = 0.3, wspace= 0.4)

    d_i = np.zeros((n_samples, 3))
    d_o = np.zeros((n_samples, 3))

    # Calculate distances
    for i, x in enumerate(index_combi):
        d_i[:, i] = np.linalg.norm(coords[:, x[0]*4+1:x[0]*4+4] - coords[:, x[1]*4+1:x[1]*4+4], axis = 1)
        d_o[:, i] = np.linalg.norm(dcoords[:, i*3:i*3+3], axis = 1)
    
    cm = plt.cm.get_cmap('jet')
    for i in range(3):
        for j, x in enumerate(index_combi):
            print(x[0], x[1])
            sc = ax[i, j].scatter(d_i[:, x[0]], d_i[:, x[1]], s = 8, c = d_o[:, i], norm=plc.LogNorm(), cmap =  cm)
            ax[i, j].set_title(r'$|a_%i|$'%(i))
            ax[i, j].set_xlabel(r'$|r_%i -r_%i|$'%(index_combi[x[0]][0], index_combi[x[0]][1]))
            ax[i, j].set_ylabel(r'$|r_%i -r_%i|$'%(index_combi[x[1]][0], index_combi[x[1]][1]))
    pcm = plt.colorbar(sc)
    plt.suptitle("Acceleration as a function of the distance between pairs of bodies", fontsize = 20)
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"distance_inputoutput.png", dpi = 100)
    plt.show()

def plot_distance3D(data, name):
    """
    plot_distance3D: plot acceleration value of asteroid for each distance combination
    INPUTS: 
        data: dataset with inputs and outputs
    """
    coords = data['coords']
    dcoords = data['dcoords']
    n_inputs = np.shape(coords)[1]
    n_outputs = np.shape(dcoords)[1]
    n_samples = min(np.shape(coords)[0], 20000)

    coords = coords[0:n_samples, :]
    dcoords = dcoords[0:n_samples, :]

    # Combinations of planet 1, planet 2, asteroid
    index_combi = [[0, 1], [0,2], [1,2]]

    fig, ax = plt.subplots(figsize = (12,8),nrows=1, ncols=1)
    fig.subplots_adjust(top=0.9,hspace = 0.3, wspace= 0.4)

    d_i = np.zeros((n_samples, 3))
    d_o = np.zeros((n_samples, 3))

    # Calculate distances
    for i, x in enumerate(index_combi): 
        d_i[:, i] = np.linalg.norm(coords[:, x[0]*4+1:x[0]*4+4] - coords[:, x[1]*4+1:x[1]*4+4], axis = 1)
        d_o[:, i] = np.linalg.norm(dcoords[:, i*3:i*3+3], axis = 1)
    
    cm = plt.cm.get_cmap('jet')
    i = 2 # plot only a of asteroid
    ax = plt.axes(projection='3d')
    sc = ax.scatter(d_i[:, 0], d_i[:, 1], d_i[:, 2], s = 15, c = d_o[:, i], norm=plc.LogNorm(), cmap =  cm)
    ax.set_title('|a_%i|'%(i))
    ax.set_xlabel('|r_J -r_S|')
    ax.set_ylabel('|r_J -r_a|')
    ax.set_zlabel('|r_S -r_a|')

    pcm = plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"distance_inputoutput3D.png", dpi = 100)
    plt.show()

def covariance_matrix(data):
    """
    covariance_matrix: calculate covariance matrix
    INPUTS: 
        data: dataset with inputs and outputs
    OUTPUTS:
        det_cov: determinant of the covariance matrix
    """
    size = np.shape(data)[1]
    samples = np.shape(data)[0]
    cov = np.identity(size)
    
    variance = np.zeros(size)
    for i in range(samples):
        variance += 1/(samples-1) * (data[i, :] - np.mean(data, axis = 0) )**2

    # Covariances
    for i in range(size):
        cov[i, i] = variance[i] # diagonal
        for j in range(i+1, size):
            element = 0
            for z in range(samples):
                element += 1/(samples-1) * (data[z, i] - np.mean(data[:, i])) * (data[z, j] - np.mean(data[:, j]))
            cov[i, j] = element
            cov[j, i] = element

    # Determinant
    det_cov = np.linalg.det(cov) #TODO: problem because of small numbers??
    return det_cov

def plot_covariance_samples(data, name):
    """
    plot_covariance_samples: plot covariance as a function of the number of samples
    INPUTS: 
        data: dataset with inputs and outputs
    """
    samples = [30, 50, 100, 500, 1000, 2000, 5000] # at least more than the number of inputs
    # samples = [30] # at least more than the number of inputs
    I = np.zeros(len(samples))
    O = np.zeros(len(samples))

    for i in range(len(samples)):
        I[i] = covariance_matrix(data['coords'][0:samples[i], :])
        O[i] = covariance_matrix(data['dcoords'][0:samples[i], :])
    
    fig, ax = plt.subplots(figsize = (12,8),nrows=1, ncols=2)
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']
    marker = ['x', 'o', 's', '^']
    for i in range(len(samples)):
        ax[0].scatter(I[i], O[i], color = color[i%len(color)], marker = marker[i%len(marker)], label = str(samples[i]))

    ax[0].set_xlabel("det(cov( I ))")
    ax[0].set_ylabel("det(cov( O ))")
    ax[0].legend()
    ax[1].plot(samples, O/I, marker = "o")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("det(cov( O )) / det(cov( I ))")
    # ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    plt.title("Covariance")
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"covariance_samples.png", dpi = 100)
    plt.show()

def plot_covariance(data, name):
    """
    plot_covariance: plot covariance 
    INPUTS: 
        data: dataset with inputs and outputs
    """
    samples = 1000
    tests = 50

    size_I = np.shape(data['coords'])[1]
    I = np.zeros(tests)
    O = np.zeros(tests)

    data_a = np.hstack((data['coords'], data['dcoords']))

    for i in range(tests):
        np.random.shuffle(data_a)
        I[i] = covariance_matrix(data_a[0:samples, 0:size_I])
        O[i] = covariance_matrix(data_a[0:samples, size_I:])

    fig, ax = plt.subplots(figsize = (12,8),nrows=1, ncols=1)
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']
    marker = ['x', 'o', 's', '^']
    for i in range(tests):
        ax.scatter(I[i], O[i], color = 'black', marker = 'o')
    ax.set_xlabel("det(cov( I ))")
    ax.set_ylabel("det(cov( O ))")
    ax.legend()
    plt.title("Covariance (%i samples)"%samples)
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"covariance.png", dpi = 100)
    plt.show()

def plot_trajectory(data, name):
    """
    plot_trajectory: plot distribution of inputs in x-y plane 
    INPUTS: 
        data: dataset with inputs and outputs
    """
    fig, ax = plt.subplots(figsize = (8,6),nrows=1, ncols=1)
    marker = [ 'o','.', 's', 's']
    markersize = 20
    axissize = 25
    ticksize = 23
    # fig.subplots_adjust(top=0.9,hspace = 0.3, wspace= 0.4)

    # Choose variable and number of samples to plot
    x_1 = data['coords'][:500, 1]
    y_1 = data['coords'][:500, 2]
    x_2 = data['coords'][:500, 5]
    y_2 = data['coords'][:500, 6]

    plt.scatter(0,0, color = color2[3], marker = marker [0], s = markersize, label = 'Sun')

    if name == 'asteroid/':
        x_3 = data['coords'][:200, 9]
        y_3 = data['coords'][:200, 10]
        plt.scatter(x_3, y_3, color = color2[2], marker = marker[1],  s = markersize, label = 'Asteroids')

    plt.scatter(x_1, y_1, color = color2[0], marker = marker[0], s = markersize, label = 'Jupiter')
    plt.scatter(x_2, y_2, color = color2[1], marker = marker[0], s = markersize, label = 'Saturn')
    plt.xlabel('x (au)', fontsize = axissize)
    plt.ylabel('y (au)', fontsize = axissize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.axis('equal')

    plt.grid(alpha = 0.5)
    # plt.legend(fontsize = 20, framealpha = 0.85, loc = 'upper left')
    lgnd = plt.legend(loc = 'lower left', fontsize = 23, \
                framealpha = 0.9, bbox_to_anchor=(-0.12, 1.02, 1.2, 1.5),\
                ncol=4, mode="expand", borderaxespad=0., handletextpad=0.)
    lgnd.legendHandles[0]._sizes = [100]
    lgnd.legendHandles[1]._sizes = [100]
    lgnd.legendHandles[2]._sizes = [100]
    lgnd.legendHandles[3]._sizes = [100]
    # plt.legend(fontsize = axissize, framealpha = 1.0, bbox_to_anchor=(1.0, 1.0))
    
    # plt.title("Distribution of positions of Jupiter, Saturn, and the asteroids", fontsize = 15)
    plt.tight_layout()
    plt.savefig( "./dataset/"+name+"trajectory_distribution.png", dpi = 100, bbox_inches='tight')
    plt.show()
    

    
