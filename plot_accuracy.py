#!/usr/bin/python
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as mtick 
import numpy as np
import random
import math 

from pylab import genfromtxt;
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

def plot_(): 
    matplotlib.rcParams.update({'errorbar.capsize': 3})

    myFile = np.genfromtxt('accuracy.csv', delimiter=',')
    x = myFile[:,0]; 
    y = myFile[:,1];
    err = myFile[:,2] 

    matplotlib.rc('font', **font)
    plt.rc('axes', labelsize=22) 
    #x = np.random.randn(10000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    (_, caps, _) = ax.errorbar(x, y, yerr=err, fmt="o--", capthick=2, color="k")
    #ax.plot(x, y, "o--", color="k")
    #ax.fill_between(x, y-err, y+err, color="k", alpha = 0.2)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlim(0,21) 
    plt.ylim(0,1.0) 
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', direction='in', length = 3, labelsize=16)
    ax.tick_params(axis='x', which='major', direction='in', length = 6, labelsize=16)
    ax.tick_params(axis='y', which='minor', direction='in', length = 3, labelsize=16)
    ax.tick_params(axis='y', which='major', direction='in', length = 6, labelsize=16)
    fig.canvas.draw()
    plt.xlabel("Number of channels ($n$)"); 
    plt.ylabel("Prediction accuracy ($\eta$)"); 
    plt.savefig('accuracy.pdf',bbox_inches='tight')
    plt.show()




