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
        'size'   : 12}


def plot_histogram(ax, mean, std):
    data = np.random.normal(mean, std, 10000000)
    ax.hist(data,1000, alpha = 0.5, density=True)


def plot_(markers): 
    
    matplotlib.rc('font', **font)
    plt.rc('axes', labelsize=12) 
    #x = np.random.randn(10000)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(markers)):
            plot_histogram(ax, markers[i].mu, markers[i].sigma)

    
    
    #ax.plot(x, y, "o--", color="k")
    #ax.fill_between(x, y-err, y+err, color="k", alpha = 0.2)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlim(250, 650) 
    plt.ylim(0, 0.020) 
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', direction='in', length = 3, labelsize=12)
    ax.tick_params(axis='x', which='major', direction='in', length = 6, labelsize=12)
    ax.tick_params(axis='y', which='minor', direction='in', length = 3, labelsize=12)
    ax.tick_params(axis='y', which='major', direction='in', length = 6, labelsize=12)
    fig.canvas.draw()
    ax.set_aspect(aspect=3000)
    
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("pdf"); 
    
    plt.text(355, 0.014, "$\Delta m = 50 nm$, $s = 30 nm$", family="serif", fontsize=12)
    plt.tight_layout()
    plt.savefig('marker_wavelength_distribution.png', dpi=600)
    plt.show()




