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

def plot_(X):
    
    m1 = 4 
    n1 = 4

    fhist, axhist = plt.subplots(m1, n1, gridspec_kw = {'wspace':0.1, 'hspace':0.1}) 
    #fhist, axhist = plt.subplots(m1, n1) 

    colors = ['blue', 'red', 'green', 'magenta', 'red']
    #fhist.tight_layout()
    ch = 0
    for i in range(m1):
        for j in range(n1):
            ch = ch + 1  
            leg = str(ch) 
            for k in range(4): 
                axhist[i,j].hist(X[k][:,i*n1+j], color=colors[k], density=True, bins=1000, alpha = 0.5)
            axhist[i,j].set_xlim([-1, 8]) 
            axhist[i,j].set_ylim([0, 1.2])
            axhist[i,j].text(6, 1.1, leg, horizontalalignment='center', verticalalignment='top') 
    for a in axhist.flatten():
        a.set_xticks([])
        a.set_yticks([])

 
    plt.savefig('signal_distribution.png', dpi=600)

    plt.show() 




