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
    #X_ = X[0][:10000]
    #fig = plt.figure(figsize=(3,3))
    #ax = fig.add_subplot(111)
    #ax.set_aspect('auto')
    #plt.imshow(X_, vmin=-0, vmax=1, cmap='OrRd', aspect='auto') 
    #ax.set_xticks([])
    #ax.set_yticks([])
    #plt.savefig('colormap_0.png', dpi=600)
    #plt.colorbar() 
    
    m1, d1, d2 = X.shape 

    n1 = m1

    f, ax = plt.subplots(m1, n1, gridspec_kw = {'wspace':0.1, 'hspace':0.1}) 
    #f, ax = plt.subplots(m1, n1) 

    colors = ['blue', 'red', 'green', 'magenta', 'red']
    #f.tight_layout()
   
    for i in range(m1): 
        for j in range(n1):
            leg = str(j+1) 
            X_ = X[j][:10000]
            ax[i,j].imshow(X_, vmin=-0, vmax=3, cmap='OrRd', aspect='auto') 
            ax[i,j].text(14, -0.1, leg, horizontalalignment='center',verticalalignment='top') 
    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])
        
    plt.savefig('colormap.png', dpi=600)
    plt.show() 


