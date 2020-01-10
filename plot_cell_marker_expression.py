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

def plot_(sample, markers): 

    matplotlib.rc('font', **font)
    plt.rc('axes', labelsize=20) 
    #x = np.random.randn(10000)
        
    colors = ['blue', 'red', 'green', 'magenta', 'red']
    n_marker = len(markers)
    n_cell_type = len(sample.cell_types)
    n_cell = len(sample.cell_types[0].cells)

    dx = 1 
    dy = 1

    nrows = 1
    ncols = n_marker   
   
    figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols)) 

    f, ax = plt.subplots(1, n_marker, figsize=figsize)

    for j, a in enumerate(ax.flatten()):
        for i in range(n_cell_type): 
            V = [] 
            for k in range(n_cell):
                V.append(np.log(sample.cell_types[i].cells[k].intensity[j])) 
        
            a.hist(V, color=colors[i], density=True, bins=1000, alpha = 0.5)
            cell_type = "Cell Type " + str(j + 1) 
            a.text(-1,1.75, cell_type, fontsize=20)
            a.set(ylim=[0, 2]) 

    pad = 0.05 # Padding around the edge of the figure
    xpad, ypad = dx * pad, dy * pad
    f.subplots_adjust(left=xpad, right=1-xpad, top=1-ypad, bottom=ypad)

    #ax.plot(x, y, "o--", color="k")
    #ax.fill_between(x, y-err, y+err, color="k", alpha = 0.2)
    plt.tight_layout()    
    plt.savefig('cell_marker_expression.png', dpi=600)
    plt.show()




