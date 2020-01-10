"""
Simple demo of a scatter plot.
"""
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('classic')

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""
	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:
	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	return base.from_list(cmap_name, color_list, N)


def plot(x, y, z, ax, num, p1, p2):
    s = ax.scatter(x, y, s=1, c=z, linewidths=0.1, alpha=1.0, cmap=discrete_cmap(10, 'jet'))
    protein_name1 = "Protein " + str(p1)
    protein_name2 = "Protein " + str(p2) 
    ax.set_xlabel(protein_name1, fontsize=16) 
    ax.set_ylabel(protein_name2, fontsize=16)
    #cbar=plt.colorbar(ticks=range(num))
    cbar = plt.colorbar(s) 
    cbar.set_clim(-0.5, num - 0.5)
    ax.set_ylim(0.3, 20) 
    ax.set_xlim(0.3, 20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    cbar.remove()  

def plot_(cells, cutoff): 

	matplotlib.rc('font', **font)
	plt.rc('axes', labelsize=16) 

	list_1 = [0, 1, 2, 3] 
	list_2 = [1, 2, 3, 4]

	dx = 1 
	dy = 1

	nrows = 1
	ncols = len(list_1)  

	figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols)) 

	f, ax = plt.subplots(1, ncols, figsize=figsize)
	plt.xscale('log')
	plt.yscale('log') 

	num = 4 
	for j, a in enumerate(ax.flatten()):
		xdata = [] 
		ydata = []
		zdata = [] 
		for i in range(len(cells)):
			xdata.append(cells[i].intensity[list_1[j]]) 
			ydata.append(cells[i].intensity[list_2[j]]) 
			zdata.append(cells[i].cell_type)  

		plot(xdata, ydata, zdata, a, num, list_1[j]+1, list_2[j]+1)            

	pad = 0.02 # Padding around the edge of the figure
	xpad, ypad = dx * pad, dy * pad
	f.subplots_adjust(left=xpad, right=5-xpad, top=5-ypad, bottom=ypad)


	#ax.plot(x, y, "o--", color="k")
	#ax.fill_between(x, y-err, y+err, color="k", alpha = 0.2)
	plt.tight_layout()
	filename = "scatter_protein_expression_" + str(cutoff) + ".png"
	plt.savefig(filename, dpi=900)
	plt.show()

