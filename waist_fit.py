import numpy as np
import matplotlib.pyplot as plt
from gaussian_fitfunctions import *
from complex_mode_overlap import *
from commons import *


def plotWaists(param_list, waists):
    param_list = np.array(param_list, dtype=float)
    plt.plot(param_list, waists, marker='o', linestyle='', label = "fit")
    plt.locator_params(axis="x", nbins=4)
    plt.ylabel("waist (nm)")
    plt.legend()
    plt.xlabel("slab width (nm)")
    plt.show()

def plotandfitNorm(dat, param_list, bound=2000, scale=1, res=500j):

    i = 0; waists = []
    while i < len(param_list):   


        x,y,z = datToxyz(dat,i)
        values, corners = cropData(x, y, z, bound, scale, res)    
      
        norm2D = np.real(values)

        plt.figure()
        plt.imshow(norm2D, origin='lower', extent=corners.tolist(),
                aspect='equal', cmap=plt.get_cmap('Greens'))

        waist, normfit = fitData(norm2D)

        waists.append(8*waist*bound/np.abs(res))

        plt.contour(normfit(*np.indices(norm2D.shape)),
            origin='lower', extent=corners.tolist(), colors='w')

        plt.show()

        i += 4

    return waists