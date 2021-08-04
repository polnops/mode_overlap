import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from gaussian_fitfunctions import *


def loadData(filename='normE_all.dat', param_name="w_slab"):

    file = open(filename)
    lst = []
    for line in file:
        lst += [line.split()]

    param_list = [re.sub(r'\D', "", param) for param in lst[8]
                  if param.startswith(param_name + "=")]

    dat = convertStrtocomplex(lst)

    return dat, param_list

def convertStrtocomplex(lst):
    b = np.char.replace(np.array(lst[9:]),'i','j')
    return np.array(b,complex)

def datToxyz(dat, i = 0):
    return np.real(dat[:, 0]), np.real(dat[:, 1]), dat[:, 2+i]


def cropData(x, y, z, bound, scale, res):

    mask = (x < bound)*(x > -bound)*(y < bound)*(y > -bound)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    # some of these lines below may be redundant for rectangular grid
    corners = scale*np.array([min(y), max(y), min(x), max(x)])

    grid_x, grid_y = np.mgrid[corners[0]:corners[1]:res,
                              corners[2]:corners[3]:res]

    values = interpolate.griddata((y, x), z, (grid_x, grid_y),
                                  method='nearest')

    return values, corners

                  
                  

def getCorrelation(values_ref, values):

        xc = signal.correlate2d(
            values_ref, values, mode="same", boundary='wrap')
        xc_norm = xc**2/np.sum(values_ref**2)/np.sum(values**2)

        return xc_norm


def fitCorrelation(xc_norm):

    gaussianfitparam = fitgaussian(xc_norm)
    xcfit = gaussian(*gaussianfitparam)
    normpeak = gaussianfitparam[0]

    return normpeak, xcfit



def plotCorr2D(xc_norm, corners, fitflag, xcfit = []):
    plt.imshow(xc_norm, origin='lower', extent=corners.tolist(),
                 aspect='auto', cmap=plt.get_cmap('YlOrRd'))
    plt.colorbar()
    if fitflag:
        plt.contour(np.transpose(xcfit(*np.indices(xc_norm.shape))),
                    origin='lower', extent=corners.tolist(), colors='w')

    plt.show()


def plotComplexData(dat, param_list, bound=2000, scale=1, res=500j):
             
    """handles norm E, complex Ex and Ey, loops through param list"""

    for i in range(len(param_list)): 
        x,y,z = datToxyz(dat,i)

        values, corners = cropData(x, y, z, bound, scale, res)    

        if i%4 == 0:

            fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(12,2))
            fig.suptitle("w="+str(param_list[i])+
            ':norm(E), Re(Ex), Im(Ex), Re(Ey), Im(Ey), Re(Ez), Im(Ez)')
            axs[0].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('Greens'))
  
        elif i%4 == 1:
            axs[1].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('bwr'))
                
            axs[2].imshow(np.imag(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('bwr'))

            axs[1].set_yticklabels([])
            axs[2].set_yticklabels([])
            
        elif i%4 == 2:
            axs[3].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('bwr'))
                
            axs[4].imshow(np.imag(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('bwr'))

            axs[3].set_yticklabels([])
            axs[4].set_yticklabels([])

        elif i%4 == 3:
            axs[5].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('bwr'))
                
            axs[6].imshow(np.imag(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('bwr'))

            axs[5].set_yticklabels([])
            axs[6].set_yticklabels([])