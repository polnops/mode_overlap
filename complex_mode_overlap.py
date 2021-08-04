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


def plotComplexData(dat, param_list, bound=2000, scale=1, res=500j):
             
    """handles norm E, complex Ex and Ey, loops through param list"""

    for i in range(len(param_list)): 
        x,y,z = datToxyz(dat,i)

        values, corners = cropData(x, y, z, bound, scale, res)    

        if i%3 == 0:

            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10,2))
            fig.suptitle("w="+str(param_list[i])+':norm(E), Re(Ex), Im(Ex), Re(Ey), Im(Ey)')
            axs[0].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='auto', cmap=plt.get_cmap('Greens'))
  
        elif i%3 == 1:
            axs[1].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='auto', cmap=plt.get_cmap('bwr'))
                
            axs[2].imshow(np.imag(values), origin='lower', extent=corners.tolist(),
                    aspect='auto', cmap=plt.get_cmap('bwr'))

            axs[1].set_yticklabels([])
            axs[2].set_yticklabels([])
            
        elif i%3 == 2:
            axs[3].imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='auto', cmap=plt.get_cmap('bwr'))
                
            axs[4].imshow(np.imag(values), origin='lower', extent=corners.tolist(),
                    aspect='auto', cmap=plt.get_cmap('bwr'))

            axs[3].set_yticklabels([])
            axs[4].set_yticklabels([])
                  
                    

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


def plotCorrvsparams(param_list, xcpeaks):
    param_list = np.array(param_list, dtype=float)
    plt.plot(param_list, xcpeaks, marker='o', linestyle='')
    plt.locator_params(axis="x", nbins=4)
    plt.ylabel("peak correlation")
    plt.xlabel("width (nm)")
    plt.show()


def plotCorr2D(xc_norm, xcfit = [], corners, fitflag):
    plt.imshow(xc_norm, origin='lower', extent=corners.tolist(),
                 aspect='auto', cmap=plt.get_cmap('YlOrRd'))
    plt.colorbar()
    if fitflag:
        plt.contour(np.transpose(xcfit(*np.indices(xc_norm.shape))),
                    origin='lower', extent=corners.tolist(), colors='w')

    plt.show()


def convolveData(dat_ref, dat_list, param_list,
                 bound=2000, scale = 1,
                 res=100j, plotflag1 = False,
                 plotflag2=True, fitflag= False,
                 i=0):

    x = dat_ref[:, 0]
    y = dat_ref[:, 1]
    z = dat_ref[:, 2]
    values_ref, corners_ref = cropData(x, y, z, bound,scale,res)     

    xclist = []
    xcpeaks = []
    for i in range(len(param_list)):

        x = dat_list[:, 0]
        y = dat_list[:, 1]
        z = dat_list[:, 2+i]

        values, corners     = cropData(x, y, z, bound,scale,res)       
        xc_norm = getCorrelation(values_ref, values)

        if fitflag:
            normpeak, xcfit = fitCorrelation(xc_norm)
        else:
            normpeak = np.max(xc_norm)

        xclist.append(xc_norm)
        xcpeaks.append(normpeak)

        if plotflag1:
            plotCorr2D(xc_norm, xcfit, corners, fitflag)

    if plotflag2:
        plotCorrvsparams(param_list, xcpeaks)
