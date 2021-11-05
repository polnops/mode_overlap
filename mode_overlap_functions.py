import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gaussian_fitfunctions import *
from commons import *


def datToxyz(dat, i = 0):
    return np.real(dat[:, 0]), np.real(dat[:, 1]), dat[:, 2+i]


def plotData(dat, bound=4000, scale=1,
             res=500j, plotflag=True, i=0):

    x,y,z = datToxyz(dat,i)

    values, corners = cropData(x, y, z, bound, scale, res)

    if plotflag:
        plt.imshow(values, origin='lower', extent=corners.tolist(),
                   aspect='auto', cmap=plt.get_cmap('YlOrRd'))
        plt.show()


                  
def getCorrelation(values_ref, values):

        xc = signal.correlate2d(
            values_ref, values, mode="same", boundary='wrap')
        xc_norm = xc**2/np.sum(values_ref**2)/np.sum(values**2)

        return xc_norm

def plotCorrvsparams(param_list, xcpeaks):
    param_list = np.array(param_list, dtype=float)
    plt.plot(param_list, xcpeaks, marker='o', linestyle='')
    plt.locator_params(axis="x", nbins=4)
    plt.ylabel("peak correlation")
    plt.xlabel("width (nm)")
    plt.show()


def plotCorr2D(xc_norm, xcfit, corners, fitflag):
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

        values, corners = cropData(x, y, z, bound,scale,res)       
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
