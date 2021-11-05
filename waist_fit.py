import numpy as np
import matplotlib.pyplot as plt
from gaussian_fitfunctions import *
from complex_mode_overlap import *
from commons import *


def plotWaists(param_list, xcpeaks, xcpeaks2 =[]):
    param_list = np.array(param_list, dtype=float)
    plt.plot(param_list, waists, marker='o', linestyle='', label = "fit")
    plt.locator_params(axis="x", nbins=4)
    plt.ylabel("waist (nm)")
    plt.legend()
    plt.xlabel("slab width (nm)")
    plt.show()

def fitNorm(dat_ref, dat, param_list,
                 bound=2000, scale = 1,
                 res=100j, plotflag1 = False,
                 plotflag2=True, fitflag= False,
                 i=0):

    param_list = np.unique(np.array(param_list,dtype=float))

    xref, yref, zref = dattoxyComplex(dat_ref, 1)
    values_ref, corners_ref = cropData(xref, yref, zref, bound,scale,res)   
    #values dimension: (res, res, 1, 3)

    n_param = len(param_list); xclist = []; xcpeaks1 = []; xcpeaks2 = []

    x, y, z   = dattoxyComplex(dat, n_param)
    values, corners = cropData(x, y, z, bound, scale, res)
    #values dimension: (res, res, n_param, 3)

    for i in range(n_param):
               
        xc_norm = getComplexCorrelation(values_ref, values, i)

        normpeak1, xcfit = fitCorrelation(xc_norm)
        
        xclist.append(xc_norm)
        xcpeaks1.append(normpeak1)
        xcpeaks2.append(np.max(xc_norm))

        if plotflag1:
            plotCorr2D(xc_norm, corners, fitflag, xcfit = xcfit)

    if plotflag2:
        plotCorrvsparams(param_list, xcpeaks1, xcpeaks2)
        return param_list, xcpeaks1, xcpeaks2


def plotandfitNorm(dat, param_list, bound=2000, scale=1, res=500j):

    i = 0

    while i < len(param_list) and i%4 == 0:         
  
        x,y,z = datToxyz(dat,i)

        values, corners = cropData(x, y, z, bound, scale, res)    
      

            plt.imshow(np.real(values), origin='lower', extent=corners.tolist(),
                    aspect='equal', cmap=plt.get_cmap('Greens'))


            plt.contour(np.transpose(normfit(*np.indices(normfit.shape))),
                        origin='lower', extent=corners.tolist(), colors='w')

            plt.show()

        i += 1