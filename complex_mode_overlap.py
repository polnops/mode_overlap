import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gaussian_fitfunctions import *
from commons import *

def dattoxyComplex(dat, n_param, n_dim = 3):
    
    x = np.real(dat[:, 0]); y = np.real(dat[:, 1])

    (m,n)   = np.shape(dat)
    z = np.reshape(dat[:,2:],(m,n_param,n_dim+1))

    return x, y, z

def getComplexCorrelation(E_ref, E_i, i):

        xc = 0j*np.zeros(np.shape(E_ref[:,:,0,1]))

        norm_factor = np.sum(E_ref[:,:,0,0]**2)*np.sum(E_i[:,:,i,0]**2)

        for j in [1,2,3]:
            xc_j = signal.correlate2d(
                np.conj(E_ref[:,:,0,j]), E_i[:,:,i,j]
                , mode="same", boundary='wrap')
            xc += np.abs(xc_j)**2/norm_factor
            
        return np.real(xc)


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

def plotCorrvsparams(param_list, xcpeaks, xcpeaks2 =[]):
    param_list = np.array(param_list, dtype=float)
    plt.plot(param_list, xcpeaks, marker='o', linestyle='', label = "fit")
    if xcpeaks2:  
        plt.plot(param_list, xcpeaks2, marker='o', linestyle='', label = "max")  
    plt.locator_params(axis="x", nbins=4)
    plt.ylabel("peak correlation")
    plt.legend()
    plt.xlabel("width (nm)")
    plt.show()

def convolveComplex(dat_ref, dat, param_list,
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

def datToxyz(dat, i = 0):
    return np.real(dat[:, 0]), np.real(dat[:, 1]), dat[:, 2+i]

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