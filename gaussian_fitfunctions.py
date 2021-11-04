import numpy as np
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    rotation = np.deg2rad(rotation)
    center_xp = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_yp = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_xp-xp)/width_x)**2 +
                ((center_yp-yp)/width_y)**2)/2.)
        return g
    return rotgauss


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    sdat = np.sort(data.flatten())
    offset = np.mean(data[:int(.01*len(sdat))])
    col = data[:, int(y)]
    var_x = abs((np.arange(col.size)-y)**2*col).sum()/col.sum()
    width_x = np.sqrt(var_x)
    row = data[int(x), :]
    var_y = abs((np.arange(row.size)-x)**2*row).sum()/row.sum()
    width_y = np.sqrt(var_y)

    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)

    def errorfunction(p): return np.ravel(
        gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
    return matrix, datum

def fitCorrelation(xc_norm):

    gaussianfitparam = fitgaussian(xc_norm)
    xcfit = gaussian(*gaussianfitparam)
    normpeak = gaussianfitparam[0]

    return normpeak, xcfit