import numpy as np
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y, rotation, offset):
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
                ((center_yp-yp)/width_y)**2)/2.) + offset
        return g
    return rotgauss


# def moments(data):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution by calculating its
#     moments """
#     total = data.sum()
#     X, Y = np.indices(data.shape)
#     x = (X*data).sum()/total
#     y = (Y*data).sum()/total
#     sdat = np.sort(data.flatten())
#     offset = np.mean(data[:int(.01*len(sdat))])
#     if y < 0:
#         col = data[:, int(0)]
#     elif y > data.shape[1]:
#         col = data[:, int(data.shape[1]-1)]
#     else:
#         col = data[:, int(y)]
#     var_x = abs((np.arange(col.size)-y)**2*col).sum()/col.sum()
#     if var_x > 0:
#         width_x = np.sqrt(var_x)
#     else:
#         width_x = data.shape[0]
#     if x < 0:
#         row = data[int(0), :]
#     elif x > data.shape[0]:
#         row = data[int(data.shape[0]-1), :]
#     else:
#         row = data[int(x), :]
#     var_y = abs((np.arange(row.size)-x)**2*row).sum()/row.sum()
#     if var_y > 0:
#         width_y = np.sqrt(var_y)
#     else:
#         width_y = data.shape[1]
#     height = data.max()
#     return height, x, y, width_x, width_y, 0.0, offset

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
    return height, x, y, width_x, width_y, 0.0, offset


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)

    def errorfunction(p): return np.ravel(
        gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
    return matrix, datum
