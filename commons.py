import re
import numpy as np
from scipy import interpolate

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
