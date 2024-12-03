"""
interpn
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html
"""

import numpy as np
from scipy.interpolate import interpn

def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z
x = np.linspace(0, 4, 5)
y = np.linspace(0, 5, 6)
z = np.linspace(0, 6, 7)
points = (x, y, z)
values = value_func_3d(*np.meshgrid(*points, indexing='ij'))
# Evaluate the interpolating function at a point

point = np.array([2.21, 3.12, 1.15])
print(interpn(points, values, point))  #  [12.63]