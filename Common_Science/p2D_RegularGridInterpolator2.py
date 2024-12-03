"""
RegularGridInterpolator
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
"""

from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt

def tst_RegularGridInterpolator2():
    #--- 1 Calc
    x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
    def ff(x, y):
        return x ** 2 + y ** 2
    xg, yg = np.meshgrid(x, y, indexing='ij')
    data = ff(xg, yg)
    interp = RegularGridInterpolator((x, y), data,
                                     bounds_error=False, fill_value=None)
    #--- 2 Visu
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
               s=60, c='k', label='data')

    # Evaluate and plot the interpolator on a finer grid
    xx = np.linspace(-4, 9, 31)
    yy = np.linspace(-4, 9, 31)
    X, Y = np.meshgrid(xx, yy, indexing='ij')
    # interpolator
    ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
                      alpha=1.0, color='r', label='linear interp')
    # ground truth
    ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
                      alpha=1.0, color='b', label='ground truth')
    plt.legend(); plt.show()

if __name__=="__main__":
    tst_RegularGridInterpolator2()
