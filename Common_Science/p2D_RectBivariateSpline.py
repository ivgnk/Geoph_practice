"""
RectBivariateSpline
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html
"""

from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D
def tst_RectBivariateSpline():
    # Регулярная грубая сетка
    dx, dy = 0.4, 0.4;  xmax, ymax = 2, 4
    nm=['Грубая сетка', 'Тонкая сетка']
    x = np.arange(-xmax, xmax, dx)
    y = np.arange(-ymax, ymax, dy)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(2 * X) ** 2 - (Y / 2) ** 2)
    interp_spline = RectBivariateSpline(y, x, Z)
    # Регулярная тонкая сетка
    dx2, dy2 = 0.16, 0.16
    x2 = np.arange(-xmax, xmax, dx2)
    y2 = np.arange(-ymax, ymax, dy2)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = interp_spline(y2, x2)
    # fig = plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'}, figsize=(14, 8))
    plt.suptitle('RectBivariateSpline')
    ax[0].plot_wireframe(X, Y, Z, color='b')
    ax[1].plot_wireframe(X2, Y2, Z2, color='r')
    for i, axes in enumerate(ax):
        axes.set_zlim(-0.2, 1)
        axes.set_axis_off()
        axes.set_title(nm[i])
    fig.tight_layout() # автоматическая настройка расстояния между осями и фигурами
    plt.show()


if __name__=="__main__":
    tst_RectBivariateSpline()
