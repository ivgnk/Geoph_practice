"""
RBFInterpolator
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import inspect
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html

# RBFInterpolator(y, d, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)[source]
krn_1 = ['cubic', 'thin_plate_spline', 'linear', 'quintic']
eps_1 = [1]*len(krn_1)
krn_2 = ['multiquadric', 'inverse_multiquadric', 'inverse_quadratic', 'gaussian']
eps_2 = [1, 1, 1, 1]
krn=krn_1+krn_2
eps=eps_1+eps_2
def tst_RBFInterpolator2():
    # Данные
    fname = "Function = " + inspect.currentframe().f_code.co_name
    rng = np.random.default_rng(125)
    nn=100
    x = 2*(rng.random(nn) - 0.5); y = 2*(rng.random(nn) - 0.5)
    xobs=np.zeros((100,2)); xobs[:,0]=x; xobs[:,1]=y
    yobs = np.sum(xobs, axis=1) * np.exp(-6 * np.sum(xobs ** 2, axis=1))
    plt.figure(figsize=(18, 12))  # x и y – ширина и высота рис. в  дюймах
    plt.suptitle(fname)
    for i in range(len(krn)):
        # Интерполирование
        xgrid = np.mgrid[-1:1:50j, -1:1:50j]
        xflat = xgrid.reshape(2, -1).T
        yflat = RBFInterpolator(xobs, yobs, kernel=krn[i],epsilon=eps[i])(xflat)
        ygrid = yflat.reshape(50, 50)
        # print(krn[i], 'good');  print('----------')
        plt.subplot(2, 4, i + 1)  # 2 - количество строк; 1 - количество столбцов; 1 - индекс ячейки в которой работаем
        # fig, ax = plt.subplots()
        plt.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
        p = plt.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
        plt.title(krn[i])
        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=8)
        cbar = plt.colorbar(p);  cbar.ax.tick_params(labelsize=8)
        plt.grid()
    plt.show()
    # print('End '+fname)

if __name__=="__main__":
    tst_RBFInterpolator2()