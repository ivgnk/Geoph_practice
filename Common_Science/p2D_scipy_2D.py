"""
griddata
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def tst_griddata():
    def func(x, y):
        return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
    np.random.seed(125)
    rng = np.random.default_rng()
    points = rng.random((1000, 2))
    values = func(points[:,0], points[:,1])
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    print(grid_x)
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    plt.subplot(221)
    plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
    plt.plot(points[:,0], points[:,1], 'k.', ms=1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    plt.title('Nearest')
    plt.subplot(223)
    plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    plt.title('Linear')
    plt.subplot(224)
    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    plt.title('Cubic')
    plt.gcf().set_size_inches(6, 6)
    plt.show()

from icecream import ic
def tst_griddata2():
    def func(x, y):
        return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
    np.random.seed(125)
    rng = np.random.default_rng()
    pnt_rnd = rng.random((1000, 2))
    xrnd, yrnd = pnt_rnd[:,0], pnt_rnd[:,1]
    ic(np.max(xrnd), np.max(yrnd), np.min(xrnd), np.min(yrnd))
    valrnd = func(xrnd, yrnd)
    xreg=np.linspace(0,1,101)
    yreg=np.linspace(0,1,101)
    # pnt_reg=np.array([xreg, yreg])
    grid_z0 = griddata((xrnd, yrnd), valrnd, (xreg[None, :], yreg[:, None]), method='nearest')
    grid_z1 = griddata((xrnd, yrnd), valrnd, (xreg[None, :], yreg[:, None]), method='linear')
    grid_z2 = griddata((xrnd, yrnd), valrnd, (xreg[None, :], yreg[:, None]), method='cubic')
    #--------------- 1
    plt.subplot(221)
    plt.grid()
    plt.scatter(xrnd, yrnd,8)
    plt.title('Original pnt')
    #--------------- 2
    plt.subplot(222)
    plt.grid()
    plt.contour(xreg, yreg, grid_z0, 15, linewidths=0.5, colors='k')
    plt.contourf(xreg, yreg, grid_z0, 15, cmap=plt.cm.jet)
    plt.colorbar()  # draw colorbar
    plt.title('Nearest')
    #--------------- 3
    plt.subplot(223)
    plt.grid()
    plt.title('Linear')
    plt.contour(xreg, yreg, grid_z1, 15, linewidths=0.5, colors='k')
    plt.contourf(xreg, yreg, grid_z1, 15, cmap=plt.cm.jet)
    plt.colorbar()  # draw colorbar
    #--------------- 3
    plt.subplot(224)
    plt.grid()
    plt.title('Cubic')
    plt.contour(xreg, yreg, grid_z2, 15, linewidths=0.5, colors='k')
    plt.contourf(xreg, yreg, grid_z2, 15, cmap=plt.cm.jet)
    plt.colorbar()  # draw colorbar
    
    plt.gcf().set_size_inches(10, 8)
    plt.show()

def tst_griddata3():
    import numpy as np
    from scipy.interpolate import griddata

    # СозданиеSample данных
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    # Создание точек для интерполяции
    points = np.random.rand(100, 2) * 10
    values = np.sin(points[:, 0]) * np.cos(points[:, 1])

    # Выполнение интерполяции
    grid_z = griddata(points, values, (X, Y), method='linear')

    # Печать результата (для демонстрации будет напечатан только фрагмент)
    print(grid_z[49:51, 49:51])

def tst_griddata4():
    # your data
    # https://stackoverflow.com/questions/10723413/scattered-x-y-z-via-pythons-matplotlib-pyplot-contourf
    x = [0, 0, 3, 7, 9, 10, 10, 10]
    y = [0, 10, 1, 4, 5 , 3, 10, 0]
    z = [24, 13, 20, 3, 7 , 8, 12, 14]
    zmax=max(z)
    # define grid.
    xi = np.linspace(0, 10, 300)
    yi = np.linspace(0, 10, 300)
    # grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(x, y, marker='o', c='k', s=12)
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.title('griddata test (%d points)' % len(x))
    plt.show()

if __name__=="__main__":
    tst_griddata2()