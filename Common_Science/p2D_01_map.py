"""
Ivan Genik, 2024
Maps
"""

import matplotlib.pyplot as plt
import numpy as np

def pcolormesh1():
    np.random.seed(19_680_801);  data1 = np.random.randn(25, 25)
    print(data1.min(), data1.max())
    data1[0,0]=10
    print(data1.min(), data1.max())
    print(data1)
    plt.figure(figsize=(13.5,5))  # x и y – ширина и высота рис. в  дюймах
    plt.subplot(1, 2, 1)
    plt.pcolormesh(data1, cmap='plasma', edgecolors='face', shading='flat')
    plt.colorbar(label='Value', drawedges=False)

    plt.subplot(1, 2, 2)
    # vmin и vmax определяют   диапазон   данных,   который   будет   покрыт цветовой картой.
    # По умолчанию цветовая карта охватывает весь диапазон значений отображаемых данных.
    vmin1=-2
    vmax1=1
    vmin1=data1.min()
    vmax1=data1.max()
    plt.pcolormesh(data1, cmap='plasma',
                   edgecolors='k', shading='gouraud', vmin=vmin1,  vmax=vmax1)
    # edgecolors не работает из-за shading='gouraud'
    plt.colorbar(label='', drawedges=True )
    plt.show()

def cntr1():
    x = np.linspace(-1, 1, 50)
    y = x
    z = np.outer(x, y)
    plt.figure(figsize=(5, 5))
    plt.contour(x, y, z)
    plt.show()

def cntr2():
    x = np.linspace(-1, 1, 50)
    y = x
    z = np.outer(x, y)
    plt.figure(figsize=(5, 5))
    lvl = np.linspace(-1, 1, 11);    print(lvl)
    # [-1.  -0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8  1. ]
    curves = plt.contour(x, y, z, lvl)
    plt.clabel(curves)
    plt.title(r'$z=xy$', fontsize=20)
    plt.show()

def cntr3():
    x = np.linspace(-1, 1, 50)
    y = x
    z = np.outer(x, y)
    fig = plt.figure(figsize=(10, 5)); lvl = np.linspace(-1, 1, 11)
    fig.add_subplot(121); curves = plt.contourf(x, y, z)
    fig.add_subplot(122); curves1 = plt.contour(x, y, z, lvl, colors='k')
    # https://matplotlib.org/stable/api/contour_api.html#matplotlib.contour.ContourLabeler.clabel
    plt.clabel(curves1, fontsize=10, colors = 'k')
    plt.show()

def cntr4():
    x = np.linspace(-1, 1, 50)
    y = x
    z = np.outer(x, y)
    fig = plt.figure(figsize=(15, 5));  lvl = np.linspace(-1, 1, 11)
    fig.add_subplot(131); curves = plt.contourf(x, y, z)
    fig.add_subplot(132); curves1 = plt.contour(x, y, z, lvl, colors='k')
    # https://matplotlib.org/stable/api/contour_api.html#matplotlib.contour.ContourLabeler.clabel
    plt.clabel(curves1, fontsize=10, colors = 'k')
    fig.add_subplot(133)
    curves2 = plt.contourf(x, y, z, lvl)
    curves3 = plt.contour(x, y, z, lvl, colors='k')
    plt.clabel(curves3, fontsize=10, colors='k')
    plt.show()

    curves2 = plt.contourf(x, y, z, lvl)
    plt.colorbar(label='', drawedges=True, orientation='vertical') # сразу после нужной карты
    curves3 = plt.contour(x, y, z, lvl, colors='k')
    plt.clabel(curves3, fontsize=10, colors='k')
    plt.show()

def cntr5():
    # координаты x, y - формируем случайным образом
    x = np.random.rand(100) * 4*np.pi - 2*np.pi
    y = np.random.rand(100) * 4*np.pi - 2*np.pi
    z = np.sin(x) * np.sin(y) / (1+np.abs(x * y))
    fig, ax = plt.subplots()
    # спец.функция при таком способе задания точек
    # на входе принимает одномерные векторы координат x, y, z.
    # триангуляция с линейной интерполяцией
    c1 = ax.tricontour(x, y, z, cmap='plasma')
    c1.clabel(colors='k', fmt='%.2f')
    plt.show()

def sphere_function():
    def f(x, y):
        return x ** 2 + y ** 2
    X = np.linspace(-32,32)
    Y = np.linspace(-32,32)
    x,y = np.meshgrid(X,Y)
    F = f(x,y)
    minF = F.min()
    maxF = F.max()
    print(minF,maxF)

    n_iso = 15 # num isolines
    fig =plt.figure(figsize=(10,8))
    curves1 = plt.contourf(x,y, F, n_iso)
    curves3 = plt.contour(x, y, F, n_iso, colors='k')
    fig.colorbar(curves1)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.title('Sphere function'); plt.show()

import sys
import matplotlib.tri as tri
def contour_plot_of_irregularly_spaced_data():
    # Часть 1 - задание функции на случайных точках
    np.random.seed(19680801)
    npts = 200 # число случайных точек по х и y
    x = np.random.uniform(-2, 2, npts)
    y = np.random.uniform(-2, 2, npts)
    z = x * np.exp(-x ** 2 - y ** 2)
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    # Часть 2 - интерполяция на сетку
    ngridx = 100 # число точек по х оси регулярной сетки
    ngridy = 200 # число точек по y оси регулярной сетки
    xi = np.linspace(-2.1, 2.1, ngridx)
    yi = np.linspace(-2.1, 2.1, ngridy)
    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y) # триангуляция - создание сетки трегольников (СТ)
    interpolator = tri.LinearTriInterpolator(triang, z) # линейная интерполяция на СТ
    Xi, Yi = np.meshgrid(xi, yi) # создание регулярной сетки
    zi = interpolator(Xi, Yi) # перенос линейной интерполяции с СТ на регулярную сеть
    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    # from scipy.interpolate import griddata
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    # Часть 3 - картопостроение - 1-часть для регулярной сетки
    ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax1)
    ax1.plot(x, y, 'ko', ms=3)
    ax1.set(xlim=(-2, 2), ylim=(-2, 2))
    ax1.set_title(f'grid and contour ({npts} points, {ngridx * ngridy} grid points)')

    # Часть 4 - картопостроение - 2-часть для исходной нерегулярной сетки
    ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr2, ax=ax2)
    ax2.plot(x, y, 'ko', ms=3)
    ax2.set(xlim=(-2, 2), ylim=(-2, 2))
    ax2.set_title(f'tricontour ({npts} points)')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def compare_lin_interpolations():
    # Часть 1 - задание функции на случайных точках
    np.random.seed(19680801)
    npts_ = [10, 50, 200]; nx=len(npts_)
    ngridx_ = [10, 100]
    ngridy_ = [10, 200]
    # число случайных точек по х и y
    n=1
    fig = plt.figure(figsize=(18, 8))
    plt.suptitle('Linear interpolation')
    for i in npts_:
        x = np.random.uniform(-2, 2, i)
        y = np.random.uniform(-2, 2, i)
        z = x * np.exp(-x ** 2 - y ** 2)
        # Часть 2 - интерполяция на сетку
        for j in range(2):
            ngridx = ngridx_[j] # число точек по х оси регулярной сетки
            ngridy = ngridy_[j] # число точек по y оси регулярной сетки
            xi = np.linspace(-2.1, 2.1, ngridx)
            yi = np.linspace(-2.1, 2.1, ngridy)
            triang = tri.Triangulation(x, y) # триангуляция - создание сетки трегольников (СТ)
            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            interpolator = tri.LinearTriInterpolator(triang, z) # линейная интерполяция на СТ
            Xi, Yi = np.meshgrid(xi, yi) # создание регулярной сетки
            zi = interpolator(Xi, Yi) # перенос линейной интерполяции с СТ на регулярную сеть
            fig.add_subplot(2, nx, n); n=n+1
            plt.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
            cntr1 = plt.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
            fig.colorbar(cntr1)
            plt.plot(x, y, 'ko', ms=3)
            plt.grid()
            plt.title(f'{i} points, {ngridx * ngridy} grd pnts')
    plt.show()

def cmp2():
    """
    https://www.demo2s.com/python/python-matplotlib-tri-cubictriinterpolator.html
    ==============
    Triinterp Demo
    ==============
    Interpolation from triangular grid to quad grid.
    """
    # Create triangulation.
    x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5])
    y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 3.0])
    triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
                 [5, 6, 8], [5, 8, 7], [7, 8, 9]]
    triang = tri.Triangulation(x, y, triangles)

    # Interpolate to regularly-spaced quad grid.
    z = np.cos(1.5 * x) * np.cos(1.5 * y)
    xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))

    interp_lin = tri.LinearTriInterpolator(triang, z)
    zi_lin = interp_lin(xi, yi)

    interp_cubic_geom = tri.CubicTriInterpolator(triang, z, kind='geom')
    zi_cubic_geom = interp_cubic_geom(xi, yi)

    interp_cubic_min_E = tri.CubicTriInterpolator(triang, z, kind='min_E')
    zi_cubic_min_E = interp_cubic_min_E(xi, yi)

    # Set up the figure
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs = axs.flatten()

    # Plot the triangulation.
    axs[0].tricontourf(triang, z)
    axs[0].triplot(triang, 'ko-')
    axs[0].set_title('Triangular grid')

    # Plot linear interpolation to quad grid.
    axs[1].contourf(xi, yi, zi_lin)
    axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    axs[1].set_title("Linear interpolation")

    # Plot cubic interpolation to quad grid, kind=geom
    axs[2].contourf(xi, yi, zi_cubic_geom)
    axs[2].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    axs[2].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    axs[2].set_title("Cubic interpolation,\nkind='geom'")

    # Plot cubic interpolation to quad grid, kind=min_E
    axs[3].contourf(xi, yi, zi_cubic_min_E)
    axs[3].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    axs[3].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    axs[3].set_title("Cubic interpolation,\nkind='min_E'")

    fig.tight_layout()
    plt.show()

if __name__=="__main__":
    # pcolormesh1()
    # cntr1()
    # cntr5()
    # sphere_function()
    # contour_plot_of_irregularly_spaced_data()
    compare_lin_interpolations()
    # cmp2()

