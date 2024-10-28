"""
numpy.meshgrid — создание прямоугольной сетки из набора одномерных массивов координат.
https://numpy.org/doc/stable/user/how-to-partition.html#mgrid
"""
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

def tst_meshgrid():
    x = np.array([1, 2, 3])
    y = np.array([0, 1, 2, 3, 4, 5])
    xx, yy = np.meshgrid(x, y)
    print('xx=\n',xx)
    print('yy=\n',yy)
    plt.plot(xx, yy, marker='.', color='k', linestyle='none')
    plt.grid(); plt.show()

def tst_meshgrid_mgrid():
    xx, yy = np.meshgrid(np.array([0, 1, 2, 3]),
                         np.array([0, 1, 2, 3, 4, 5]))
    print('xx.T=\n', xx.T) # транспонированные
    print('yy.T=\n', yy.T)
    print('np.mgrid = \n',np.mgrid[0:4, 0:6])

def tst_meshgrid_mgrid_withF():
    def f(x, y):  return x ** 2 + y ** 2
    xmi=-32; xma=32
    ymi=0; yma=64
    xma1=xma+1; yma1=yma+1
    X = np.linspace(xmi, xma, num=65) # конец (32) включен
    Y = np.linspace(ymi, yma, num=65) # конец (32) включен
    # --- 1 var
    x, y = np.meshgrid(X, Y)
    F1 = f(x, y)
    # --- 2 var
    mgr = np.mgrid[xmi:xma1, ymi:yma1]
    F2= mgr[0]**2 + mgr[1]**2

    ic(np.array_equal(F1, F2.T))

    n_iso = 15  # num isolines
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title('meshgrid')
    curves1 = plt.contourf(x, y, F1, n_iso)
    curves2 = plt.contour(x, y, F1, n_iso, colors='k')
    fig.colorbar(curves1); plt.grid()
    plt.subplot(1, 2, 2)
    plt.title('mgrid')
    curves3 = plt.contourf(mgr[0], mgr[1], F2, n_iso)
    curves4 = plt.contour(mgr[0], mgr[1], F2, n_iso, colors='k')
    fig.colorbar(curves3); plt.grid()
    plt.show()

def tst_meshgrid_out():
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    x = np.linspace(-5, 5, 201)
    y = np.linspace(0, 10, 101)
    xs, ys = np.meshgrid(x, y)
    zs = np.sqrt(xs ** 2 + ys ** 2)
    n_iso = 15
    fig = plt.figure(figsize=(8, 6))
    curves1 = plt.contourf(xs, ys, zs, n_iso)
    curves2 = plt.contour(xs, ys, zs, n_iso, colors='k')
    ic(zs.shape)
    fig.colorbar(curves1)
    plt.grid(); plt.show()
    with (open('x-y-z.txt','w+') as f):
        for i,x1 in enumerate(x):
            for j,y1 in enumerate(y):
                f.write(f'{x1:0.2f} {y1:0.2f} {zs[j,i]} \n')


if __name__=="__main__":
    # tst_meshgrid()
    # tst_meshgrid_mgrid()
    # tst_meshgrid_mgrid_withF()
    tst_meshgrid_out()