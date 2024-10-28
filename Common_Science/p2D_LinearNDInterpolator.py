import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from icecream import ic

def tst_LinearNDInterpolator():
    rng = np.random.default_rng(seed=125)
    x = rng.random(10) - 0.5;  y = rng.random(10) - 0.5
    z = np.hypot(x, y)
    X = np.linspace(min(x), max(x));   Y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)

    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title('pcolormesh')
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(x, y, "ok", label="input point")
    plt.colorbar(); plt.legend(); plt.grid()
    # одинаковый масштаб для вертикальной и горизонтальной осей
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    n_iso = 15  # num isolines
    plt.title('contourf+contour')
    curves1 = plt.contourf(X, Y, Z, n_iso)
    curves2 = plt.contour(X, Y, Z, n_iso, colors='k')
    plt.plot(x, y, "oy", label="input point")
    fig.colorbar(curves1); plt.legend(); plt.grid()
    plt.axis("equal"); plt.show()

from math import floor, ceil
def tst_LinearNDInterpolator_export():
    rng = np.random.default_rng(seed=125)
    x = rng.random(10) - 0.5;  y = rng.random(10) - 0.5
    z = np.hypot(x, y)
    mi_x=floor(min(x)) # округляет о ближайшего меньшего целого
    ma_x=ceil(max(x))  # округляет о ближайшего большего целого
    ic(mi_x, ma_x)
    mi_y=floor(min(y));  ma_y=ceil(max(y))
    ic(mi_y, ma_y)
    # ny=ma_y-mi_y+1
    step=0.1
    x1 = np.arange(mi_x, ma_x+step, step)
    y1 = np.arange(mi_y, ma_y+step, step)
    ic(x); ic(x1)
    X, Y = np.meshgrid(x1, y1)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)

    with (open('LinearNDInterpolator2D.txt','w+') as f):
        for i,x1_ in enumerate(x1):
            for j,y1_ in enumerate(y1):
                f.write(f'{x1_:0.2f} {y1_:0.2f} {Z[j,i]} \n')

if __name__=="__main__":
    # tst_LinearNDInterpolator()
    tst_LinearNDInterpolator_export()