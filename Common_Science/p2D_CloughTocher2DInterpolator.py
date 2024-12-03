"""
CloughTocher2DInterpolator
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
import inspect

def tst_CloughTocher2DInterpolator():
    #-- Данные
    fname="Function = "+inspect.currentframe().f_code.co_name
    print(fname)  # Вывод имени функции
    rng = np.random.default_rng(seed=125)
    x = rng.random(10) - 0.5;  y = rng.random(10) - 0.5
    z = np.hypot(x, y)
    #-- Обработка
    X = np.linspace(min(x), max(x))
    Y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(x, y, "ok", label="input point")
    plt.title(fname);  plt.legend()
    plt.axis("equal"); plt.grid(); plt.colorbar();
    plt.show()

if __name__=="__main__":
    tst_CloughTocher2DInterpolator()