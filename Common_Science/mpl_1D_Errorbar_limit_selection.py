"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/errorbar_limits_simple.html#sphx-glr-gallery-lines-bars-and-markers-errorbar-limits-simple-py
"""
import matplotlib.pyplot as plt
import numpy as np

def errorbar_limits_simple():
    d = 0.08 # Ошибка, доли 1
    n = 41 # Число точек
    plt.title(f'Кривые с диапазонами ошибок')
    x = np.arange(n)
    y = 2.5 * np.sin(x / 20 * np.pi)
    dy=y*d # все значения ошибок
    # Как меняются ощибки: начальная, конечная, шаг
    yerr = np.linspace(d, d, n)
    yerr2 = np.linspace(d, d*5, n)

    plt.errorbar(x, y + dy, yerr=yerr, label='1 - const' )
    plt.errorbar(x, y+1 + dy, yerr=yerr2, label='2 - var')
    plt.grid(); plt.legend();  plt.show()

if __name__=="__main__":
    errorbar_limits_simple()