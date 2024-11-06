"""
irregularly-gridded-data
https://matplotlib.org/stable/plot_types/index.html#irregularly-gridded-data
"""

import matplotlib.pyplot as plt
import numpy as np

def the_tricontour():
    np.random.seed(125)
    x = np.random.uniform(-3, 3, 256)
    y = np.random.uniform(-3, 3, 256)
    z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    levels = np.linspace(z.min(), z.max(), 7)
    print('levels=\n', levels)
    fig, ax = plt.subplots()
    plt.title('tricontour, tricontourf')
    # https://matplotlib.org/stable/api/tri_api.html#matplotlib.tri.TriContourSet
    # matplotlib.tri.TriContourSet - результат tricontour, tricontourf, нет хорошего описания
    # stackoverflow.com/questions/18560319/get-numpy-array-of-matplotlib-tricontourf
    ax.plot(x, y, 'o', markersize=2, color='lightgray')
    tcf=ax.tricontourf(x, y, z, levels=levels)
    fig.colorbar(tcf)
    ax.tricontour(x, y, z, levels=levels, colors='k')
    # ax.set(xlim=(-3, 3), ylim=(-3, 3))
    plt.grid(); plt.show()

def the_tripcolor():
    np.random.seed(125)
    x = np.random.uniform(-3, 3, 256)
    y = np.random.uniform(-3, 3, 256)
    z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    fig, ax = plt.subplots()
    plt.title('the_tripcolor')
    ax.plot(x, y, 'o', markersize=2, color='grey')
    tc=ax.tripcolor(x, y, z)
    fig.colorbar(tc)
    plt.grid(); plt.show()

def the_triplot():
    np.random.seed(125)
    x = np.random.uniform(-3, 3, 256)
    y = np.random.uniform(-3, 3, 256)
    z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    fig, ax = plt.subplots()
    plt.title('the_triplot')
    ax.plot(x, y, 'o', markersize=2, color='grey')
    lines, markers =ax.triplot(x, y) # markers здесь пустой
    print('xd,yd')
    xd=lines.get_xdata(); yd=lines.get_ydata()
    for i in range(len(xd)):  print(i, xd[i], yd[i])
    plt.grid(); plt.show()


if __name__=="__main__":
    # the_tricontour()
    # the_tripcolor()
    the_triplot()