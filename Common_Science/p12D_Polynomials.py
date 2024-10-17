"""
https://numpy.org/doc/stable/reference/routines.polynomials.html
https://numpy.org/doc/stable/search.html?q=fit
"""

import matplotlib.pyplot as plt
import numpy as np
import random


def noise(k):
    return k + (random.random() * 2 - 1) * 0.15*k


def Poly01():
    """ https://dev.to/eastrittmatter/1d-polynomial-curve-fitting-in-numpy-and-matplotlib-2560 """
    num=50
    x = np.linspace(0, 50, num)
    y = np.linspace(0, 30, num)
    random.seed(125)

    rng = np.random.default_rng(seed=125)
    rnd = rng.uniform(low=-1, high=1,size=num)
    # y1 = np.vectorize(noise)(y)
    y1=y*(1+0.25*rnd)

    lin_fit = np.polyfit(x, y1, 1) # 1 - степень полинома
    lin_model = np.poly1d(lin_fit)
    print('y=kx+b')
    print(f'k={lin_fit[0]}')
    print(f'b={lin_fit[1]}')
    # Коэфф детерминации линейн.функ
    lin_R_squared = np.corrcoef(x, y1)[0, 1] ** 2
    print(f'R**2={lin_R_squared}')

    quad_fit = np.polyfit(x, y1, 2)
    quad_model = np.poly1d(quad_fit)
    print('y=ax**2+bx+c')
    print(f'a={quad_fit[0]:13.10f}')
    print(f'b={quad_fit[1]:13.10f}')
    print(f'c={quad_fit[2]:13.10f}')
    # Коэфф детерминации квадр.функ
    quad_fit = np.polyfit(x, yq, 2, full=True)
    quad_model = np.poly1d(quad_fit[0])
    mean_yq = yq.sum() / yq.size
    quad_tss = np.sum((yq - mean_yq) ** 2)
    quad_R_squared = 1 - quad_fit[1] / quad_tss
    quad_R_squared = quad_R_squared[0]


    plt.plot(x,y,label='ini')
    plt.plot(x,y1,label='noise1 (y1)')
    plt.plot(x, lin_model(x), '--k', label='linear')
    plt.plot(x, quad_model(x), '--r', label='quad')
    plt.grid(); plt.legend()
    plt.show()

Poly01()
