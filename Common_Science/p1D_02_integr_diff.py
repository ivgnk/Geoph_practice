"""
2024
General geophysical (and general scientific) tasks in Python
Service for p1D_02.py
Integration, differentiation of 1D
"""
import numpy as np
import matplotlib.pyplot as plt
import p1D_02

from scipy.integrate import quad

def func_integr():
    # x = np.arange(-10.0, 10.1, 0.2)

    def integrand(x):
        return -x + x ** 2 - x ** 3

    a = 2
    b = 1
    I = quad(integrand, 0, 1, args=(a, b))


if __name__=='__main__':
    pass

