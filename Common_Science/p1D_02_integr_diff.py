"""
2024
General geophysical (and general scientific) tasks in Python
Service for p1D_02.py
Integration, differentiation of 1D
"""
import numpy as np
import matplotlib.pyplot as plt
import p1D_02

# https://docs.scipy.org/doc/scipy/tutorial/integrate.html
from scipy.integrate import quad
def integr1D_quad():
    # x = np.arange(-10.0, 10.1, 0.2)
    def integrand(x,a,b,c):
        return a*x + b*x ** 2 +c*x ** 3
    a = -10
    coeff=(-1,1,-1)
    for i in range(-10,11,1):
        b = i
        res = quad(integrand, a, b, args=coeff)
        print(f"{i:4} {b:4}  {res[0]:12.4f}   {res[1]:16.14f}")

from scipy import integrate
def integr1D_simps():
    x = np.arange(-10.0, 12, 1)
    y = -x + x ** 2 - x ** 3
    for i in range(1,len(x)):
        yi=y[0:i]
        xi=x[0:i]
        res = integrate.simpson(yi, x=xi)
        print(f"{i:4} {xi[-1]:7.2f} {res:12.4f}")

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html
from scipy.interpolate import InterpolatedUnivariateSpline
def integr1D_spl():
    pass

def diff1D():
    x = np.arange(-10.0, 10.1, 0.05)
    y = -x + x ** 2 - x ** 3
    lab=['Исходная функция','Дифференцированная функция']
    res=np.diff(y)
    xres=[(x[i]+x[i-1])/2 for i in range(1,len(x))]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x,y,'g-',   label=lab[0])
    ax2.plot(xres, res,'b-', label=lab[1])
    ax1.set_xlabel('X')

    ax1.set_ylabel(lab[0], color='g')
    ax2.set_ylabel(lab[1], color='b')

    ax1.grid(); plt.show()


if __name__=='__main__':
    # integr1D_quad()
    # integr1D_simps()
    diff1D()
