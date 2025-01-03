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
    # Часть 1
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

    # Часть 2
    quad_fit = np.polyfit(x, y1, 2, full=True) # True - полный набор
    quad_model = np.poly1d(quad_fit[0])
    print('y=ax**2+bx+c')
    print(f'a={quad_fit[0][0]:13.10f}')
    print(f'b={quad_fit[0][1]:13.10f}')
    print(f'c={quad_fit[0][2]:13.10f}')
    # Коэфф детерминации квадр.функ
    mean_yq = y1.sum() / y1.size
    quad_tss = np.sum((y1 - mean_yq) ** 2)
    quad_R_squared = 1 - quad_fit[1] / quad_tss
    quad_R_squared = quad_R_squared[0]  # нужен только этот элемент
    print(f'R**2={quad_R_squared}')

    plt.plot(x,y,label='ini')
    plt.plot(x,y1,label='noise1 (y1)')
    plt.plot(x, lin_model(x), '--k', label='linear')
    plt.plot(x, quad_model(x), '--r', label='quad')
    plt.grid(); plt.legend(); plt.show()

def Poly02_def():
    z=np.poly([-1, 1, 1, 10])
    print(z)
    z2=np.roots([1, -11,   9,  11, -10])
    print(z2)

def Poly03_calc():
    """
    https://numpy.org/doc/stable/reference/generated/numpy.polyval.html#numpy-polyval
    numpy.polyval(p,x) – вычисление полинома в точках x, p - коэффициенты полинома
    """
    # Part 1
    num=10; x = np.linspace(start=0, stop=50, num=num) # от 0 до 50, включая 50
    coeff=[3, 0, 1]
    n5=np.poly1d(5)
    res1=np.polyval(coeff, 5)  # 3 * 5**2 + 0 * 5**1 + 1 = 76
    res2=np.polyval(coeff, n5)
    res3=np.polyval(np.poly1d(coeff), 5)
    res4=np.polyval(np.poly1d(coeff), n5)
    print(res1); print(res2); print(res3); print(res4)
    # Part 2
    # --- poly1d
    print(n5,type(n5),'\n')
    r = np.poly1d(coeff)
    print(r)
    #--- График
    y= np.polyval(coeff, x)
    plt.plot(x,y); plt.plot(x,y,'ro')
    plt.grid(); plt.show()

def Poly04_diff():
    """ https://numpy.org/doc/stable/reference/generated/numpy.polyder.html """
    num=51; x = np.linspace(start=-6, stop=6, num=num)
    coeff=[3, 0, 1]
    ppol = np.poly1d(coeff)
    pdif = np.polyder(ppol)
    print('ppol=','\n',ppol,'\n')
    print('pdif =','\n',pdif)
    y = np.polyval(ppol, x)
    yd = np.polyval(pdif, x)
    plt.plot(x,y, label='ini')
    plt.plot(x,yd, label='diff')
    plt.grid(); plt.legend();  plt.show()

def Poly05_intgr():
    """ https://numpy.org/doc/stable/reference/generated/numpy.polyint.html """
    num=51; x = np.linspace(start=-6, stop=6, num=num)
    coeff=[3, 0, 1]
    ppol = np.poly1d(coeff)

    pint = np.polyint(ppol)
    print('ppol=','\n',ppol,'\n')
    print('pdif =','\n',pint)
    y = np.polyval(ppol, x)
    yd = np.polyval(pint, x)
    plt.plot(x,y, label='ini')
    plt.plot(x,yd, label='integr')
    plt.grid(); plt.legend();  plt.show()

from pprint import pp
from icecream import ic
def Poly06_oper():
    p1 = np.poly1d([1, 2])
    p2 = np.poly1d([9, 5, 4])
    padd=np.polyadd(p1, p2)  # сложение
    padd2=np.polyadd(padd, -p1) # вычитание
    pmul = np.polymul(p1,p2) # умножение
    ic(p1); ic(p2); ic(padd); ic(padd2); ic(pmul)
    pdiv = np.polydiv(pmul,p1) # деление
    ic(pdiv) # деление - основной результат и остаток от деления
    # остаток - значение числителя, знаменатель p1

from sympy import poly, pdiv, simplify
from sympy.abc import x
def SymPy_Poly():
    top=poly(1.5*x+1.75)
    bot=poly(2*x+1)
    expr=simplify(top*bot)
    print(expr)
    ###########
    top2 = poly(3 * x ** 2 + 5 * x + 1.75)
    top3 = poly(3 * x ** 2 + 5 * x + 2)
    expr2 = pdiv(top2, bot)
    print(expr2)
    expr3 = pdiv(top3, bot)
    print(expr3)

def Leg_appr():
    nn=100
    x = [-50, 0,  5 , 15, 25, 75 , 100]
    coeff = [0.1, 0.1, 0.3, 0.5]
    y = np.polyval(np.poly1d(coeff), x)
    print(np.poly1d(coeff))

    new_x = np.linspace(np.min(x), np.max(x),100)
    y2 = np.polyval(np.poly1d(coeff), new_x)
    plt.plot(x, y, 'o', label='ini points')
    plt.plot(new_x, y2, '-', label='ini line')
    plt.xlabel('x');  plt.ylabel('y'); plt.grid()
    plt.title('Аппроксимация полиномами Лежандра')

    nleg=[2,3,4]
    lst=['dotted','dashed','dashdot']
    lw=[1,2,3]
    for i in reversed(range(len(nleg))):
        legendre_coefs = np.polynomial.legendre.legfit(x, y, nleg[i])
        new_y = np.polynomial.legendre.legval(new_x, legendre_coefs)
        plt.plot(new_x, new_y, linestyle=lst[i], linewidth=lw[i], label='appr '+str(nleg[i]))
    plt.legend()
    plt.show()

if __name__=="__main__":
    # Poly02_def()
    # Poly03_calc()
    # Poly04_diff()
    # Poly05_intgr()
    # SymPy_Poly()
    # Poly06_oper()
    Leg_appr()
