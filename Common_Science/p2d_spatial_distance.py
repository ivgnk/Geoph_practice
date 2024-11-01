"""
Functions from
https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
"""
from scipy.spatial import distance
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
# from math import factorial
import scipy.special
from math import *
import inspect
import sys

from scipy.spatial.distance import squareform

def the_pdist():
    """
    Pairwise distances between observations in n-dimensional space
    Попарные расстояния между наблюдениями в n-мерном пространстве
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    """
    print("Function = ",inspect.currentframe().f_code.co_name)  # Вывод имени функции
    # --1-- Calc
    rng = np.random.default_rng(seed=125); nn=5
    pnts = rng.random((nn, 2))
    # print(pnts)
    npairs=int(scipy.special.binom(nn,2)) # Число пар = биномиальные коэффициенты
    # print(npairs)
    names=['euclidean',  'minkowski', 'cityblock', 'sqeuclidean']
    eucl=lambda x1, x2, y1, y2: hypot(x1-x2, y1-y2)
    mink=lambda x1, x2, y1, y2: sqrt(abs(x1-x2)**2+abs(y1-y2)**2)
    city=lambda x1, x2, y1, y2: abs(x2-x1)+abs(y2-y1)
    sqeu=lambda x1, x2, y1, y2: (x1-x2)**2+(y1-y2)**2
    fun=[eucl, mink, city,sqeu]
    for ii,fun1 in enumerate(fun):
        print(ii,names[ii])
        res = distance.pdist(pnts,metric=names[ii])
        curr = 0
        for i in range(nn):
            for j in range(nn):
                if j>i:
                    print(f'{curr:2}  {i} {j}   {fun1(pnts[i][0],pnts[j][0], pnts[i][1], pnts[j][1]):.3f}   {res[curr]:.3f}')
                    curr+=1
    # --2-- View
    plt.title("Различие расстояний")
    for name_ in names:
        res = distance.pdist(pnts,metric=name_)
        plt.plot(res,label=name_)

    plt.legend()
    plt.grid()
    plt.show()

def the_cdist():
    """
    Compute distance between each pair of the two collections of inputs.
    Вычислите расстояние между каждой парой двух наборов входных данных.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """
    print("Function = ",inspect.currentframe().f_code.co_name)  # Вывод имени функции
    # --1-- Calc
    rng = np.random.default_rng(seed=125); nn=5
    pnts = rng.random((nn, 2))
    res = distance.cdist(pnts, pnts, 'euclidean') # для одного и того же вектора
    print(res)


def the_corr():
    x=np.arange(9)
    y=np.array([3,2,54,4,56,6,13,454,32])
    d=np.vstack([x, y]).T
    print('the_corr()')
    Y = distance.pdist(d, 'correlation')
    # print(Y)
    print(np.corrcoef(x,y))



def the_dice():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html
    ic(distance.dice([1, 0, 0], [0, 1, 0])) # 1
    ic(distance.dice([2, 0, 0], [0, 31, 0])) # 1
    ic('-------')
    ic(distance.dice([1, 0, 0], [1, 1, 0])) # 0.3333333333333333
    ic(distance.dice([9, 0, 0], [9, 9, 0]))
    ic(distance.dice([111, 0, 0], [221, 251, 0]))
    # distance.dice([1, 0, 0], [2, 0, 0]) # 0.3333333333333333

def npairs():
    nn = 5
    # ----1
    # Количество пар = (Общее число элементов X Общее число элементов — 1) / 2
    # https://www.sciencedebate2008.com/kak-poschitat-kolichestvo-par-dlya-zadannogo-chisla-elementov/
    # npair=(nn*nn-1)/2
    # Неправильно


    # ----2
    # C(n, k) = n! / (k! * (n - k)!)
    # n - количество элементов в наборе.
    # k - количество элементов в каждой паре (в нашем случае k = 2).
    # https://telegra.ph/Kak-poschitat-kolichestvo-par-chisel-ot-kombinatoriki-do-prakticheskih-primerov-09-25
    n = factorial(nn)
    k = factorial(2)
    n_k= factorial(nn-2)
    ic(n//(k*n_k))

    # ----3
    # https://stackoverflow.com/questions/26560726/python-binomial-coefficient
    ic(int(scipy.special.binom(nn,2)))


def breaks3():
    n=0
    for x in range(10):
        for y in range(10):
            for z in range(10):
                print(n, x, y, z, x * y * z)
                n+=1
                if x * y * z == 30:
                    break  # прерываем внутренний цикл
            else:
                continue  # продолжаем, если внутренний цикл не был прерван
            break  # внутренний цикл был прерван, прерываем и этот цикл
        else:
            continue
        break

def breaks2():
    n=0
    for x in range(10):
        for y in range(10):
                print(n, x, y, x + y)
                n+=1
                if x + y == 15: break  # прерываем внутренний цикл
        else: continue  # продолжаем, если внутренний цикл не был прерван
        break  # внутренний цикл был прерван, прерываем и этот цикл

def the_squareform():
    print("Function = ",inspect.currentframe().f_code.co_name)  # Вывод имени функции
    rng = np.random.default_rng(seed=125); nn=5
    pnts = rng.random((nn, 2))
    vec=distance.pdist(pnts)
    print('vec=\n',vec,'\n')
    mat=squareform(vec)
    print('mat=\n',mat,'\n')
    vec2=squareform(mat)
    print('vec2=\n',vec2,'\n')

if __name__=="__main__":
    # the_dice()
    # the_cdist()
    the_pdist()
    # the_squareform()
    # npairs()
    # breaks2()
    # the_corr()