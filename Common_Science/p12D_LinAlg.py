"""
https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html#numpy.vecdot
https://bemind.gitbook.io/neural/uchebniki/uchebniki-po-pandas-i-numpy/numpy/kak-rasschitat-vektornoe-proizvedenie-v-python
"""
import numpy as np
import matplotlib.pyplot as plt

def vec_proizv():
    a = np.array([2, -3, -1])
    b = np.array([3, -1, -4])
    print(np.cross(a, b))

def vec_proizv2():
    a = np.array([3, 2, 1])
    b = np.array([6, 5, 4])
    print(np.cross(a, b))
    # cross_prod = np.cross(a, b)
    # return cross_prod

def vec_proizv3():
    a=dict(); b=dict(); c = dict()
    a['x'] = 3; a['y'] = 2; a['z'] = 1
    b['x'] = 6; b['y'] = 5; b['z'] = 4

    c['x'] = a['y']*b['z'] - a['z']*b['y']
    c['y'] = a['z']*b['x'] - a['x']*b['z']
    c['z'] = a['x']*b['y'] - a['y']*b['x']
    print(c)

def smesh_proizv():
    """
    https://ru.wikipedia.org/wiki/Произведения_векторов#Смешанное_произведение
    | ax ay az |          |
    | bx by bz |
    | cx cy cz |
    """
    a = np.array([-3, 2,-1])
    b = np.array([ 1, 0, 2])
    c = np.array([ 4, 5,-2])
    ab=np.cross(a, b) # векторное произв.
    print(np.dot(ab,c)) # скалярное произв.

def smesh_proizv2():
    """
    https://ru.wikipedia.org/wiki/Произведения_векторов#Смешанное_произведение
    | ax ay az |          |
    | bx by bz |
    | cx cy cz |
    """
    a = np.array([-3, 2,-1])
    b = np.array([ 1, 0, 2])
    c = np.array([ 4, 5,-2])
    # https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    m = np.stack((a,b,c))  # соединение векторов-строк
    det_a =np.linalg.det(m) # Вычисление определителя матрицы
    print(f'Определитель = {det_a} {det_a:6.3f}')

from icecream import ic
def vec_proizv4():
    a = np.array([3, 2, 1])
    b = np.array([6, 5, 4])
    ic(np.cross(a, b))
    ic(np.dot(a, b))  #  np.int64(32)
    ic(np.linalg.vecdot(a, b)) #  np.int64(32)
    ic(np.inner(a, b)) # np.int64(32)
    ic(np.outer(a, b)) # array([[18, 15, 12],
                       # [12, 10,  8],
                       # [ 6,  5,  4]])
    ic(np.matmul(a, b)) # np.int64(32)
    # ic(np.tensordot(a, b)) # - error
    ic(np.kron(a, b)) # array([18, 15, 12, 12, 10,  8,  6,  5,  4])

def tensorsolve():
    #  https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html#numpy.linalg.tensorsolve
    a = np.eye(2 * 3 * 4)
    a.shape = (2 * 3, 4, 2, 3, 4)
    rng = np.random.default_rng()
    b = rng.normal(size=(2 * 3, 4))
    print(a)
    print(b)
    x = np.linalg.tensorsolve(a, b)
    print(x.shape)
    print(np.allclose(np.tensordot(a, x, axes=3), b))

def SLAU():
    # declaring the arrays array
    X = np.array([[60, 40, 50], [10, 20, 30], [70, 80, 90]])
    Y = np.array([18, 19, 20])

    out1 = np.linalg.tensorsolve(X, Y)
    print('tensorsolve =', out1)
    out2 = np.linalg.solve(X, Y)
    print('solve = ', out2)

def lstsq():
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])

import random
from icecream import ic
def corr_coeff():
    x = np.arange(-10.0, 10.1, 0.2)
    y = -x + x ** 2 - x ** 3
    random.seed(123)
    # т.к. хотим чтобы случайные
    # последовательности всегда были одинаковы
    # т.к. умножаем каждый элемент списка на число
    rnd = np.array([random.uniform(-1,1) for i in range(len(x))])
    noise1 = 0.75 * y * rnd;  y1 = y + noise1
    noise2 = 0.75 * np.max(y) * rnd; y2 = y + noise2
    ic(np.corrcoef(y, y1))
    ic(np.corrcoef(y, y2))
    ic(np.corrcoef(y1, y2))
    # np.corrcoef возвращает корреляционную матрицу переменных.
    # Это двумерный массив с коэффициентами корреляции.
    # Диагональные элементы матрицы всегда равны 1,
    # они представляют корреляцию переменной с самой собой.
    # Остальные элементы матрицы — коэффициенты корреляции каждой пары переменных.


    # plt.plot(x, y,  label='Исход.')
    # plt.plot(x, y1, label='+шум1')
    # plt.plot(x, y2, label='+шум2')
    # plt.legend(loc='upper right'); plt.grid(); plt.show()

def cheb01():
    x = np.linspace(-1.0, 1.0)
    fun = np.polynomial.Chebyshev((6,7,9))
    y = fun(x)
    plt.plot(x,y); plt.grid()
    plt.show()

if __name__=="__main__":
    # print(vec_proizv3()) # [ 3 -6  3]
    # smesh_proizv2()
    # vec_proizv4()
    # tensorsolve()
    # SLAU()
    # lstsq()
    # corr_coeff()
    cheb01()


