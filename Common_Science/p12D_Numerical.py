"""
Переходное от 1D к 2D

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import inspect # www.geeksforgeeks.org/python-how-to-get-function-name/
from icecream import ic

def make_and_view_data(is_view=False):
    """
    p1D_01.py - > task_03_easy_1Dfunc_with_noise_and_grap()
    :return:
    """
    x = np.arange(-10.0, 10.1, 0.2)
    y = -x + x ** 2 - x ** 3
    random.seed(123)
    # т.к. хотим чтобы случайные
    # последовательности всегда были одинаковы
    # т.к. умножаем каждый элемент списка на число
    rnd = np.array([random.uniform(-1,1) for i in range(len(x))])

    noise1 = 0.75 * y * rnd;  y1 = y + noise1
    noise2 = 0.75 * np.max(y) * rnd; y2 = y + noise2
    if is_view:
        plt.plot(x, y,  label='Исход.')
        plt.plot(x, y1, label='+шум1')  # меньший шум
        plt.plot(x, y2, label='+шум2')  # больший шум
        plt.legend(loc='upper right'); plt.grid(); plt.show()
    return x, y2 # больший шум

# from math import sqrt
import numpy as np
def vec_length_1():
    # Найти длину вектора 1D
    y=[1,2,3,4]      # задаем список
    ynp=np.array(y)  # из списка создаем массив numpy
    # Первый вариант расчета длины
    len_1=np.linalg.norm(y)
    print('Первый вариант расчета длины = ',len_1)
    # Второй вариант расчета длины
    len_2 =  np.sqrt(sum(ynp**2))
    print('Второй вариант расчета длины = ',len_2)
    print(f'Второй вариант расчета длины, {len_2:5.2f} форматирование')

import inspect # www.geeksforgeeks.org/python-how-to-get-function-name/
def the_len(x):
    return np.sqrt(sum(x**2))

def vec_length_2():
    print('Работает функция ',inspect.currentframe().f_code.co_name)
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4, 6, 8])
    b_min_a=b-a
    print(f'Для векторов {a} и {b}')
    print(f'Разность = {b_min_a}')
    print(f'Длина вектора разности =',the_len(b_min_a))

from math import sqrt
def skal_proizv():
    #--1 Скалярное прозведение
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4, 6, 6])
    c=a*b
    print('Поэлементное произведение a b = ', c)
    print('Сумма поэлементных произведений (скалярное произведение) =',np.sum(c))
    print('np.dot(a,b) /скалярное произведение/ =  ', np.dot(a,b))

def cos_ugla():
    #--2 Косинус угла между векторами
    # i,j,k - единичные векторы по осям x, y, z
    # a = i+2j+3k
    # b = 6+4j-2k
    a = np.array([1, 2, 3])
    b = np.array([6, 4, -2])
    an1 = np.linalg.norm(a)
    an2 = the_len(a)
    print(f'Длина вектор a = {an1:6.4f}, {an2:6.4f}')
    bn1 = np.linalg.norm(b)
    print(f'Длина вектор b = {bn1:6.4f} {np.sqrt(56):6.4f}' )
    print(f'Косинус угла между a и b = {np.dot(a,b)/(an1*bn1):6.4f} {2/7:6.4f}')

def umn_matr_1():
    A = np.array([[-7, 4, 0], [0, -1, 0], [-1, 5, 7]])
    B = np.array([[1, 0], [0, -2], [1, 1] ])
    F1 = A @ B
    F2 = np.dot(A, B)
    print(F1)
    print(F2)

def calc_det():
    # вычисление определителя матрицы
    a = np.array([[7,-3],
                  [1, 1]])
    det_a =np.linalg.det(a)
    print(f'{det_a} {det_a:8.4f}')
# x, y2 = make_and_view_data(is_view=bool(1))

def calc_inv_matr():
    a = np.array([[7,-3],
                  [1, 1]])
    ainv=np.linalg.inv(a)
    print(ainv)

def calc_matr_minor():
    # https://stackoverflow.com/questions/3858213/numpy-routine-for-computing-matrix-minors
    a = np.array([[1, 2, 3, 4],
                  [11, 22, 33, 44],
                  [111, 222, 333, 444],
                  [1111, 2222, 3333, 4444]])

def calc_inv_matr2():
    a = np.array([[7,-3], [1, 1]])
    ainv=np.linalg.inv(a);  ed_matr=np.eye(2)
    m1=np.dot(a, ainv)
    print(f'{m1=}'); x = np.allclose(m1, ed_matr)
    print('Близость матриц np.dot(a, ainv) и np.eye(2) = ', x)
    m2=np.dot(ainv, a)
    print(f'{m2=}'); x = np.allclose(m2, ed_matr)
    print('Близость матриц np.dot(ainv, a) и np.eye(2) = ', x)

def sys_lin_eq_sol():
    # x + 2y +  z = 7
    # 2x -  y + 3z = 7
    # 3x +  y + 2z =18
    a = np.array([[1, 2, 1], [2, -1, 3], [3, 1, 2]])
    b = np.array([7, 7, 18])
    x1= np.linalg.solve(a, b)
    print('Решение СЛАУ solve(a, b) ',x1)  # [ 7.  1. -2.]

    x2 = np.linalg.inv(a).dot(b)
    print('Решение СЛАУ inv(a).dot(b) ', x2)  # [ 7.  1. -2.]
    y1 = np.allclose(np.dot(a, x1), b)
    y2 = np.allclose(np.dot(a, x2), b)
    print(f'Решения 1 и 2 правильные? {y1=} {y2=}')

import pandas as pd
def gen_matrices():
    """
    Генератор случайных векторов и матриц
    """
    n=51
    for i in range(n):
        np.random.seed(i) # задаем генерацию сл.чис
        x = np.random.randint(n, size=3) # вектор сл.чис
        a = np.random.randint(n, size=(3,3)) # матрица сл.чис
        # print(x); print(a)
        b = np.dot(a,x)
        # print(b)
        print(f' {i=}  {b=}')
        print(f'{a=}')
    # print(np.linalg.solve()

if __name__=="__main__":
    # vec_length_1()
    # vec_length_2()
    # skal_proizv()
    # cos_ugla()
    # umn_matr_1()
    # calc_det()
    # calc_inv_matr()
    # sys_lin_eq_sol()
    gen_matrices()