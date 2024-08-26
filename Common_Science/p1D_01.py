"""
2024
General geophysical (and general scientific) tasks in Python

Calculation of one-dimensional functions (1D), plotting.
Adding noise
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from icecream import ic

def task_01_easy_1Dfunc_and_grap():
    x = np.linspace(-3, 3, 100)
    plt.plot(x, np.sin(x))
    plt.grid()
    plt.show()

def task_02_random_number_generation():
    n = 2000
    # генерация имени = порядковый номер
    name = [i for i in range(n)]
    # генерация случайного числа
    dat = [random.random() for i in range(n)] # от 0 до 1
    plt.bar(name, dat)
    plt.show()

def task_02_random_rock_density():
    n = 2000
    name = [i for i in range(n)]
    # генерация плотностей горных пород
    dat = [random.uniform(2.1, 3.0) for i in range(n)] # от 0 до 1
    plt.bar(name, dat)
    plt.show()

def task_02_random_gauss():
    n = 2000
    x = [i for i in range(n)]
    yy = [2, 1, 0.5]
    dat = []
    for i in range(len(yy)):
        sps = [random.gauss(0, yy[i]) for j in range(n)]
        dat.append(sps)
        plt.plot(x, dat[i],  label ='s=' + str(yy[i]))
    plt.legend(loc='upper right'); plt.grid(); plt.show()

def task_03_easy_1Dfunc_with_noise_and_grap():
    x = np.arange(-10.0, 10.1, 0.2)
    y = -x + x ** 2 - x ** 3
    random.seed(123)
    # т.к. хотим чтобы случайные
    # последовательности всегда были одинаковы
    # т.к. умножаем каждый элемент списка на число
    rnd = np.array([random.uniform(-1,1) for i in range(len(x))])

    noise1 = 0.75 * y * rnd;  y1 = y + noise1
    noise2 = 0.75 * np.max(y) * rnd; y2 = y + noise2

    plt.plot(x, y,  label='Исход.')
    plt.plot(x, y1, label='+шум1')
    plt.plot(x, y2, label='+шум2')
    plt.legend(loc='upper right'); plt.grid(); plt.show()

def home_tasks():
    n=26
    xmin = np.linspace(-3.5, -27.5, n)
    xmax = np.linspace(23.5,-0.5, n)
    nn=np.linspace(110,350, n, dtype=np.integer)
    fun = ["sin(x) + 0.5*x",    # 1
           "sin(x) + x",        # 2
           "sin(x) + 1.5*x",    # 3

           "sin(2*x) + 0.5*x",  # 4
           "sin(2*x) + x",      # 5
           "sin(2*x) + 1.5*x",  # 6

           "x*sin(2*x) + x * x",# 7
           "x*sin(2*x) + x * x", # 8
           "x*sin(2*x) + x * x", # 9

           "cos(x) + 0.5*x",    # 10
           "cos(x) + x",        # 11
           "cos(x) + 1.5*x",    # 12

           "cos(2*x) + 0.5*x",  # 13
           "cos(2*x) + x",      # 14
           "cos(2*x) + 1.5*x",  # 15

           "x*cos(2*x) + x * x",  # 16
           "x*cos(2*x) + x * x",  # 17
           "x*cos(2*x) + x * x",  # 18

            "sin(x) + cos(2*x)",  # 19
            "sin(x) + cos(2*x)",  # 20
            "sin(x) + cos(2*x)",  # 21

            "-13*x - 0.01 * x**2 + 0.02 * x**3",  # 22
            "-13*x - 0.02 * x**2 + 0.03 * x**3",  # 23
            "-13*x - 0.03 * x**2 + 0.04 * x**3",  # 24
            "-13*x - 0.04 * x**2 + 0.05 * x**3",  # 25

           "x+x*sin(x) + x*cos(2*x)",  # 26
            ]
    for i in range(24,len(fun)):
        plt.figure(i)
        plt.title('N = '+str(i)+', y = '+fun[i])
        x = np.linspace(xmin[i], xmax[i], nn[i])
        rnd = np.array([random.uniform(-1, 1) for i in range(len(x))])
        ys = ne.evaluate(fun[i])
        plt.plot(x, ys, label='Ini')
        noise2 = 0.75 * np.max(ys) * rnd
        y2 = ys + noise2
        plt.plot(x,y2, label='Noise2')
        plt.legend()
        plt.grid()
    plt.show()


import numexpr as ne
def tst_():
    # https://stackoverflow.com/questions/25826500/python-eval-function-with-numpy-arrays-via-string-input-with-dictionaries
    n=4200
    zz=50
    x = np.linspace(-3.5, 23.5, 110)
    ys = ne.evaluate("sin(x)+x")
    ic(ys)
    plt.plot(x, ys)
    plt.grid(); plt.show()

if __name__=="__main__":
    # task_01_easy_1Dfunc_and_grap()
    # task_02_random_number_generation()
    # task_02_random_rock_density()
    task_02_random_gauss()
    # task_03_easy_1Dfunc_with_noise_and_grap()
    # home_tasks()
    # tst_()
