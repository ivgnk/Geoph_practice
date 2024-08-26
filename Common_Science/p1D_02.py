"""
2024
General geophysical (and general scientific) tasks in Python

Noise removal (smoothing).
Interpolation and extrapolation.
Integration, differentiation of 1D
"""

import numpy as np
import matplotlib.pyplot as plt
import random
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

def MA_equal_weight(dat, win_size):
    """
    Просто сглаживание
    from D:\Work_Lang\Python\PyCharm\GTimeSeries\psd_filters.py
    """
    window= np.ones(win_size)/win_size
    return np.convolve(dat, window, 'same')

def MA_equal_weight_with_res_anom(dat, win_size):
    """
    Сглаживание с вычислением остаточных аномалий
    from D:\Work_Lang\Python\PyCharm\GTimeSeries\psd_filters.py
    """
    # from D:\Work_Lang\Python\PyCharm\GTimeSeries\psd_filters.py
    window= np.ones(win_size)/win_size
    sgl = np.convolve(dat, window, 'same')
    res_anom = dat - sgl
    return sgl, res_anom


def Many_MA_equal_weight_and_res_anom_and_view():
    x, y = make_and_view_data()
    win=[3,5,7]
    plt.figure(figsize=(8, 5))  # задаенм размер окна, чтобы умещался заголовок
    plt.plot(x, y, label='Шум')
    plt.title('Остаточные аномалии после равновесового сглаживания с '+str(win)+' точек ')
    for i in win:
        y1, ra1 = MA_equal_weight_with_res_anom(y, i)
        plt.plot(x, ra1, label='Ост.аном.'+str(i))
    plt.legend(loc='upper right'); plt.grid(); plt.show()

def Many_MA_equal_weight_and_view():
    x, y = make_and_view_data() # ooood
    win=[3,5,7]
    res=[]
    plt.plot(x, y, label='Шум')
    for i in win:
        y1, ra1 = MA_equal_weight_with_res_anom(y, i)
        res.append((y1,ra1))
        plt.plot(x, y1, label='Сглаж.'+str(i))

    plt.title('Равновесовое сглаживание с '+str(win)+' точек ')
    plt.legend(loc='upper right'); plt.grid(); plt.show()

    plt.title('Остаточные аномалии после равновесового сглаживания с '+str(win)+' точек ')
    plt.plot(x, y, label='Шум')
    for i in range(len(res)):
        plt.plot(x, res[i][1], label='Ост. аном.' + str(win[i]))
    plt.legend(loc='upper right'); plt.grid(); plt.show()


if __name__=="__main__":
    # Many_MA_equal_weight_and_view()
    Many_MA_equal_weight_and_res_anom_and_view()