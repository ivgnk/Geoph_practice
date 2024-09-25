"""
2024
General geophysical (and general scientific) tasks in Python
Service for p1D_02.py
Non moving average filters
"""
import inspect # www.geeksforgeeks.org/python-how-to-get-function-name/
from cProfile import label

from icecream import ic

from math import sqrt
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

import p1D_02

SMA_w = [3, 7, 13]  # размер окон , 25, 51
lt=['--','-.', ':',]

# Определяем желаемую степень аппроксимирующего полинома
def get_poly_order_for_savgol(window_length: int) -> int:
    if window_length > 30:
        res = 6
    elif window_length > 20:
        res = 5
    elif window_length > 10:
        res = 4
    elif window_length > 3:
        res = 3
    else:
        res = 2
    return res

def calc_savgol():
    '''
    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-for-a-dataset
    '''
    res=[]
    x, y = p1D_02.make_and_view_data()
    len_y= len(y) # число отсчетов данных
    npvar = len(SMA_w) # число вариантов фильтра
    for i in range(npvar):
        win_len = SMA_w[i]
        polyorder_ = get_poly_order_for_savgol(win_len)
        if win_len >= len_y:
            print('Ошибка: вариант = ',i,'длина окна = ',win_len,'длина набора данных = ',len_y)
        else:
            res.append(sc.signal.savgol_filter(y, window_length=win_len, polyorder=polyorder_))
    return x, y, res

def view_savgol():
    lw=[1,2,3] # толщина линий для разных фильтров
    plt.figure(figsize=(8, 5))
    x, y, res = calc_savgol()
    plt.title(f'Фильтр Савицкого–Голея. В сигнале {len(y)} точка')
    plt.plot(x,y,label='исходный', linewidth=4)

    for i in range(len(SMA_w)):
        plt.plot(x,res[i],label='окно '+str(SMA_w[i]), linewidth=lw[i])

    plt.legend();   plt.grid(); plt.show()

def wiener_filter():
    res=[]
    x, y = p1D_02.make_and_view_data()
    npvar = len(SMA_w) # число вариантов фильтра
    for i in range(npvar):
        res.append(sc.signal.wiener(y, SMA_w[i]))
    # Визуализация
    lw=[2,2,3] # толщина линий для разных фильтров
    plt.figure(figsize=(12, 8))
    plt.title(f'Фильтр Винера. В сигнале {len(y)} точка')
    plt.plot(x,y,label='исходный', linewidth=4)
    for i in range(len(SMA_w)):
        plt.plot(x,res[i],label='окно '+str(SMA_w[i]), linewidth=lw[i])
    plt.legend();   plt.grid(); plt.show()


import random
def make_data_for_Butter(is_view:bool=False):
    """
    p1D_01.py - > task_03_easy_1Dfunc_with_noise_and_grap()
    :return:
    """
    # x = np.arange(-10.0, 10.1, 0.2)
    # y = -x + x ** 2 - x ** 3
    x = np.arange(0, 20.1, 0.1)
    t = np.linspace(0, 20.1, 201, False)
    y = -(x-10) + (x-10)**2 - (x-10)** 3
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


from scipy.signal import butter, bessel, sosfilt, freqs
def butter_filter():
    x,y = make_data_for_Butter(False) # сдвинули вигнал так, чтобы он начинался с нуля
    fs_ = [10] # частота дискретизации
    nfs=len(fs_) # сколько вариантов fs рассматриваем
    # 0 < Wn < fs/2 (fs=20 -> fs/2=10.0)
    Wn_ = [4,2,1] # критические частоты (частоты среза) ФНЧ
    nwn = len(Wn_) # сколько вариантов Wn рассматриваем
    btype_= 'low' # 'lowpass'
    res=[[0] * nfs for i in range(nwn)]
    plt.figure(figsize=(12, 8))
    plt.title(f'Фильтр Баттерворта. В сигнале {len(y)} точка')
    plt.plot(x,y,label='ini',linewidth=2)
    for i in range(nwn):
        for j in range(nfs):
            w=Wn_[i]; f=fs_[j]
            sos = butter(N=1, Wn=w, btype=btype_, fs=f, output='sos')  # butter, bessel
            res[i][j] = sosfilt(sos, y)
            plt.plot(x, res[i][j], label='критическая частота =' + str(w))
    plt.xlabel('Время, сек'); plt.ylabel('Амплитуда, у.е.')
    plt.legend();   plt.grid(); plt.show()

def butter_filter_freq():
    # вывод частоной характеристики фильтра
    fs_ = [10] # частота дискретизации
    nfs=len(fs_) # сколько вариантов fs рассматриваем
    # 0 < Wn < fs/2 (fs=20 -> fs/2=10.0)
    Wn_ = [4,2,1] # критические частоты (частоты среза) ФНЧ
    nwn = len(Wn_) # сколько вариантов Wn рассматриваем
    btype_= 'low' # 'lowpass'
    col=['blue','red','green']
    plt.figure(figsize=(12, 8))
    plt.title(f'Фильтр Баттерворта. Амплитудно-частотные характеристики')
    for i in range(nwn):
        print(i, Wn_[i])
        w1=Wn_[i]
        b, a = butter(N=1, Wn=w1, btype=btype_, analog=True)  # butter, bessel
        w, h = freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)), color=col[i]) # ,label=str(w)+' Hz'
        plt.axvline(w1, color=col[i], label='cutoff frequency ' + str(w1),linestyle='dashed')  # cutoff frequency
        plt.grid(which='both', axis='both')
    # plt.xlabel('Frequency [radians / second], 1 рад/с = 0,159155 Гц ')
    plt.xlabel('Frequency [Гц]')
    plt.ylabel('Amplitude [dB]')
    plt.legend(); plt.show()
def cmp_arange_linspace():
    '''
    Comaprae functions np.arange, np.linspace
    :return:
    '''
    # np.arange
    # x = np.arange(-10.0, 10.1, 0.1)  # хвосты после нуля
    x = np.arange(0.0, 20.1, 0.1)    # без хвостов после нуля
    print('x')
    print(x.min(), x.max(), len(x), x[1]-x[0])
    n=5 # round(number, ndigits=None) - ndigits precision after the decimal point
    print(round(x.min(),n), round(x.max(),n), len(x), round(x[1]-x[0],n))

    # np.linspace
    print('t')
    t = np.linspace(0, 20.1, 201, False)
    print(t.min(), t.max(), len(x), t[1]-t[0])
    print('tnorm')
    t = np.linspace(0, 1, 10, False)
    print(t.min(), t.max(), len(x), t[1]-t[0])


from scipy.integrate import quad
def integrextrap1D():
    # - вариант p1D_02.make_and_view_data, выводящий все данные
    # return x, y, y1, y2 # x, y, y+ меньший шум, y+больший шум
    x, y, y1, y2 = p1D_02.make_and_view_data_all(False)

if __name__=='__main__':
    # view_savgol()
    # wiener_filter()
    # butter_filter()
    # butter_filter_freq()
    cmp_arange_linspace()
    # t = np.linspace(0, 1, 1000, False)
