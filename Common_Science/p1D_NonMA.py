"""
2024
General geophysical (and general scientific) tasks in Python
Service for p1D_02.py
Non moving average filters
"""
import inspect # www.geeksforgeeks.org/python-how-to-get-function-name/
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

from scipy.signal import butter, bessel, sosfilt
def butter_filter():
    n = 3
    res=[[0] * n for i in range(n)]
    x, y = p1D_02.make_and_view_data()
    Wn_ = [10,20,30]; btype_= 'low' # 'lowpass'
    fs_ = [1000,2000,3000]
    plt.title(f'Фильтр Баттерворта. В сигнале {len(y)} точка')
    for i in range(n):
        for j in range(n):
            w=Wn_[i]; f=fs_[j]
            sos = butter(N=15, Wn=w, btype=btype_, fs=f, output='sos')  # butter, bessel
            res[i][j] = sosfilt(sos, y)
            plt.plot(x, res[i][j], label='окно ' + str(w) +' частота '+str(f))
    plt.legend();   plt.grid(); plt.show()


if __name__=='__main__':
    # view_savgol()
    # wiener_filter()
    butter_filter()


