"""
2024
General geophysical (and general scientific) tasks in Python
Service for p1D_02.py
Triangle moving average filters
"""
import inspect # www.geeksforgeeks.org/python-how-to-get-function-name/
from icecream import ic

import numpy as np
import matplotlib.pyplot as plt

import p1D_02

def calc_triang_weights_for_1Dfilter(win_size:int)->np.ndarray:
    '''
    Расчет значений треугольной функции для фильтра
    не проверяем win_size, оно должно быть нечетным
    '''
    x = np.array([0.0 for i in range(win_size)])
    half_win = win_size // 2
    frst_odd = 1
    for i in range(x.size):
        if i <= half_win:  # левая половина массива, включая центр
            x[i] = frst_odd
            frst_odd += 2
        else:
            x[i] = x[half_win-(i-half_win)]  # правая половина массива, без центра
    return x

def norm_weights_for_1Dfilter(dat: np.ndarray) -> np.ndarray:
    '''
    Нормализация весов для 1D фильтра
    '''
    return dat/np.sum(dat)

def view_triangle():
    ic(inspect.currentframe().f_code.co_name)
    llen=5
    tr_win=calc_triang_weights_for_1Dfilter(llen)
    # Нормируем на 1
    tr_win_w=tr_win/np.sum(tr_win)
    # Делаем х-координаты
    x=np.linspace(0,llen-1,llen)
    #---- Графики результатов
    name=f'Треугольное окно {llen} точек'
    plt.plot(x,tr_win,label=name)
    plt.plot(x,tr_win_w,label=name+' нормированное')
    plt.legend() # loc='upper right'
    plt.grid(); plt.show()



def compare_weighted_and_nonweighted_filters():
    """
    Сравнение фильтра с весами (треугольные) и без весов
    """
    win_size=13
    x, y = p1D_02.make_and_view_data()
    tr_win=calc_triang_weights_for_1Dfilter(win_size)
    tr_win_w=tr_win/np.sum(tr_win)

    tr1=np.convolve(y, tr_win, 'same')
    tr2=np.convolve(y, tr_win_w, 'same')

    nontriang, _ = p1D_02.MA_equal_weight_with_res_anom(y, win_size)

    #---- Графики результатов
    plt.title(f'Сглаживание окном из {win_size} точек')
    plt.plot(x,y,label='исходный')
    # plt.plot(x,tr1,label='треугольное без весов')
    plt.plot(x,tr2,label='треугольное с весами')
    plt.plot(x,nontriang,label='равномерное')
    plt.legend() # loc='upper right'
    plt.grid(); plt.show()



if __name__=="__main__":
    # view_triangle()
    compare_weighted_and_nonweighted_filters()