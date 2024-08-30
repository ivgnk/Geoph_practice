"""
2024
General geophysical (and general scientific) tasks in Python

Noise removal (smoothing).
Interpolation and extrapolation.
Integration, differentiation of 1D
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import random
import inspect # www.geeksforgeeks.org/python-how-to-get-function-name/
from icecream import ic

# -- 00 -- Создание массивов и визуализация
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

# -- 01 -- Базовая функция для равновесового сглаживания
def MA_equal_weight(dat, win_size):
    """
    Просто сглаживание
    from D:\Work_Lang\Python\PyCharm\GTimeSeries\psd_filters.py
    """
    window= np.ones(win_size)/win_size
    return np.convolve(dat, window, 'same')

# -- 02 -- одно окно сглаживания и одна остаточная аномалия после
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


# -- 03 -- Несколько окон сглаживания и несколько остаточных аномалий, визуализация остаточных аномалий
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

# -- 04 -- Несколько окон и несколько остаточных аномалий, визуализация сглаженного и остаточных аномалий
def Many_MA_equal_weight_and_view():
    x, y = make_and_view_data() #
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

# -- 05 - base calculating for weighting
# from stackoverflow.com/questions/46230222/smoothing-a-series-of-weighted-values-in-numpy-pandas
def rolling_weighted_triangle_conv(x, w, window_size, isview=False):
    ic(inspect.currentframe().f_code.co_name)
    """Smooth with triangle window, also using per-element weights."""
    # Simplify slicing
    wing = window_size // 2

    # Pad both arrays with mirror-image values at edges
    xp = np.concatenate(( x[wing-1::-1], x, x[:-wing-1:-1] ))
    wp = np.concatenate(( w[wing-1::-1], w, w[:-wing-1:-1] ))

    if isview:
        ic("xp"); ic(xp)

    # Generate a (triangular) window of weights to slide
    incr = 1. / (wing + 1)
    ramp = np.arange(incr, 1, incr)
    triangle = np.r_[ramp, 1.0, ramp[::-1]]

    D = np.convolve(wp*xp, triangle)[window_size-1:-window_size+1]
    N = np.convolve(wp, triangle)[window_size-1:-window_size+1]
    res = D/N
    ic(window_size, wing, len(x),len(xp), len(res))
    return res

def make_wing_const(x:np.ndarray, wing_len:int):
    ones=np.ones(wing_len)
    left = x[0]*ones
    right = x[-1]*ones
    return np.concatenate(( left, x, right )), ' const'

def make_wing_periodical(x:np.ndarray, wing_len:int):
    return np.concatenate((x[wing_len - 1::-1], x, x[:-wing_len - 1:-1])),  ' periodical'


# -- 05-2 - base calculating for triangular weighting
def rolling_weighted_triangle_conv_2(x, window_size, edge_type):
    ic(inspect.currentframe().f_code.co_name)
    wing = window_size // 2
    if edge_type:
        xp, name = make_wing_periodical(x,wing)
    else:
        xp, name = make_wing_const(x,wing)

        # Generate a (triangular) window of weights to slide
    incr = 1. / (wing + 1)
    ramp = np.arange(incr, 1, incr)
    triangle = np.r_[ramp, 1.0, ramp[::-1]]
    ic(triangle)
    res = np.convolve(xp, triangle)[window_size-1:-window_size+1]
    return res, name

def rolling_weighted_triangle_conv_3(x, window_size, edge_type):
    ic(inspect.currentframe().f_code.co_name)
    wing = window_size // 2
    if edge_type:
        xp, name = make_wing_periodical(x,wing)
    else:
        xp, name = make_wing_const(x,wing)

    wp=np.ones(len(xp))
    # Generate a (triangular) window of weights to slide
    incr = 1. / (wing + 1)
    ramp = np.arange(incr, 1, incr)
    triangle = np.r_[ramp, 1.0, ramp[::-1]]
    ic(triangle)
    D = np.convolve(wp*xp, triangle)[window_size-1:-window_size+1]
    N = np.convolve(wp, triangle)[window_size-1:-window_size+1]
    res = D/N
    return res, name


def random_nparr(llen:int):
    rng = np.random.default_rng()
    return rng.standard_normal(llen)

# --- 05.1 - test function 05 & 05-1
def the_test_fun05_1(isview=False):
    ic(inspect.currentframe().f_code.co_name)
    n = 101
    # x = np.linspace(0,100,n)
    x = random_nparr(n)
    if isview: ic(x)
    w = np.ones(n)
    plt.plot(x,label='ini')
    res = rolling_weighted_triangle_conv(x, w, 19 )
    plt.plot(res,label='res')
    plt.legend(loc='upper right')
    plt.grid(); plt.show()

# --- 05.1 - test function 01 with different edges
def the_test_fun05_2():
    ic(inspect.currentframe().f_code.co_name, ' = ',"test function 01 with different edges")
    x, y = make_and_view_data()
    x = random_nparr(len(x))
    win = [5,7,11]
    lt=['-','-.','--']
    plt.figure(figsize=(18,10))
    plt.plot(x,label='ini')
    for i,win1 in enumerate(win):
        for ed_tp in [True, False]:
            res, name = rolling_weighted_triangle_conv_3(x, win[i], ed_tp)
            plt.plot(res,label=name+str(win1),linestyle=lt[i])
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


#  --06 - My Shell for 05
def work_with_05fun():
    fun_name=inspect.currentframe().f_code.co_name
    x, y = make_and_view_data()  #
    y = random_nparr(len(x))
    win = [3,7,9,15]
    z=[]
    plt.figure(figsize=(12,8))
    for win_size in win:
        w=np.ones(len(x))
        z.append(rolling_weighted_triangle_conv(y, w, win_size))
    ic(len(x),[len(z1) for z1 in z])
    view(x, [y]+z, ['ini']+[str(i) for i in win],fun_name)

#  --07- Special function for visualisation
def view(x:np.ndarray,y:list,ylabels:list=(), title:str=''):
    if title: plt.title(title)
    print(len(x),len(y[0]))
    for i in range(len(y)):
        lbl  = ylabels[i] if ylabels != [] else ""
        plt.plot(x,y[i],label=lbl)
    if ylabels: plt.legend(); # loc='upper right'
    plt.grid(); plt.show()


if __name__=="__main__":
    # Many_MA_equal_weight_and_view()
    # Many_MA_equal_weight_and_res_anom_and_view()
    # the_test_fun05_1()
    # the_test_fun05_2()
    work_with_05fun()