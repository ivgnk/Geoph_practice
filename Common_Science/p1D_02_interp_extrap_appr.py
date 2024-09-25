"""
2024
General geophysical (and general scientific) tasks in Python
Service for p1D_02.py
Interpolation, extrapolation, approximation
"""

import numpy as np
import matplotlib.pyplot as plt
import p1D_02

def intep1D():
    x, y, y1, y2 = p1D_02.make_and_view_data_all(False)
    # - вариант p1D_02.make_and_view_data, выводящий все данные
    # return x, y, y1, y2 # x, y, y+ меньший шум, y+больший шум
    print(len(x)) # len(x) = 101
    print(x) # x = np.arange(-10.0, 10.1, 0.2)
    # проредим данные для интерполяции, возьмем каждую 5 точку
    xp = [x[i] for i,s in enumerate(x) if i%10==0]
    yp = [y[i] for i,s in enumerate(y) if i%10==0]
    # Посмотреть разницу с исходными массивами
    plt.plot(x,y,'bx', label='Исход.')
    plt.plot(xp,yp, 'ro',  label='Прореж.') # круглые красные точки
    plt.legend(loc='upper right'); plt.grid(); plt.show()
    #------------ Непосредственно интерполяция
    yint = np.interp(x, xp, yp)  # Расчет в точках интерполяции
    plt.plot(x,y,  label='Исход.') # голубая линия
    plt.plot(xp,yp, 'ro',  label='Прореж.') # круглые красные точки
    plt.plot(x, yint,'--.y', label='Интерп.')
    plt.legend(loc='upper right'); plt.grid(); plt.show()

def extrap1D():
    # - вариант p1D_02.make_and_view_data, выводящий все данные
    # return x, y, y1, y2 # x, y, y+ меньший шум, y+больший шум
    x, y, y1, y2 = p1D_02.make_and_view_data_all(False)

    # Разобьем массив на несколько частей
    # stackoverflow.com/questions/14406567/partition-array-into-n-chunks-with-numpy
    xpl4=np.array_split(x, 2) # число частей = 2
    ypl4=np.array_split(y, 2)

    # Непосредственно экстраполяция
    # делаем экстраполяцию на основе первой части
    # stackoverflow.com/questions/19406049/extrapolating-data-with-numpy-python
    # numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    plt.title('Экстраполяция на основе полиномиальной аппроксимации')
    plt.plot(x, y, 'y',linewidth=4.5, label='Исход.', )  # голубая линия
    col=['-r','--g','-.k']
    for i in range(3): # от 0 до 2 включительно
        por=i+1 # порядок полинома
        z = np.polyfit(xpl4[0], ypl4[0], por) # получаем коэффициенты полинома
        P = np.poly1d(z) # создаем функцию P для вычисления c новыми данными
        zy = P(x)
        plt.plot(x, zy,col[i], label='Экстрап. '+str(por)+' пор')
    plt.legend(loc='upper center'); plt.grid(); plt.show()

if __name__=='__main__':
    # intep1D()
    extrap1D()