"""
https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
https://www.geeksforgeeks.org/numpy-linear-algebra/
"""
import matplotlib.pyplot as plt
import numpy as np
def lst_sq01():
    # https://www.geeksforgeeks.org/numpy-linear-algebra/
    x = np.arange(0, 9)
    np1=np.ones(9) # вектор из 9 единиц
    a = np.array([x, np1])
    y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
    print(a)
    w = np.linalg.lstsq(a.T, y)  # a.T - транспонирование
    k = w[0][0]  # - угловой коэффициент
    b = w[0][1]  # - пересечение с осью Y
    line = k * x + b  # regression line
    plt.plot(x, line, 'r-',label='линия по МНК')
    plt.plot(x, y, 'o',label='исх.данные')
    plt.text(0,23.5,f' y = {k:6.3f}x+{b:6.3f}')
    plt.legend(); plt.title('Использование функции np.linalg.lstsq')
    plt.grid();   plt.show()
    print(w)

def lst_sq02():
    A = np.array([[1, 2],
                  [-1, 1],
                  [0, 3]])
    B = np.array([1, 3, 0])
    print("The coefficient matrix is =", A)
    print("the coordinate matrix is=", B)
    R, residuals, RANK, sing = np.linalg.lstsq(A, B, rcond=None)
    print("the least square solutions are'", R)

def calc_step_f(x):
    return x
def lst_sq03():
    # Часть 1
    stepen = 2; stall=stepen+1
    x = np.array([0, 1, 2, 3])
    lenx=len(x)
    y = np.array([-1.2, -0.1, 1.3, 1.8])
    A = x[:, np.newaxis] ** [0, 1, 2]
    # Задать матрицу А по другому
    a1 = np.zeros(lenx*stall, dtype=np.int64).reshape(lenx,stall)
    for i in range(stall): a1[:,i]=x**i
    print(a1)
    print(A)
    # Часть 2
    res = np.linalg.lstsq(A, y, rcond=None)
    res2 = np.linalg.lstsq(a1, y, rcond=None)
    print("Значения коэффициентов")
    for i in range(3):
        print(f"с({i})={res[0][i]:11.7f}    {res2[0][i]:11.7f}")
    a, b, c = res[0][0], res[0][1], res[0][2]
    x1 = np.linspace(min(x), max(x), 100)
    ym = a + b * x1 + c * x1 ** 2
    plt.text(0.6,-0.9,f"Уравнение кривой\nYm={a:.6f}+({b:.6f})*x+({c:.6f})*x^2",
             color='r', fontweight='bold', fontsize=10)
    plt.plot(x, y, 'bo', label="Исходные\nданные")
    plt.plot(x1, ym, label="Аппроксимирующая\nкривая", c='g')
    plt.legend(); plt.grid(); plt.title('np.linalg.lstsq: Аппроксимация квадратичной зависимости')
    plt.show()


lst_sq03()