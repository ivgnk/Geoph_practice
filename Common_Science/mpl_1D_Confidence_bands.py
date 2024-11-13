"""
Confidence bands
Полосы доверия
https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-demo-py

fill_between использует цвета цветового цикла в качестве цвета заливки.
Они могут быть немного резкими при применении к областям заливки.
Поэтому часто бывает полезно осветлить цвет,
сделав область полупрозрачной с помощью альфа.
"""

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

def Confidence_bands():
    x = np.linspace(0, 10, 11)
    y = np.array([3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5])
    xmin=int(floor(x.min())); xmax=int(ceil(x.max()))
    ymin=int(floor(y.min())); ymax=int(ceil(y.max()))
    # Подгонка методом наименьших квадратов
    plt.figure(figsize=(14, 8))
    plt.suptitle('Подгонка методом наименьших квадратов')
    n=[1,2,3,4]
    for i in n:
        plt.subplot(1, len(n), i)
        if i==1:
            a, b = np.polyfit(x, y, deg=i)
            y_est = a * x + b
        elif i==2:
            a, b, c = np.polyfit(x, y, deg=i)
            y_est = a * x**2 + b*x + c
        elif i==3:
            a, b, c, d = np.polyfit(x, y, deg=i)
            y_est = a * x**3 + b * x**2 + c*x+d
        else:
            a, b, c, d, ee = np.polyfit(x, y, deg=i)
            y_est = a * x**4 + b * x**3 + c*x**2 + d*x + ee
        y_err = x.std() * np.sqrt(1 / len(x) +
                                  (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
        plt.plot(x, y_est, '-')
        plt.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
        plt.title('Степень '+str(i))
        plt.plot(x, y, 'o', color='brown')
        plt.grid();  plt.xlim((xmin,xmax));   plt.ylim((ymin,ymax))
    plt.show()

if __name__ == "__main__":
    Confidence_bands()