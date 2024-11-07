"""
https://numpy.org/doc/stable/reference/generated/numpy.cov.html
"""
import sys
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

def cov_01():
    x1 = [1, 1, 1, 1, 1]; cv = np.cov(x1); ic(x1, cv); ic('----')
    y1 = [1, 1, 1, 1, 1]; cv = np.cov(x1, y1); ic(x1, y1, cv); ic('----')
    y2 = [2, 2, 2, 2, 2]; cv = np.cov(x1, y2); ic(x1, y2, cv); ic('----')
    y3 = [1, 2, 3, 4, 5]; cv = np.cov(x1, y3); ic(x1, y3, cv); ic('----')
    y4 = [5, 4, 3, 2, 1]; cv = np.cov(x1, y4); ic(x1, y4, cv); ic('----');
    ic('----')
    x2 = [1, 5, 2, 4, 3]
    cv = np.cov(x2, y1); ic(x2, y1, cv); ic('----')
    cv = np.cov(x2, y2); ic(x2, y2, cv); ic('----')
    cv = np.cov(x2, y3); ic(x2, y3, cv); ic('----')
    cv = np.cov(x2, y4); ic(x2, y4, cv); ic('----')
    x3 = [1, 2, 3, 4, 5]
    cv = np.cov(x3, y4);ic(x3, y4, cv); ic('----')
    ic('----случайные распределения')
    n=15; np.random.seed(125)
    a=np.arange(n)
    rnd1=np.random.rand(n); a_rnd=a*(1+0.5*rnd1)
    rnd2=np.random.rand(n); b_rnd=a*(1+0.5*rnd2)
    # plt.plot(a_rnd); plt.plot(b_rnd);  plt.grid();  plt.show()
    cv = np.cov(a, a);          ic('=np.cov(a, a)=', cv)
    cv = np.cov(a_rnd, b_rnd);  ic('=np.cov(a_rnd, b_rnd)=', cv)

    cr = np.corrcoef(a, a);  ic('=np.cor(a, a)=', cr); ic('----')
    cr = np.corrcoef(a, b_rnd);  ic('=np.cor(a, b_rnd)=', cr); ic('----')
    cr = np.corrcoef(a_rnd, b_rnd);  ic('=np.cor(a_rnd, b_rnd)=', cr); ic('----')

def np_concat_array():
    a = [1,2,3]
    b = [10, 20, 30]
    zv = np.vstack((a, b))
    zh = np.hstack((a, b))
    print(zv)
    print(zh)
    c0=np.concatenate((a,b), axis=0)
    print(c0)
    # c1=np.concatenate((a,b), axis=1)
    # print(c1)

if __name__=="__main__":
    cov_01()
