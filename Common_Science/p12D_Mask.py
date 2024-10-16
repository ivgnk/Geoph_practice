"""
The numpy.ma module
https://numpy.org/doc/stable/reference/maskedarray.generic.html
"""
import numpy as np
import numpy.ma as ma
from statistics import mean

def prf_d01():
    x = np.array([1, 2, 3, -1, 5])
    mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
    # Среднее значение
    print(mx.mean(), mean([1,2,3,5]))

import numpy.ma
def prf_d02():
    # Маскирование целых и вещественных массивов
    xi = np.array([1, 2, 3]);  xf = np.array([1.0, 2.0, 3.0]);
    mxi = ma.masked_array(xi,mask=[0, 0, 0])
    mxf = ma.masked_array(xf, mask=[1, 1, 1])
    print(mxi.data,mxf.data)
    print(mxi.mask,mxf.mask)
    print(mxi.fill_value,mxf.fill_value)

def prf_d03():
    # ------------- 1 - ma.asarray
    # x = np.ma.asarray([1, 2, 3])
    # xm=np.ma.asarray(x)

    # ------------- 2 - ma.array
    # xm = np.ma.array([1., -1, np.nan, np.inf], mask=[1] + [0] * 3)
    # print(xm)
    # print(xm.data)
    # print(xm.mask)
    # print(xm.fill_value)
    # ------------- 3 - ma.fix_invalid
    # xm = np.ma.array([1., -1, np.nan, np.inf], mask=[1] + [0] * 3)
    # xm = np.ma.fix_invalid(xm)
    # print(xm)
    # print(xm.data)
    # print(xm.mask)
    # print(xm.fill_value)
    # ------------- 4 - ma.masked_equal
    # x = np.array([0, 1, 2, 3])
    # xm=ma.masked_equal(x, 2)
    # print(xm)
    # print(xm.data)
    # print(xm.mask)
    # ------------- 5 - ma.masked_inside
    # x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
    # xm = ma.masked_inside(x, -0.3, 0.3)
    # print(xm)
    # print(xm.data)
    # print(xm.mask)
    # ------------- 6 - ma.masked_outside
    # x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
    # xm = ma.masked_outside(x, -0.3, 0.3)
    # print(xm)
    # print(xm.data)
    # print(xm.mask)
    # ------------- 7 - ma.masked_where
    # x = np.arange(4) # [0, 1, 2, 3]
    # xm = ma.masked_where(x <= 2, x)
    # print(xm)
    # print(xm.data)
    # print(xm.mask)
    # ------------- 8 - ma.all
    # x=np.ma.array([1, 2, 3, np.nan, np.inf])
    # xm = np.ma.fix_invalid(x)
    # print(xm)
    # print(ma.count(xm))
    # print(ma.count_masked(xm))
    pass


prf_d03()
# x = np.ma.array([1., -1, np.nan, np.inf], mask=[1] + [0]*3)
# print(x.data)
# print(x.mask)
# print(x.fill_value)