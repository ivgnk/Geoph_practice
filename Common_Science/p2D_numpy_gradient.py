"""
numpy.gradient
https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
"""

import numpy as np
def gradient_example1():  # 1-мерный массив
    f = np.array([1, 2, 4, 7, 11, 16])
    print(np.gradient(f)) #  array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    print(np.gradient(f,1)) #  array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    print(np.gradient(f, 2)) #  array([0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

if __name__=="__main__":
    gradient_example1()