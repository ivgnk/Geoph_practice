"""
https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html#numpy.vecdot
https://bemind.gitbook.io/neural/uchebniki/uchebniki-po-pandas-i-numpy/numpy/kak-rasschitat-vektornoe-proizvedenie-v-python
"""
import numpy as np
def vec_proizv():
    a = np.array([2, -3, -1])
    b = np.array([3, -1, -4])
    print(np.cross(a, b))

def vec_proizv2():
    a = np.array([3, 2, 1])
    b = np.array([6, 5, 4])
    print(np.cross(a, b))
    # cross_prod = np.cross(a, b)
    # return cross_prod

def vec_proizv3():
    a=dict(); b=dict(); c = dict()
    a['x'] = 3; a['y'] = 2; a['z'] = 1
    b['x'] = 6; b['y'] = 5; b['z'] = 4

    c['x'] = a['y']*b['z'] - a['z']*b['y']
    c['y'] = a['z']*b['x'] - a['x']*b['z']
    c['z'] = a['x']*b['y'] - a['y']*b['x']
    print(c)

def smesh_proizv():
    """
    https://ru.wikipedia.org/wiki/Произведения_векторов#Смешанное_произведение
    | ax ay az |          |
    | bx by bz |
    | cx cy cz |
    """
    a = np.array([-3, 2,-1])
    b = np.array([ 1, 0, 2])
    c = np.array([ 4, 5,-2])
    ab=np.cross(a, b) # векторное произв.
    print(np.dot(ab,c)) # скалярное произв.

def smesh_proizv2():
    """
    https://ru.wikipedia.org/wiki/Произведения_векторов#Смешанное_произведение
    | ax ay az |          |
    | bx by bz |
    | cx cy cz |
    """
    a = np.array([-3, 2,-1])
    b = np.array([ 1, 0, 2])
    c = np.array([ 4, 5,-2])
    # https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    m = np.stack((a,b,c))  # соединение векторов-строк
    det_a =np.linalg.det(m) # Вычисление определителя матрицы
    print(f'Определитель = {det_a} {det_a:6.3f}')

from icecream import ic
def vec_proizv4():
    a = np.array([3, 2, 1])
    b = np.array([6, 5, 4])
    ic(np.cross(a, b))
    ic(np.dot(a, b))  #  np.int64(32)
    ic(np.linalg.vecdot(a, b)) #  np.int64(32)
    ic(np.inner(a, b)) # np.int64(32)
    ic(np.outer(a, b)) # array([[18, 15, 12],
                       # [12, 10,  8],
                       # [ 6,  5,  4]])
    ic(np.matmul(a, b)) # np.int64(32)
    # ic(np.tensordot(a, b)) # - error
    ic(np.kron(a, b)) # array([18, 15, 12, 12, 10,  8,  6,  5,  4])


if __name__=="__main__":
    # print(vec_proizv3()) # [ 3 -6  3]
    # smesh_proizv2()
    vec_proizv4()
