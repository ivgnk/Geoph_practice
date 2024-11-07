"""
https://docs.scipy.org/doc/scipy/reference/cluster.html
"""
import inspect
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt


def the_whiten(view=False):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.htm
    """
    ini = np.array([[1.9, 2.3, 1.7],
                    [1.5, 2.5, 2.2],
                    [0.8, 0.6, 1.7, ]])
    # Вычилить res - ручной аналог whiten
    res=ini.copy(); res_=ini.copy()
    shp=ini.shape
    row = shp[0]; col = shp[1]
    for j in range(col):
        st=np.std(ini[:,j])
        res[:,j]/=st
    print(res)
    for j in range(col):
        mn=np.mean(res[:,j])
        res_[:, j]=res[:, j]-mn
    res2=whiten(ini)
    # 4 Ways to Compare Two NumPy Arrays
    # https://www.slingacademy.com/article/ways-to-compare-two-numpy-arrays-examples/
    print(f'1 var {res==res2 = }')
    print(f'2 var {np.array_equal(res, res2) = }')
    print(f'3 var {np.allclose(res, res2, atol=0.00001) = }')
    print(f'4 var {np.logical_and(np.sum(res-res2)<1e-8, np.sum(res2) == np.sum(res)) = }')
    print(ini)
    print('res_\n', res_)
    print(res2)
    if view:
        x=[i for i in range(col)]
        colo=['blue','orange','green']
        for j in range(col):
            plt.plot(x, ini[:, j],'-', label='ini-' + str(j), color=colo[j])
            plt.plot(x, res[:, j], '--', label='res-' + str(j), color=colo[j], linestyle='--')
        plt.grid(); plt.legend(); plt.show()

def fun(i,nn):
    if i==1:
        rng = np.random.default_rng(seed=125)
        points = rng.random((nn, 2))
        whitened = points.copy()
        codebook, distortion = kmeans(whitened, 2)
        name='not whiten, k=2'
    if i==2:
        rng = np.random.default_rng(seed=125)
        points = rng.random((nn, 2))
        whitened = whiten(points)
        book = np.array((whitened[0], whitened[2]))
        codebook, distortion = kmeans(whitened, book)
        name='whiten, book'
    if i == 3:
        rng = np.random.default_rng(seed=125)
        points = rng.random((nn, 2))
        whitened = whiten(points)
        codebook, distortion = kmeans(whitened, 2)
        name = 'whiten, k=2 '
    if i == 4:
        rng = np.random.default_rng(seed=125)
        points = rng.random((nn, 2))
        whitened = whiten(points)
        codebook, distortion = kmeans(whitened, 4)
        name = 'whiten, k=4'
    return whitened, codebook, distortion, name

def the_kmeans():
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    plt.figure(figsize=(16, 8))
    nn=50
    plt.suptitle(tit)
    for i in range(4):
        whitened, codebook, distortion, name=fun(i+1,nn)
        s=f'\nVar {i} - {name}'; print(s)
        print('codebook=\n',codebook)
        print('distortion=',distortion)

        plt.subplot(2, 2, i+1)
        plt.scatter(whitened[:, 0], whitened[:, 1])
        plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
        plt.title(s)
        plt.grid()
    plt.show()

import sys
import pandas as pd
from scipy.cluster.vq import kmeans, vq
def the_kmeans2():
    k = 4
    df = pd.read_csv("kmeans_dataset.csv")
    X = df.values[:,1:]
    centroids, distortion = kmeans(X, k)
    code, data = vq(X, centroids)
    print(code)
    colors = ['r', 'g', 'b', 'y']
    for i in range(k):
        # выбрать только данные наблюдений с меткой кластера == i
        ds = X[np.where(code == i)]
        # нанести на график наблюдения данных
        plt.scatter(ds[:,0], ds[:,1], c=colors[i],label=str(i))
        # нанести на график центроиды
        # plt.scatter(centroids[:,0], centroids[:,1],  s='o') # , facecolors='none', edgecolors='k'
        plt.plot(centroids[:,0], centroids[:,1], linestyle=' ', color='k', marker='+', markersize=16)  # c=z
    plt.grid(); plt.legend()
    plt.show()

def the_kmeans3():
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    k = 4; nn=[151, 112, 72, 165]
    XY=((1,1), (2, 5) , (4, 1.5), (5, 4.5))
    dxy=((-1,1), (-1,1), (-1,1), (-1,1))
    res=np.zeros((1,2)) # для удобства объедигнения в цикле
    for i in range(k):
        np.random.seed(i)
        x = np.random.uniform(dxy[i][0], dxy[i][1], nn[i])
        y = np.random.uniform(dxy[i][0], dxy[i][1], nn[i])
        x+=XY[i][0]
        y+=XY[i][1]
        res = np.vstack((res, np.array((x, y)).T))
    X = res[1:,:] # удаление лишней первой строки
    centroids, distortion = kmeans(X, k)
    code, data = vq(X, centroids)
    colors = ['r', 'g', 'b', 'y']
    for i in range(k):
        # выбрать только данные наблюдений с меткой кластера == i
        ds = X[np.where(code == i)]
        # наблюдения данных
        plt.scatter(ds[:,0], ds[:,1], c=colors[i],label=str(i))
        # центроиды
        plt.plot(centroids[:,0], centroids[:,1], linestyle=' ', color='k', marker='+', markersize=16)  # c=z
    plt.title(tit); plt.grid(); plt.legend(); plt.show()


def tst_mark():
    x = np.random.randn(60)
    y = np.random.randn(60)
    z = np.random.randn(60)
    # g = plt.plot(x, y, s=80, edgecolors='k', facecolors='none') #c=z
    plt.plot(x, y, linestyle=' ', color='k', marker='+', markersize=16)  # c=z

    # plt.colorbar()
    plt.show()

def the_vstack():
    z = np.array([-1, -1, -1])
    x = np.array([[0, 1, 2], [0, 1, 2]])
    print("First Input array : \n", x)
    y = np.array([[3, 4, 5], [3, 4, 5]])
    print("Second Input array : \n", y)
    res = np.vstack((z, x, y))
    print("Vertically stacked array:\n ", res)

def the_2D():
    z = np.array([-1, -1, -1])
    z0 = np.array([ 0,  0, 0])
    z1= np.array([[ 0,  0, 0], [ 1,  1, 1]])
    x=np.vstack((z, z1))
    print(x)
    print('-----------')
    x = np.array([1, 1, 1])
    y = np.array([2, 2, 2])
    z=np.array((x, y)).T
    print(z)

def the_rng_multivariate_normal():
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    # Cлучайное распределение. mean - среднее по x и y
    # cov - матрица ковариации
    # size - число точек
    rng = np.random.default_rng(125);
    cov = [([1, 0], [0, 1])]
    cov.append(([0, 1], [1, 0])) # RuntimeWarning: covariance is not symmetric positive-semidefinite
    cov.append(([0, 0], [0, 0]))
    cov.append(([1, 1], [1, 1]))
    plt.figure(figsize=(16, 8))
    plt.suptitle(tit+' разные средние и ковариационные матрицы')
    for ii,cov_ in enumerate(cov):
        plt.subplot(2, 2, ii + 1)
        plt.title('Ковар. матр. = '+str(cov_))
        for i in [0, 6]:
            for j in [0, 6]:
                mn = [i, j]
                a = rng.multivariate_normal(mean=mn, cov=cov_, size=15)
                plt.scatter(a[:, 0], a[:, 1], label=str(mn))
        plt.grid();  plt.legend()
    plt.show()

def the_rng_multivariate_normal_view():
    nn=75
    cov = [([1, 0.25], [0.25, 1])]
    cov.append(([1, 0.75], [0.75, 1]))
    cov.append(([5, 0], [0, 5]))
    cov.append(([2, 0.25], [0.25, 2]))
    # cov.append(([1, 0.25], [0.75, 1]))
    # cov.append(([1, 0.75], [0.25, 1]))
    the_rng_multivariate_normal2(cov, nn)
    # cov=[]
    # mi=0.25; ma=2.25
    # cov.append(([1, mi], [mi, 1]))
    # cov.append(([1, mi], [ma, 1]))
    # cov.append(([1, ma], [mi, 1]))
    # cov.append(([1, ma], [ma, 1]))
    # the_rng_multivariate_normal2(cov, nn)

def the_rng_multivariate_normal2(the_cov,sz):
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    # Cлучайное распределение. mean - среднее по x и y
    # cov - матрица ковариации
    # size - число точек
    rng = np.random.default_rng(125);
    cov = the_cov
    plt.figure(figsize=(16, 8))
    plt.suptitle(tit+' разные средние и ковариационные матрицы')
    for ii,cov_ in enumerate(cov):
        plt.subplot(2, 2, ii + 1)
        plt.title('Ковариационная матрица = '+str(cov_))
        for i in [0, 6]:
            for j in [0, 6]:
                mn = [i, j]
                a = rng.multivariate_normal(mean=mn, cov=cov_, size=sz)
                plt.scatter(a[:, 0], a[:, 1], label=str(mn))
        plt.grid();  plt.legend()
    plt.show()

from scipy.cluster.vq import kmeans2
def the_kmeans2fun():
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html
    """
    ###### -1 - исходные данные
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    rng = np.random.default_rng(125)
    # Для плоскости задание 3 наборов случайных точек многомерного (2-мерного)
    # случайного распределения. mean - среднее по x и y
    # cov - матрица ковариации
    # size - число точек
    mmean=np.array([[0,6], [2,0], [6,4]]); nname='ист. центры'
    ca=[[2,  1], [ 1, 1.5]]
    cb=[[1, -1], [-1, 3.0]]
    cc=[[5,  0], [ 0, 1.2]]
    a = rng.multivariate_normal(mean = mmean[0], cov = ca, size=45)
    b = rng.multivariate_normal(mean = mmean[1], cov = cb, size=30)
    c = rng.multivariate_normal(mean = mmean[2], cov = cc, size=25)
    plt.figure(figsize=(16, 5))
    plt.title(tit)
    plt.scatter(a[:, 0], a[:, 1], label='a - cov matr '+str(ca))
    plt.scatter(b[:, 0], b[:, 1], label='b - cov matr '+str(cb))
    plt.scatter(c[:, 0], c[:, 1], label='c - cov matr '+str(cc))
    plt.grid();
    plt.gca().set_aspect("equal") # принудительно сделать квадратную сетку
    plt.legend();  plt.show()

    ###### -2 - кластеризация
    # z = np.concatenate((a, b, c))
    # rng.shuffle(z) # перемешивание
    # centroid, label = kmeans2(z, 3, iter=90, minit='points')
    # print('centroid - близки к указанным в mean')
    # print(centroid); print(mmean)
    # counts = np.bincount(label)
    # print('counts = \n',counts)
    # без shuffle
    print('\nбез shuffle')
    z = np.concatenate((a, b, c))
    centroid, label = kmeans2(z, 3, iter=90, minit='points')
    print('centroid - близки к указанным в mean')
    print(centroid)
    print(nname); print(mmean)
    counts = np.bincount(label)
    print('counts = \n',counts)
    ###### -3 - Plot the clusters.
    w0 = z[label == 0]
    w1 = z[label == 1]
    w2 = z[label == 2]
    plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
    plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
    plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
    plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
    plt.plot(mmean[:,0], mmean[:,1], 'r*', label=nname)
    plt.axis('equal'); plt.grid()
    plt.legend(shadow=True)
    plt.show()


if __name__=="__main__":
    # print(the_whiten())
    # print(the_kmeans())
    # print(the_kmeans2())
    # tst_mark()
    # the_vstack()
    # the_2D()
    # the_kmeans3()
    # the_rng_multivariate_normal()
    # the_rng_multivariate_normal_view()
    the_kmeans2fun()