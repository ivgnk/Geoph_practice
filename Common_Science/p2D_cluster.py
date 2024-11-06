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

if __name__=="__main__":
    # print(the_whiten())
    # print(the_kmeans())
    # print(the_kmeans2())
    # tst_mark()
    # the_vstack()
    # the_2D()
    the_kmeans3()
