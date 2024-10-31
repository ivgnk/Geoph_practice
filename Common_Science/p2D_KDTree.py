"""
Functions from
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
"""
import numpy as np
from scipy.spatial import KDTree
from icecream import ic
import matplotlib.pyplot as plt

def the_count_neighbors():
    """
    count_neighbors(other, r, p=2.0, weights=None, cumulative=True)[source]
    distance(x1, x2, p) <= r
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.count_neighbors.html
    """

    rng = np.random.default_rng(seed=125);  nn=5
    points1 = rng.random((nn, 2));  points2 = rng.random((nn, 2))
    kd_tree1 = KDTree(points1)
    kd_tree2 = KDTree(points2)
    ic(kd_tree1.count_neighbors(kd_tree2, 0.2))
    # plt.plot(points1[:,0], points1[:,1],'bo')
    # plt.plot(points2[:,0], points2[:,1],'rx')
    plt.scatter(points1[:, 0], points1[:, 1],label='points1')
    plt.scatter(points2[:, 0], points2[:, 1],label='points2')
    for i in range(nn):
        plt.annotate(str(i), (points1[i, 0], points1[i, 1]))
        plt.annotate(str(i), (points2[i, 0], points2[i, 1]))
    plt.legend(); plt.grid();  plt.show()

from scipy.spatial import distance_matrix
def the_distance_matrix():
    rng = np.random.default_rng(seed=125); nn=5
    points1 = rng.random((nn, 2));  points2 = rng.random((nn, 2))
    kd_tree1 = KDTree(points1);  kd_tree2 = KDTree(points2)
    # 100 - т.е. вычислять для всех пар точек
    sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 100)
    # {(np.int32(0), np.int32(0)): np.float64(0.3403898529432339),
    #                 (np.int32(0), np.int32(1)): np.float64(0.6968233109714843), .......
    # print(f'{dict(sdm)=}')
    res=sdm.toarray();       mmax=0.2
    print(f'{res=}\n'); print(f'{res[res<=mmax]=}\n')
    # ----------- выбираем координаты для res<=0.2
    print(f'{sdm=}\n') # dictionary
    print('i points1(x,y) j points2(x,y) distance')
    for k,v in sdm.items():
        if v<mmax: print(k[0], points1[k[0]], k[1], points2[k[1]], v)
    print('\n')
    #----------- res2
    res2=distance_matrix(points1,points2)
    print(f'{res2=}\n')
    # Plot
    plt.scatter(points1[:, 0], points1[:, 1],label='points1')
    plt.scatter(points2[:, 0], points2[:, 1],label='points2')
    for i in range(nn):
        plt.annotate(str(i), (points1[i, 0], points1[i, 1]))
        plt.annotate(str(i), (points2[i, 0], points2[i, 1]))
    for k,v in sdm.items():
        if v<mmax:
            x=[points1[k[0]][0], points2[k[1]][0]]; y=[points1[k[0]][1], points2[k[1]][1]]
            plt.plot(x, y, color='r')
    plt.legend(); plt.grid();  plt.show()

def the_scatter():
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py
    np.random.seed(19680801)
    fig, ax = plt.subplots()
    for color in ['tab:blue', 'tab:orange', 'tab:green']:
        n = 750
        x, y = np.random.rand(2, n)
        scale = 200.0 * np.random.rand(n)
        ax.scatter(x, y, c=color, s=scale, label=color,
                   alpha=0.3, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.show()

import matplotlib.colors as mcolors
def the_query():
    # --1-- Calc
    rng = np.random.default_rng(seed=125); nn=5
    points1 = rng.random((nn, 2));  points2 = rng.random((nn, 2))
    kd_tree1 = KDTree(points1)
    k=[1,2,3,4,5]
    k=[1,2,3]
    dd, ii = kd_tree1.query(points2,k=k)
    for i in range(len(dd)):
        print(dd[i], ii[i])
    # --2-- Plot
    plt.scatter(points1[:, 0], points1[:, 1],label='points1')
    plt.scatter(points2[:, 0], points2[:, 1],label='points2')
    for i in range(nn):
        plt.annotate(str(i), (points1[i, 0], points1[i, 1]))
        plt.annotate(str(i), (points2[i, 0], points2[i, 1]))
    col = list(mcolors.BASE_COLORS.keys()) # ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(nn): #по точкам
        for j in range(len(k)):
            x = [points2[i][0], points1[ii[i][j]][0]]
            y = [points2[i][1], points1[ii[i][j]][1]]
            plt.plot(x,y, color=col[i])
    plt.title(f'Ближайшие {k=} точек ponts1 для каждой points2')
    plt.legend(); plt.grid();  plt.show()

def the_query_ball_point():
    # --1-- Calc
    rng = np.random.default_rng(seed=125); nn=5
    points1 = rng.random((nn, 2));  points2 = rng.random((nn, 2))
    kd_tree1 = KDTree(points1)
    rr=0.3
    dd = kd_tree1.query_ball_point(points2,r=rr)
    print(dd)
    for i in range(len(dd)): print(i, dd[i])
    # --2-- Plot
    plt.scatter(points1[:, 0], points1[:, 1],label='points1')
    plt.scatter(points2[:, 0], points2[:, 1],label='points2')
    for i in range(nn):
        plt.annotate(str(i), (points1[i, 0], points1[i, 1]))
        plt.annotate(str(i), (points2[i, 0], points2[i, 1]))
    col = list(mcolors.BASE_COLORS.keys()) # ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(nn): #по точкам
        for j in range(len(dd[i])):
            x = [points2[i][0], points1[dd[i][j]][0]]
            y = [points2[i][1], points1[dd[i][j]][1]]
            plt.plot(x,y, color=col[i])
    plt.title(f'Ближайшие точки ponts1 в радиусе {rr:.2f} для каждой points2')
    plt.legend(); plt.grid();  plt.show()

def the_query_ball_tree():
    # --1-- Calc
    rng = np.random.default_rng(seed=125); nn=5
    points1 = rng.random((nn, 2));  points2 = rng.random((nn, 2))
    kd_tree1 = KDTree(points1); kd_tree2 = KDTree(points2)
    rr = 0.2
    indexes = kd_tree1.query_ball_tree(kd_tree2, r=rr)
    print(indexes)
    # --2-- Plot
    plt.scatter(points1[:, 0], points1[:, 1],label='points1')
    plt.scatter(points2[:, 0], points2[:, 1],label='points2')
    for i in range(nn):
        plt.annotate(str(i), (points1[i, 0], points1[i, 1]))
        plt.annotate(str(i), (points2[i, 0], points2[i, 1]))
    col = list(mcolors.BASE_COLORS.keys()) # ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(nn): #по точкам
        for j in indexes[i]:
            plt.plot([points1[i, 0], points2[j, 0]],
                     [points1[i, 1], points2[j, 1]], color=col[i])
    plt.title(f'Ближайшие точки ponts1 в радиусе {rr:.2f} для каждой points2 \n '
              f'функция query_ball_tree')
    plt.legend(); plt.grid();  plt.show()

def the_query_pairs():
    # --1-- Calc
    rng = np.random.default_rng(seed=125); nn=26
    points1 = rng.random((nn, 2))
    kd_tree1 = KDTree(points1)
    rr = 0.2
    pairs = kd_tree1.query_pairs(r=rr)
    print(pairs)
    # --2-- Plot
    plt.title(f'Ближайшие точки в радиусе {rr:.2f} функция query_pairs')
    for (i, j) in pairs:
        plt.plot([points1[i, 0], points1[j, 0]],
                 [points1[i, 1], points1[j, 1]], "-r")
    plt.plot(points1[:, 0], points1[:, 1], "ob", markersize=4)
    plt.grid(); plt.show()

if __name__=="__main__":
    # the_count_neighbors()
    # the_distance_matrix()
    # the_query()
    # the_query_ball_point()
    # the_query_ball_tree()
    the_query_pairs()
    # the_scatter()
    # x = [1, 2, 3, 4]
    # y = [1, 2, 3, 4]
    # x1 = [5, 6, 7, 8]
    # y1 = [6, 7, 8, 9]
    # for i in range(4):
    #     plt.plot(x+x1,y+y1)
    # plt.show()
    # import matplotlib.colors as mcolors
    # print(list(mcolors.BASE_COLORS.keys()))

