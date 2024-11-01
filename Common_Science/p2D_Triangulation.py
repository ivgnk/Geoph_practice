"""
Delaunay triangulation, convex hulls, and Voronoi diagrams
https://docs.scipy.org/doc/scipy/reference/spatial.html#delaunay-triangulation-convex-hulls-and-voronoi-diagrams
Plotting helpers
https://docs.scipy.org/doc/scipy/reference/spatial.html#plotting-helpers
"""

import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d

def the_delaunay_plot_2d():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.delaunay_plot_2d.html
    tit=f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    rng = np.random.default_rng(seed=125)
    points = rng.random((30, 2))
    tri = Delaunay(points)
    fig = delaunay_plot_2d(tri)
    plt.title(tit); plt.grid()
    plt.show()

from scipy.spatial import ConvexHull, convex_hull_plot_2d
def the_ConvexHull():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.htm
    tit=f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    rng = np.random.default_rng(seed=125)
    points = rng.random((30, 2))  # 30 random points in 2-D
    hull = ConvexHull(points)

    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-')
    plt.title(tit); plt.grid();  plt.show()

def the_ConvexHull_visu():
    tit=f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    rng = np.random.default_rng(seed=125)
    points = rng.random((30, 2))  # 30 random points in 2-D
    hull = ConvexHull(points)
    fig = convex_hull_plot_2d(hull)
    plt.title(tit); plt.grid();  plt.show()

from scipy.spatial import Voronoi, voronoi_plot_2d
def the_Voronoi():
    tit=f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    rng = np.random.default_rng(seed=125)
    points = rng.random((30, 2))  # 30 random points in 2-D
    vor = Voronoi(points)
    # fig = voronoi_plot_2d(vor)
    fig = voronoi_plot_2d(vor, show_vertices=True, line_colors='darkgreen',
                      line_width=1, line_alpha=0.9, point_size=6)
    plt.title(tit); plt.grid();  plt.show()
    print(f'{vor.vertices=}\n') # вершины
    print(f'{vor.regions=}\n') # многоугольники
    print(f'{vor.ridge_vertices=}\n') # вершины_хребта
    print(f'{vor.ridge_points=}\n') # точки хребтов

if __name__ == "__main__":
    # the_delaunay_plot_2d()
    # the_ConvexHull()
    # the_ConvexHull_visu()
    the_Voronoi()