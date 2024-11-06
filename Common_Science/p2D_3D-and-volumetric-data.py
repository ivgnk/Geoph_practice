"""
3D and volumetric data
https://matplotlib.org/stable/plot_types/index.html#d-and-volumetric-data
"""
import matplotlib.pyplot as plt
import numpy as np

# https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.voxels.html#mpl_toolkits.mplot3d.axes3d.Axes3D.voxels
# https://matplotlib.org/stable/gallery/mplot3d/voxels.html#sphx-glr-gallery-mplot3d-voxels-py

def the_voxels():
    # Prepare some coordinates
    x, y, z = np.indices((8, 8, 8))
    print('x\n',x)
    print('y\n',y)
    print('z\n',z)
    # Draw cuboids in the top left and bottom right corners
    cube1 = (x < 3) & (y < 3) & (z < 3)
    cube2 = (x >= 5) & (y >= 5) & (z >= 5)
    levels = np.linspace(z.min(), z.max(), 7)

    # Combine the objects into a single boolean array
    voxelarray = cube1 | cube2

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('the_voxels')
    ax.voxels(voxelarray, edgecolor='k')
    # ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    plt.show()

def the_v2():
    x, y, z = np.indices((8, 8, 8))
    voxels = (x == y) | (y == z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors='#1f77b430', edgecolor='k')
    plt.show()

def the_col():
    # https://www.geeksforgeeks.org/how-to-draw-3d-cube-using-matplotlib-in-python/
    # Create axis
    axes = [5, 5, 5]
    # Create Data
    data = np.ones(axes, dtype=np.bool)
    # Control Transparency
    alpha = 1.0
    # Control colour
    colors = np.zeros(axes + [4], dtype=np.float32)
    print(colors.shape)
    # colors[0] = [1, 0, 0, alpha]  # red
    # colors[1] = [0, 1, 0, alpha]  # green
    # colors[2] = [0, 0, 1, alpha]  # blue
    # colors[3] = [1, 1, 0, alpha]  # yellow
    # colors[4] = [1, 1, 1, alpha]  # grey

    colors[0] = [1, 1, 1, alpha]
    colors[1] = [1, 1, 0, alpha]
    colors[2] = [0, 0, 1, alpha]
    colors[3] = [0, 1, 0, alpha]
    colors[4] = [1, 0, 0, alpha]

    # colors[0] = [1, 1, 1, alpha]
    # colors[1] = [1, 0, 0, alpha]
    # colors[2] = [1, 0, 0, alpha]
    # colors[3] = [1, 0, 1, alpha]
    # colors[4] = [1, 0, 1, alpha]

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Voxels is used to customizations of
    # the sizes, positions and colors.
    ax.voxels(data, facecolors=colors, edgecolors='grey')
    plt.show()

# BAD
# def the_hist():
#     from mpl_toolkits.mplot3d import Axes3D
#     data = np.random.normal(size=(3, 1000))
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     hist, xedges, yedges, zedges = np.histogramdd(data.T, bins=(5, 5, 5))
#     xpos, ypos, zpos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, zedges[:-1] + 0.25)
#     xpos = xpos.flatten()
#     ypos = ypos.flatten()
#     zpos = zpos.flatten()
#     dx = dy = dz = 0.5 * np.ones_like(zpos)
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, hist.flatten(), shade=True)
#     plt.show()

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
def the_KDE():
    data = np.random.multivariate_normal(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=1000)
    kde = gaussian_kde(data.T)
    x, y, z = np.mgrid[-3:3:30j, -3:3:30j, -3:3:30j]
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    density = kde(positions).reshape(x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density)
    plt.show()

if __name__ == "__main__":
    # the_voxels()
    # the_col()
    # the_hist()
    # the_KDE()

    fig = plt.figure()
    ax = fig.gca()
    # Make grid
    voxels = np.zeros((6, 6, 6))
    # Activate single Voxel
    voxels[1, 0, 4] = True

    x, y, z = np.indices(np.array(voxels.shape) + 1)

    ax.voxels(x * 0.5, y, z, voxels, edgecolor="k")
    ax.set_xlabel('0 - Dim')
    ax.set_ylabel('1 - Dim')
    ax.set_zlabel('2 - Dim')
    plt.show()