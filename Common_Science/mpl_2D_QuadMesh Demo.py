"""
QuadMesh Demo
https://matplotlib.org/stable/gallery/images_contours_and_fields/quadmesh_demo.html#sphx-glr-gallery-images-contours-and-fields-quadmesh-demo-py

"""
import numpy as np
from matplotlib import pyplot as plt

n = 12
x = np.linspace(-1.5, 1.5, n)
y = np.linspace(-1.5, 1.5, n)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2) / 5
# Нормируем
Z = (Z - Z.min()) / (Z.max() - Z.min())

# The color array can include masked values.
Zm = np.ma.masked_where(np.abs(Z) < 0.5 * np.max(Z), Z)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].pcolormesh(X, Y, Z, shading='gouraud')
axs[0].set_title('Without masked values')

# # You can control the color of the masked region.
# cmap = plt.colormaps[plt.rcParams['image.cmap']].with_extremes(bad='y')
# axs[1].pcolormesh(X, Y, Zm, shading='gouraud', cmap=cmap)
# axs[1].set_title('With masked values')

# Or use the default, which is transparent.
axs[1].pcolormesh(X, Y, Zm, shading='gouraud')
axs[1].set_title('With masked values')

fig.tight_layout()
plt.show()