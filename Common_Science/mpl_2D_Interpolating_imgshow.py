"""
Interpolating images
https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html#interpolating-images
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(125)
A = np.random.rand(5, 5)

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
for ax, interp in zip(axs, ['nearest', 'bilinear', 'bicubic']):
    ax.imshow(A, interpolation=interp)
    ax.set_title(interp.capitalize())
    ax.grid(True)
plt.show()
