"""
Simple Axisline4
https://matplotlib.org/stable/gallery/axes_grid1/simple_axisline4.html#sphx-glr-gallery-axes-grid1-simple-axisline4-py
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot

plt.figure(figsize=(10,8))
ax = host_subplot(111)
n=0.3
xx = np.arange(0, 2*np.pi*n, 0.01); ll=len(xx)
ax.plot(xx, np.sin(xx))
ax.scatter(xx, np.sin(xx),s=[4 for i in range(ll)], c='black')

ax2 = ax.twin()  # ax2 is responsible for "top" axis and "right" axis
ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
               labels=["$0$", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.axis["top"].major_ticklabels.set_visible(True)
plt.grid(); plt.show()