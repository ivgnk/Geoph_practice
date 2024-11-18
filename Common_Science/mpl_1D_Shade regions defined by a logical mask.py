"""
Shade regions defined by a logical mask
https://matplotlib.org/stable/gallery/lines_bars_and_markers/span_regions.html#sphx-glr-gallery-lines-bars-and-markers-span-regions-py
"""

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2, 0.01)
s = np.sin(2*np.pi*t)
ll=len(t); sz=[10]*ll # размеры для scatter
plt.plot(t, s, color='black')
plt.axhline(0, color='black')
plt.scatter(t,s, s=sz)
plt.fill_between(t, 1, where=s > 0, facecolor='green', alpha=.5)
plt.fill_between(t, -1, where=s < 0, facecolor='red', alpha=.5)
plt.title('Shade regions defined by a logical mask'); plt.grid(); plt.show()