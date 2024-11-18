"""
Plotting with keywords
https://matplotlib.org/stable/gallery/misc/keyword_plotting.html#sphx-glr-gallery-misc-keyword-plotting-py
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(125)
data = {'a': np.arange(50), # x -координата
        'c': np.random.randint(0, 50, 50), # цвет
        'd': np.random.randn(50)} # размер
data['b'] = data['a'] + 10 * np.random.randn(50)  # y -координата
data['d'] = np.abs(data['d']) * 100  # размер меняем

fig, ax = plt.subplots()
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set(xlabel='entry a', ylabel='entry b')
plt.grid(); plt.show()