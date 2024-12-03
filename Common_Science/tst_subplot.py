"""
plt.subplots
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html
"""

# Implementation of matplotlib function
import numpy as np
import matplotlib.pyplot as plt

# First create some toy data:
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)
y2 = np.sin(x ** 2)

# create 2 subplots
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(x, y1)
ax[1].plot(x, y2)

# plot 2 subplots
ax[0].set_title('Simple plot with sin(x)')
ax[1].set_title('Simple plot with sin(x**2)')

fig.suptitle('Stacked subplots in one direction')
plt.show()