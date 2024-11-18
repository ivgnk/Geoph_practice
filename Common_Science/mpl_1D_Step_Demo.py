"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/step_demo.html#sphx-glr-gallery-lines-bars-and-markers-step-demo-py
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(14)
y = np.sin(x / 2)

plt.plot(x, y, 'o--', color='green', alpha=0.3, label='plot')
plt.step(x, y, label='step')
# Сдвинут для наглядности
plt.plot(x, y+1, drawstyle='steps-post', label='plot steps-post')

plt.legend(title='Parameter where:')
plt.title('Ступенчатые графики'); plt.grid(); plt.show()