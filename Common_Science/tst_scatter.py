import matplotlib.pyplot as plt
import numpy as np
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

if __name__=="__main__":
    the_scatter()
