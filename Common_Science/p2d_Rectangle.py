"""

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.html
"""

import numpy as np
from scipy.spatial import Rectangle
import matplotlib.pyplot as plt

def the_max_distance_point():
    rect=Rectangle([0,0],[1,1])
    print(rect)
    plt.plot(rect)
    plt.show()

if __name__ == "__main__":
    the_max_distance_point()