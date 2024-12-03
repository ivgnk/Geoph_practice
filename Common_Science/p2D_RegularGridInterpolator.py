"""
RegularGridInterpolator
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
"""
def tst_RegularGridInterpolator1():
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np
    def f(x, y, z):
        return 2 * x**3 + 3 * y**2 - z
    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    z = np.linspace(7, 9, 33)
    xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    data = f(xg, yg, zg)
    # data is now a 3-D array with data[i, j, k] = f(x[i], y[j], z[k]).
    # Next, define an interpolating function from this data:
    interp = RegularGridInterpolator((x, y, z), data)
    # Evaluate the interpolating function at the two points
    # (x,y,z) = (2.1, 6.2, 8.3) and (3.3, 5.2, 7.1):
    pts = np.array([[2.1, 6.2, 8.3],
                    [3.3, 5.2, 7.1]])
    res1 = interp(pts)
    print(res1) # array([ 125.80469388,  146.30069388])
    # which is indeed a close approximation to
    # f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)
    # (125.54200000000002, 145.894)

if __name__=="__main__":
    tst_RegularGridInterpolator1()
