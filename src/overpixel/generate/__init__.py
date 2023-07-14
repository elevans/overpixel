import numpy as np
from typing import List

def _midpoints(narr: np.ndarray):
    """
    Calculate midpoints along each dimension of an input array.

    This function calculates midpoints along each dimension of an input array.
    It takes an input array `narr` and returns an array of midpoints along each dimension.
    Midpoints are calculated by taking the average between consecutive values along each dimension.

    The input array can be of any shape and dimension.
    The output array will have the same shape as the input array, but the size of each dimension is reduced by 1.

    :param x: Input array.
    :type x: ndarray

    :return: Array with midpoints along each dimension.
    :rtype: ndarray

    .. note::
        This function assumes that the input array has at least two elements along each dimension.

    """
    sl = ()
    for _ in range(narr.ndim):
        narr = (narr[sl + np.index_exp[:-1]] + narr[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return narr

def sphere(voxels: int, center: List[float] = None):
    """
    :param voxels: Number of voxels used to generate the sphere.
    :param center: Center coordinates of the sphere in (X, Y, Z) order.
    """
    # default to center of coordinate grid
    if center is None:
        center = [0.5, 0.5, 0.5]

    # prepare coordinates
    pln, row, col = np.indices((voxels, voxels, voxels)) / (voxels - 1)
    pln_c = _midpoints(pln)
    row_c = _midpoints(row)
    col_c = _midpoints(col)

    # define a sphere with given dimensions
    return (pln_c - center[0])**2 + (row_c - center[1])**2 + (col_c - center[2])**2 < 0.5**2