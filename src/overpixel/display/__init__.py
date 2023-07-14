import numpy as np
from overpixel.generate import _midpoints
from matplotlib import pyplot as plt
from typing import Tuple

def show_3d(data: np.ndarray, size: Tuple[int] = None):
    # default to figure size
    if size is None:
        size = (8, 8)

    # compute coordnates to attach RGB values
    pln, row, col = np.indices((data.shape[0] + 1, data.shape[1] + 1, data.shape[2] + 1)) / (data.shape[0])
    pln_c = _midpoints(pln)
    row_c = _midpoints(row)
    col_c = _midpoints(col)

    # combine the color components
    colors = np.zeros(data.shape + (3,))
    colors[..., 0] = pln_c
    colors[..., 1] = row_c
    colors[..., 2] = col_c

    # plot the voxel sphere
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(pln, row, col, data,
            facecolors=colors,
            edgecolors=np.clip(2*colors - 0.5, 0, 1),  # Brighter
            linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()