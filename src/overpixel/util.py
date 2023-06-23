import numpy as np
from tifffile import imread

def load_file(path: str) -> np.ndarray:
    narr = imread(path)
    return narr.transpose(1, 2, 0)