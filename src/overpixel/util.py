from typing import List
from tifffile import imread

def load_data(path: str):
    """
    Load input data from the path.
    """
    # TODO: Add logic to check the shape of the input array.
    narr = imread(path)
    return narr.transpose(1, 2, 0)
