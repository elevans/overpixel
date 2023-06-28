import numpy as np

def scale_to_0_1(narr: np.ndarray) -> np.ndarray:
    """
    Scale the input numpy array to a range of (0, 1), where the minimum value
    is mapped to 0 and the maximum value is mapped to 1.

    :param narr: The input numpy array to be scaled.
    :type narr: np.ndarray

    :return: The scaled numpy array.
    :rtype: np.ndarray

    :raises TypeError: If the input array is not numeric.

    :example:
        >>> arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        >>> scaled_arr = scale_to_0_1(arr)
        >>> print(scaled_arr)
        [0.   0.25 0.5  0.75 1.  ]
    """
    if not np.issubdtype(narr.dtype, np.number):
        raise TypeError(f"Input array is not numeric: {narr.dtype}")
    
    return (narr - narr.min()) / (narr.max() - narr.min())