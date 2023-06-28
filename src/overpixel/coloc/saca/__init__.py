import numpy as np
from skimage.filters import threshold_otsu
from overpixel.coloc.saca.adaptive_smoothed_kendall_tau import AdaptiveSmoothedKendallTau
from overpixel.array import scale

def run(narr_a: np.ndarray, narr_b: np.ndarray, threshold_a: float = None, threshold_b: float = None) -> np.ndarray:
    """Run SACA

    :param narr_a:
    :param narr_b:
    :param threshold_a:
    :param threshold_b:
    """
    if not np.issubdtype(narr_a.dtype, np.number) and np.issubdtype(narr_b.dtype, np.number):
        raise TypeError(f"An input array is non-numeric. narr_a: {narr_a.dtype} | narr_b: {narr_b.dtype}")
    if narr_a.shape != narr_b.shape:
        raise ValueError(f"Input array shapes do not match: narr_a: {narr_a.shape} | narr_b: {narr_b.shape}")
    if threshold_a is not None and threshold_b is None:
        raise ValueError(f"Two threshold values are needed, but only one was provided (threshold={threshold_a}).")
    if threshold_a is None and threshold_b is not None:
        raise ValueError(f"Two threshold values are needed, but only one was provided (threshold={threshold_b}).")
    
    # scale input arrays to a range of 0,1
    narr_a = scale.scale_to_0_1(narr_a)
    narr_b = scale.scale_to_0_1(narr_b)

    # get thresholds if they are None
    if threshold_a is None and threshold_b is None:
        threshold_a = threshold_otsu(narr_a)
        threshold_b = threshold_otsu(narr_b)

    askt = AdaptiveSmoothedKendallTau(narr_a, narr_b, threshold_a, threshold_b)

    return askt.compute()