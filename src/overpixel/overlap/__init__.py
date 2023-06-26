import skimage.filters as skif
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import patches
from typing import List, Sequence

threshold_methods = {
    "isodata": skif.threshold_isodata,
    "li": skif.threshold_li,
    "local": skif.threshold_local,
    "mean": skif.threshold_mean,
    "minimum": skif.threshold_minimum,
    "multiotsu": skif.threshold_multiotsu,
    "niblack": skif.threshold_niblack,
    "otsu": skif.threshold_otsu,
    "sauvola": skif.threshold_sauvola,
    "triangle": skif.threshold_triangle,
    "yen": skif.threshold_yen
}

def run(narr: np.ndarray, threshold_method: str = None, threshold_values: Sequence[float] = None, show: bool = False) -> List[float]:
    """
    :param
    """
    # TODO: make threshold_values -> Sequence[float] matching narr channel length
    # TODO: adapt to apply to n-channel data
    if threshold_method is None and threshold_values is None:
        return _compute(narr, threshold_method="otsu", show=show)
    elif threshold_method is not None and threshold_values is None:
        return _compute(narr, threshold_method=threshold_method, show=show)
    elif threshold_values is not None and threshold_method is None:
        if narr.shape[2] != len(threshold_values):
            raise ValueError(f"The number of channels ({narr.shape[2]}) and threshold values ({len(threshold_values)}) do not match.")
        return _compute(narr, threshold_values=threshold_values, show=show)
    else:
        raise ValueError(f"A threshold method ({threshold_method}) and threshold value ({threshold_values}) was provided, only one is supported.")


def _compute(narr: np.ndarray, threshold_method: str = None, threshold_values: float = None, show: bool = False) -> List[float]:
    """
    :param narr: Input numpy.ndarray with dimension shape (row, col, ch).
    :param threshold_method: Threshold method to use (default="otsu").
    """
    masks = []
    overlaps = []
    # get mask from threshold or predefined threshold value
    if threshold_method is not None and threshold_values is None:
        for i in range(narr.shape[2]):
            masks.append(_get_mask(narr[:, :, i], threshold_method))
    elif threshold_values is not None and threshold_method is None:
        for i in range(narr.shape[2]):
            masks.append(narr[:, :, i] > threshold_values[i])
    else:
        raise ValueError(f"Threshold method ({threshold_method}) and threshold values ({threshold_values}) provided are incompatible.")

    # compute overlaps
    overlaps.append(get_mutual_overlap(masks))
    overlaps.extend(get_exclusive_overlaps(masks))

    if show:
        _show(narr, masks)

    return overlaps


def get_exclusive_overlaps(masks: List[np.ndarray]) -> List[float]:
    """Calculate the percentage of pixels exclusively overlapped by the first
    mask in the given list.

    This function takes a list of numpy ndarrays representing masks, where the first mask (mask A)
    is at index 0. It calculates the percentage of pixels in each mask that are exclusively overlapped
    by the first mask, and returns a list of floats representing these percentages.

    :param masks: A list of numpy ndarrays representing masks, where mask A is at index 0.
    :return: A list of floats representing the percentages of pixels exclusively overlapped by mask A.
    """
    results = []
    masks_OR = np.logical_or.reduce(masks)
    count_OR = np.count_nonzero(masks_OR)
    for mask in masks:
        count_mask = np.count_nonzero(mask)
        results.append(count_mask / count_OR)

    return results

def get_mutual_overlap(masks: List[np.ndarray]) -> float:
    """
    """
    if len(masks) < 2:
        raise ValueError("At least two masks are required.")
    
    masks_AND = np.logical_and.reduce(masks)
    count_AND = np.count_nonzero(masks_AND)
    counts = [np.count_nonzero(mask) for mask in masks]

    return count_AND / min(counts)

def _get_mask(narr: np.ndarray, method: str) -> np.ndarray:
    """
    Compute a binary mask using the specified threshold method.
    """
    thres = threshold_methods[method](narr)
    return narr > thres

def _show(narr: np.ndarray, masks: Sequence[np.ndarray]):
    """
    Display the data
    """
    # setup matplotlib figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7), sharex=True, sharey=True)
    fig.suptitle("sample view")
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]
    # draw images
    ax0.imshow(narr[:, :, 0], cmap='gray', aspect='equal')
    ax0.set_title("input a")
    ax1.imshow(narr[:, :, 1], cmap='gray', aspect='equal')
    ax1.set_title("input b")
    ax2.imshow(masks[0], cmap='Reds', alpha=0.5)
    ax2.imshow(masks[1], cmap='Blues', alpha=0.5)
    ax2.set_title("overlap")
    # create color legend
    legend_patches = [
        patches.Patch(color='red', alpha=0.5, label='Mask A'),
        patches.Patch(color='blue', alpha=0.5, label='Mask B')
        ]
    ax3.legend(handles=legend_patches, loc="center", bbox_to_anchor=(0, 0.5), frameon=False)
    ax3.axis('off')
    # display plot
    fig.tight_layout()
    plt.show()
