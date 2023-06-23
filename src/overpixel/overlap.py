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

def run(batch: bool = False):
    if batch:
        # # call the run method on each dataset in the batch
        pass
    else:
        # call the run method only on the data once.
        pass
    # create new dataframe and append data
    # return dataframe
    return

def compute(narr: np.ndarray, threshold_method: str = "otsu", show: bool = False) -> List:
    """
    :param narr: Input numpy.ndarray with dimension shape (row, col, ch).
    :param threshold_method: Threshold method to use (default="otsu").
    """

    # create threshold based on config -- default to otsu
    mask_a = _get_mask(narr[:, :, 0], threshold_method)
    mask_b = _get_mask(narr[:, :, 1], threshold_method)

    # compute overlaps
    mutual_overlap = _get_mutual_overlap(mask_a, mask_b)
    a_overlap = _get_exclusive_overlap(mask_a, mask_b)
    b_overlap = _get_exclusive_overlap(mask_b, mask_a)
    if show:
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
        ax2.imshow(mask_a, cmap='Reds', alpha=0.5)
        ax2.imshow(mask_b, cmap='Blues', alpha=0.5)
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

    return [mutual_overlap, a_overlap, b_overlap]


def _get_exclusive_overlap(narr_a: np.ndarray, narr_b: np.ndarray) -> float:
    """
    Compute the percent of all pixels that are overlapped by mask "A", exclusively.
    """
    mask_OR = np.logical_or(narr_a, narr_b)
    count_OR = np.count_nonzero(mask_OR)
    narr_a_count = np.count_nonzero(narr_a)

    return narr_a_count / count_OR

def _get_mutual_overlap(narr_a: np.ndarray, narr_b: np.ndarray) -> float:
    """
    Compute the percent that mask "A" and mask "B" overlap.
    """
    mask_AND = np.logical_and(narr_a, narr_b)
    count_AND = np.count_nonzero(mask_AND)
    narr_a_count = np.count_nonzero(narr_a)
    narr_b_count = np.count_nonzero(narr_b)
    
    return count_AND / min(narr_a_count, narr_b_count)


def _get_mask(narr: np.ndarray, method: str) -> np.ndarray:
    """
    Compute a binary mask using the specified threshold method.
    """
    thres = threshold_methods[method](narr)
    return narr > thres

def _to_dataframe(data: Sequence[float]) -> pd.DataFrame:
    columns = ["mutual_overlap", "a_overlap", "b_overlap"]
    return