import numpy as np

from typing import List, Sequence


def exclusive_overlaps(masks: Sequence[np.ndarray]) -> List[float]:
    """
    Calculates the overlap percentage of each mask in the input list with the logical OR of all masks.

    :param masks: A sequence of numpy arrays representing masks.
        Each mask should be of dtype uint8 and contain only the values 0 and 255.
    :type masks: Sequence[numpy.ndarray]
    :return: A list of float values representing the overlap percentage of each mask
        with the logical OR of all masks. The list is in the same order as the input masks.
    :rtype: List[float]
    :raises ValueError: If the number of masks is less than 2.
    :raises TypeError: If any mask is not of dtype uint8 or contains values other than 0 and 255.

    .. note::
        The overlap percentage of a mask is calculated as the ratio of the number of True pixels
        in the mask to the number of True pixels in the logical OR of all masks.
    """
    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    for i in range(len(masks)):
        if not _is_binary(masks[i]):
            raise TypeError(
                f"Mask at index {i} is not boolean or binary (uint8). Type: {masks[i].dtype}"
            )

    results = []
    masks_OR = np.logical_or.reduce(masks)
    count_OR = np.count_nonzero(masks_OR)
    for mask in masks:
        count_mask = np.count_nonzero(mask)
        results.append(count_mask / count_OR)

    return results


def mutual_overlap(masks: List[np.ndarray]) -> float:
    """Compute the intersection over minimum count for a list of binary numpy arrays.

    This function computes the intersection over minimum count for a list of binary numpy arrays.
    The intersection over minimum count is a measure of overlap between the masks, calculated
    by dividing the count of pixels that are 255 in all the masks by the minimum count of pixels
    that are 255 in any individual mask.

   :param masks: A list of numpy arrays where each array represents a binary mask.
                  The arrays must have a data type of uint8 and contain only two distinct values: 0 and 255.
   :type masks: List[numpy.ndarray]
   :return: The intersection over minimum count, a float value between 0 and 1.
   :rtype: float
   :raises ValueError: If the number of masks in the input list is less than 2.
   :raises TypeError: If any mask in the list is not binary (contains values other than 0 and 255).

    """
    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    for i in range(len(masks)):
        if not _is_binary(masks[i]):
            raise TypeError(
                f"Mask at index {i} is not boolean or binary (uint8). Type: {masks[i].dtype}"
            )

    masks_AND = np.logical_and.reduce(masks)
    count_AND = np.count_nonzero(masks_AND)
    counts = [np.count_nonzero(mask) for mask in masks]

    return count_AND / min(counts)


def _is_binary(narr: np.ndarray) -> bool:
    # check if boolean array
    if np.issubdtype(narr.dtype, np.bool_) or np.issubdtype(narr.dtype, np.uint8):
        return True
    else:
        return False
