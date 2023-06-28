import numpy as np

from typing import List, Sequence


def exclusive_overlaps(masks: Sequence[np.ndarray]) -> List[float]:
    """Calculate the percentage of pixels exclusively overlapped by the first
    mask in the given list.

    This function takes a list of numpy ndarrays representing masks, where the first mask (mask A)
    is at index 0. It calculates the percentage of pixels in each mask that are exclusively overlapped
    by the first mask, and returns a list of floats representing these percentages.
    The masks can be boolean or binary (uint8) arrays.

    :param masks: A list of numpy ndarrays representing masks (bool or uint8), where mask A is at index 0.
    :return: A list of floats representing the percentages of pixels exclusively overlapped by mask A.
    :rtype: List[float]
    :raises ValueError: If less than two masks are provided.
    :raises TypeError: If any mask in the list is not boolean or binary (uint8).
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
    """Calculate the percentage of pixels mutually overlapped by multiple masks.

    This function calculates the mutual overlap percentage of a list of masks.
    The masks can be boolean or binary (uint8) arrays.

    :param masks: A list of numpy ndarrays representing masks (bool or uint8).
    :return: A float representing the mutual overlap of all the masks in the input list.
    :rtype: float
    :raises ValueError: If less than two masks are provided.
    :raises TypeError: If any mask in the list is not boolean or binary (uint8).
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
