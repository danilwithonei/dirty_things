import numpy as np


def percentage_non_black_pixels(mask_array: np.ndarray) -> float:
    total_pixels = mask_array.size
    non_black_pixels = np.sum(mask_array > 0)
    percentage = (non_black_pixels / total_pixels) * 100

    return float(percentage)
