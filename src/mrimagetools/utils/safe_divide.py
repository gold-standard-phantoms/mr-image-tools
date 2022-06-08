"""safe_divide utilities"""
import numpy as np


def safe_divide(num: np.ndarray, div: np.ndarray) -> np.ndarray:
    """safe divide function - does the division of each element
    from num and div except for the 0 values of div, in which case
    it'll just return 0
    Arrays need to be broadcastable

    :param num: the numerator for the safe division
    :type num: numpy ndarray
    :param div: the divisor for the safe division
    :type div: numpy ndarray
    :return: return the safe division, ignore the 0 values of div
    :rtype: numpy ndarray
    :raises ValueError: ValueError from numpy true_divide if
        arrays are not broadcastable
    """
    return np.true_divide(
        num, div, out=np.zeros_like(div), where=div > 0, dtype=np.float32
    )
