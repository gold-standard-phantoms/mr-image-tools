"""safe_divide utilities"""
from typing import Union

import numpy as np

InputType = Union[np.ndarray, float, int, complex]


def safe_divide(numerator: InputType, divisor: InputType) -> InputType:
    """safe divide function - does the division of each element
    from numerator and divisor except for the 0 values of divisor, in which case
    it'll just return 0
    Arrays need to be broadcastable
    It can also take floats and integers as inputs

    :param numerator: the numeratorerator for the safe division
    :type numerator: numpy ndarray or float or int or complex
    :param divisor: the divisor for the safe division
    :type divisor: numpy ndarray or float or int or complex
    :return: return the safe division, ignore the 0 values of divisor
    :rtype: numpy ndarray or float or int or complex
    :raises ValueError: ValueError from numpy true_divide if
        arrays are not broadcastable
    """

    if np.iscomplexobj(divisor) or np.iscomplexobj(numerator):
        if isinstance(divisor, np.ndarray) and isinstance(numerator, np.ndarray):
            return np.true_divide(
                numerator,
                divisor,
                where=divisor > 0,
                dtype=np.complex128,
                casting="unsafe",
            )

        if not isinstance(divisor, np.ndarray) and not isinstance(
            numerator, np.ndarray
        ):
            if divisor == 0:
                return 0 + 0j
            return np.true_divide(numerator, divisor, dtype=np.complex128)
        if not isinstance(divisor, np.ndarray):
            if divisor == 0:
                return np.zeros_like(numerator, dtype=np.complex128)
            return np.true_divide(numerator, divisor, dtype=np.complex128)
        if not isinstance(numerator, np.ndarray):
            return np.true_divide(
                numerator, divisor, where=divisor > 0, dtype=np.complex128
            )

    if isinstance(divisor, np.ndarray) and isinstance(numerator, np.ndarray):
        return np.true_divide(
            numerator,
            divisor,
            out=np.zeros_like(divisor),
            where=divisor > 0,
            dtype=np.float64,
            casting="unsafe",
        )

    if not isinstance(divisor, np.ndarray) and not isinstance(numerator, np.ndarray):
        if divisor == 0:
            return 0
        return np.true_divide(numerator, divisor, dtype=np.float64)
    if not isinstance(divisor, np.ndarray):
        if divisor == 0:
            return np.zeros_like(numerator, dtype=np.float64)
        return np.true_divide(numerator, divisor, dtype=np.float64)
    if not isinstance(numerator, np.ndarray):
        return np.true_divide(
            numerator,
            divisor,
            out=np.zeros_like(divisor),
            where=divisor > 0,
            dtype=np.float64,
        )
    type_numerator = type(numerator)
    type_divisor = type(divisor)
    raise TypeError(
        f"Input type not supported by safe_divide (numerator is {type_numerator} and divisor is {type_divisor}"
    )
