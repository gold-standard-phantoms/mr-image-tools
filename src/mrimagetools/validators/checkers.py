"""Contains checkers that will return a True or False value based on the input"""

import pint


def is_a_unit(unit: str) -> bool:
    """Uses 'pint' to determine whether the given string is
    a valid representation of an SI unit. For example:
    - ml/100g/min
    - meters
    - m
    - mm**2*s-1
    - Nm
    - gigapascals/kilometers**3
    are all valid and would return True"""

    if not isinstance(unit, str):
        return False
    # allow the empty unit
    if not unit:
        return True
    try:
        pint.Quantity(unit)
    except Exception:  # pylint: disable=broad-except
        return False
    return True
