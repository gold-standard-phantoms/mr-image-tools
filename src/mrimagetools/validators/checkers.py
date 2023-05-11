"""Contains checkers that will return a True or False value based on the input"""

import logging

from pint import UnitRegistry

_ureg = UnitRegistry()
logger = logging.getLogger(__name__)


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
        logger.warning("Input to is_a_unit is not a string")
        return False
    # allow the empty unit
    if not unit:
        return True
    try:
        _ureg.Quantity(unit)  # type: ignore
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Input %s to is_a_unit is not a valid unit: %s", unit, e)
        return False
    return True
