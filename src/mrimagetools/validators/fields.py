"""Custom Pydantic fields"""

import typing
from typing import Any, Dict, Final, Literal, Tuple

from mrimagetools.validators.checkers import is_a_unit


class UnitField(str):
    """A unit. Uses 'pint' to determine whether the given string is
    a valid representation of a unit. For example:
    - meters
    - m
    - mm^2*s^-1
    - mm**2*s**-1
    - Nm
    - gigapascals/kilometers**3
    are all valid."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(
            pattern=r"^[A-Za-z0-9-\/*^]+$",
            examples=[
                "meters",
                "m",
                "mm^2*s^-1",
                "Nm",
                "gigapascals/kilometers**3",
                "mm**2*s**-1",
            ],
        )

    @classmethod
    def validate(cls, value: Any) -> "UnitField":
        """Validate the unit"""
        if not isinstance(value, str):
            raise TypeError(f"String required and a {type(value)} was supplied.")

        if not is_a_unit(value):
            raise ValueError(f"{value} is an invalid unit.")
        return cls(value)
