"""Contains parameters for performing multiplication operation on a filter."""

from pydantic import BaseModel


class MulParam(BaseModel):
    """Contains parameters for performing multiplication operation on a filter."""

    multiplier: float = 1.0
    """Multiplier to apply to the input value."""
