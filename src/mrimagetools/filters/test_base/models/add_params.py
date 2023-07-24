"""Contains parameters for performing addition operation on a filter."""

from pydantic import BaseModel


class AddParam(BaseModel):
    """Contains parameters for performing addition operation on a filter."""

    addend: float = 0.0
    """Addend to apply to the input value."""
