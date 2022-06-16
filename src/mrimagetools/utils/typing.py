"""Typing utilities"""
from typing import Any, Type, TypeVar

T = TypeVar("T")


def typed(value: Any, input_type: Type[T]) -> T:
    """Get a value (typed), checking that the type is correct.
    (helps with mypy type checking). e.g.:
    a:Any = 5.0
    v:float = typed(a, float)  # v is now typed (or will throw a TypeError)
    :raises: TypeError if the value is not of the given type."""
    if isinstance(value, input_type):
        return value
    raise TypeError(
        f"Value with {value} is of type {type(value)} and should be {input_type}"
    )
