"""Parameter model classes and function.
Used for parameter validation and typing in the filters"""
from typing import Any, Callable, Dict

# Reexport some pydantic imports so they can be imported from this module
# pylint: disable=unused-import
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

from mrimagetools.filters.basefilter import FilterInputValidationError


class ParameterModel(BaseModel):
    """Used for the filter parameters.
    Ensures that raised ValueError, TypeError, and AssertionError
    are caught and raised as FilterInputValidationError.
    Also, arbitrary object types are checked with `isinstance`,
    so any type can be used in the models.
    """

    def __init__(__pydantic_self__, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as error:
            raise FilterInputValidationError from error

    class Config:
        """Configuration options for pydantic"""

        # Currently empty, but can contain pair, for example:
        # BaseImageContainer : str
        # where the left is a Type, and the right is a function for encoding (e.g. str)
        # If a type is not in this dictionary, the __repr__ or __str__ function is called
        json_encoders: Dict[type, Callable[[Any], str]] = {}

        # Allow non-json types to be processed with a ParameterModel
        arbitrary_types_allowed = True


def validate_field(
    field_name: str,
) -> Callable[[Callable[..., Any]], classmethod]:
    """Wraps validator with the set parameters allow_reuse=True"""
    return validator(field_name, allow_reuse=True)
