"""Parameter model classes and function.
Used for parameter validation and typing in the filters"""
from typing import Any, Callable, Dict, Type, Union
from unittest.mock import Base

# Reexport some pydantic imports so they can be imported from this module
# pylint: disable=unused-import
from pydantic import (
    BaseConfig,
    BaseModel,
    Field,
    ValidationError,
    root_validator,
    validator,
)
from pydantic.generics import GenericModel
from pydantic.typing import AnyCallable

from mrimagetools.filters.basefilter import FilterInputValidationError


class ModelMixin:
    """Used for the filter parameters.
    Ensures that raised ValueError, TypeError, and AssertionError
    are caught and raised as FilterInputValidationError.
    Also, arbitrary object types are checked with `isinstance`,
    so any type can be used in the models.
    """

    def __init__(self, **data: Any) -> None:
        try:
            if data is None:
                super().__init__()
            else:
                super().__init__(**data)
        except ValidationError as error:
            raise FilterInputValidationError from error

    class Config(BaseConfig):
        """Configuration options for pydantic"""

        # Currently empty, but can contain pair, for example:
        # BaseImageContainer : str
        # where the left is a Type, and the right is a function for encoding (e.g. str)
        # If a type is not in this dictionary, the __repr__ or __str__ function is called
        json_encoders: Dict[Union[Type[Any], str], AnyCallable] = {}

        # Allow non-json types to be processed with a ParameterModel
        arbitrary_types_allowed = True


class ParameterModel(ModelMixin, BaseModel):
    """Derived from pydantic.BaseModel, using the ModelMixin"""

    class Config(ModelMixin.Config):
        """Use the general config"""


class GenericParameterModel(ModelMixin, GenericModel):
    """Derived from pydantic.generics.GenericModel, using the ModelMixin"""

    class Config(ModelMixin.Config):
        """Use the general config"""


def validate_field(
    field_name: str,
) -> Callable[[Callable[..., Any]], classmethod]:
    """Wraps validator with the set parameters allow_reuse=True"""
    return validator(field_name, allow_reuse=True)
