"""Parameter model classes and function.
Used for parameter validation and typing in the filters"""
from typing import Any, Callable, Dict, Type, Union

# Reexport some pydantic imports so they can be imported from this module
# pylint: disable=unused-import
from pydantic import (
    BaseConfig,
    BaseModel,
    Extra,
    Field,
    ValidationError,
    root_validator,
    validator,
)
from pydantic.generics import GenericModel
from pydantic.typing import AnyCallable
from pydantic.utils import Representation

from mrimagetools.filters.basefilter import FilterInputValidationError


class ModelMixin(Representation):
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
            raise FilterInputValidationError(str(error)) from error

    class Config(BaseConfig):
        """Configuration options for pydantic"""

        # Allow arbitrary, non-json-like types to be processed be the JSON schema generation
        arbitrary_types_allowed = True
        # Does not allow extra (undefined) attributes to be added to a model
        extra = Extra.forbid

    def __repr_str__(self, join_str: str) -> str:
        """A string represenation of the model. Excludes any attributes that are None.
        :param join_str: the string used between attribute names/values."""
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in self.__repr_args__()
            if v is not None  # don't output anything if the value is None
        )

    def __str__(self) -> str:
        """The string representation of the model attributes"""
        return self.__repr_str__(" ")

    def __repr__(self) -> str:
        """The string representation of the model"""
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'


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
