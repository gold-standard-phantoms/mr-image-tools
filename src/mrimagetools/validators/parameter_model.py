"""Parameter model classes and function.
Used for parameter validation and typing in the filters"""
import sys
import types
from pathlib import PosixPath
from typing import Any, Callable, Dict, ForwardRef, Type, Union

from pydantic import BaseConfig, BaseModel, Extra, FilePath, ValidationError, validator
from pydantic.generics import GenericModel
from pydantic.typing import AnyCallable
from pydantic.utils import Representation

from mrimagetools.filters.basefilter import FilterInputValidationError


def path_encoder(path: Union[PosixPath, FilePath]) -> str:
    """Takes a PosixPath and returns the string represenation"""
    return path.as_posix()


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
            try:
                # Remove this frame from the traceback - it's not very helpful!
                frame = sys._getframe(1)
                this_traceback = types.TracebackType(
                    None, frame, frame.f_lasti, frame.f_lineno
                )
                raise FilterInputValidationError(error).with_traceback(
                    this_traceback
                ) from error
            except AttributeError:
                # catch the case that _getframe is not implemented
                raise FilterInputValidationError(error) from error

    class Config(BaseConfig):
        """Configuration options for pydantic"""

        # Allow arbitrary, non-json-like types to be processed be the JSON schema generation
        arbitrary_types_allowed = True
        # Does not allow extra (undefined) attributes to be added to a model
        extra = Extra.forbid
        # Additional JSON encoders
        json_encoders: Dict[Union[Type[Any], str, ForwardRef], AnyCallable] = {
            PosixPath: path_encoder,
            FilePath: path_encoder,
        }
        # whether to validate field default
        validate_all = True

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
