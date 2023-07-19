"""Custom Pydantic fields"""

import typing
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, Final, Literal, Tuple

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    RootModel,
    TypeAdapter,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from mrimagetools.validators.checkers import is_a_unit


# _UnitField = RootModel[str]
class _UnitField(RootModel[str]):
    root: str

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _UnitField):
            return False
        return self.root == other.root


"""A unit. Uses 'pint' to determine whether the given string is
a valid representation of a unit. For example:
- meters
- m
- mm^2*s^-1
- mm**2*s**-1
- Nm
- gigapascals/kilometers**3
are all valid."""


class _UnitFieldPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * strs will be parsed as `_UnitField` instances
        * `_UnitField` instances will be parsed as `_UnitField` instances without any
        changes
        * Nothing else will pass validation
        * Serialization will always return just an str
        """

        def validate_from_str(value: str) -> _UnitField:
            if not is_a_unit(value):
                raise ValueError(f"{str} is not a valid unit")
            return _UnitField(root=value)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(
                    function=validate_from_str
                ),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(_UnitField),
                    from_str_schema,
                ]
            ),
        )


UnitField = Annotated[_UnitField, _UnitFieldPydanticAnnotation]


NiftiDataType = Literal[
    "none",
    "binary",
    "uint8",
    "int16",
    "int32",
    "float32",
    "complex64",
    "float64",
    "RGB",
    "all",
    "int8",
    "uint16",
    "uint32",
    "int64",
    "uint64",
    "float128",
    "complex128",
    "complex256",
    "RGBA",
]
(
    """A NIFTI data type. Corresponds with:
https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/"""
    """nifti1fields_pages/datatype.html"""
)

NIFTI_DATATYPES: Final[tuple[NiftiDataType, ...]] = typing.get_args(NiftiDataType)
# nifti datatypes (adapted from nibabel.nifti1._dtdefs)
NIFTI_DATATYPE_MAP: Final[dict[NiftiDataType, int]] = {
    "none": 0,
    "binary": 1,
    "uint8": 2,
    "int16": 4,
    "int32": 8,
    "float32": 16,
    "complex64": 32,
    "float64": 64,
    "RGB": 128,
    "all": 255,
    "int8": 256,
    "uint16": 512,
    "uint32": 768,
    "int64": 1024,
    "uint64": 1280,
    "float128": 1536,
    "complex128": 1792,
    "complex256": 2048,
    "RGBA": 2304,
}


def type_code(value: NiftiDataType) -> int:
    """Return the associated NIFTI data type code"""
    return NIFTI_DATATYPE_MAP[str(value)]  # type: ignore


class _NiftiDataTypeField(RootModel[str]):
    root: str

    @property
    def type_code(self) -> int:
        """Return the associated NIFTI data type code"""
        return NIFTI_DATATYPE_MAP[self.root]  # type: ignore

    def __hash__(self) -> int:
        return hash(self.root)

    @property
    def value(self) -> str:
        return self.root

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _NiftiDataTypeField):
            return False
        return self.root == other.root

    def __str__(self) -> str:
        return self.root


class _NiftiDataTypePydanticAnnotation:
    """A nifti data type. Must be one of :attr:`NiftiDataType`."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * strs will be parsed as `_NiftiDataTypeField` instances
        * `_NiftiDataTypeField` instances will be parsed as `_NiftiDataTypeField` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return just a str
        """

        def validate_from_str(value: str) -> _NiftiDataTypeField:
            if value not in NIFTI_DATATYPES:
                raise ValueError(
                    f"{value} is an invalid NIFTI datatype. Options are:"
                    f" {NIFTI_DATATYPES}"
                )
            return _NiftiDataTypeField.model_validate(value)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_str),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(_NiftiDataTypeField),
                    from_str_schema,
                ]
            ),
        )


NiftiDataTypeField = Annotated[_NiftiDataTypeField, _NiftiDataTypePydanticAnnotation]
