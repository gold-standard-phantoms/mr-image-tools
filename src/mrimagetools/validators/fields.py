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
            pattern=r"^[A-Za-z0-9-\/*^]*$",
            examples=[
                "ml/100g/min",
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

NIFTI_DATATYPES: Final[Tuple[NiftiDataType, ...]] = typing.get_args(NiftiDataType)
# nifti datatypes (adapted from nibabel.nifti1._dtdefs)
NIFTI_DATATYPE_MAP: Final[Dict[NiftiDataType, int]] = {
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


class NiftiDataTypeField(str):
    """A nifti data type. Must be one of :attr:`NiftiDataType`."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(
            # the full list of allowed types
            examples=list(NIFTI_DATATYPES)
        )

    @classmethod
    def validate(cls, value: Any) -> "NiftiDataTypeField":
        """Validate the NIFTI data type"""
        if not isinstance(value, str):
            raise TypeError(f"String required and a {type(value)} was supplied.")

        if value not in NIFTI_DATATYPES:
            raise ValueError(
                f"{value} is an invalid NIFTI datatype. Options are: {NIFTI_DATATYPES}"
            )
        return cls(value)

    @property
    def type_code(self) -> int:
        """Return the associated NIFTI data type code"""
        return NIFTI_DATATYPE_MAP[self]  # type: ignore
