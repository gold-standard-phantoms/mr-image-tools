"""Tests for fields.py"""
from typing import Any

import pytest

from mrimagetools.validators.checkers import is_a_unit
from mrimagetools.validators.fields import NiftiDataTypeField, UnitField


def test_units() -> None:
    """Test the unit validator"""
    unit: Any
    for unit in ["m", "kilometer/hour", "m^2/s**-1", "newton*meters"]:
        assert is_a_unit(unit)
        UnitField.validate(unit)  # no exception raised

    # bad type
    for unit in [1, Exception(), str]:
        assert not is_a_unit(unit)
        with pytest.raises(TypeError):
            UnitField.validate(unit)

    # bad value
    for unit in ["foobar", "foo"]:
        assert not is_a_unit(unit)
        with pytest.raises(ValueError):
            UnitField.validate(unit)


def test_nifti_data_type_field() -> None:
    """Test the nifti data type field"""
    data_type: Any
    for data_type in ["int8", "float32", "complex128"]:
        NiftiDataTypeField.validate(data_type)  # no exception raised

    # bad type
    for data_type in [1, Exception(), str]:
        with pytest.raises(TypeError):
            NiftiDataTypeField.validate(data_type)

    # bad value
    for data_type in ["foobar", "foo"]:
        with pytest.raises(ValueError):
            NiftiDataTypeField.validate(data_type)


def test_get_nifti_data_type_codes() -> None:
    """Check that we can return the integer codes for NIFTI data type strings"""
    for data_type in [("none", 0), ("float32", 16), ("uint64", 1280)]:
        assert NiftiDataTypeField(data_type[0]).type_code == data_type[1]
