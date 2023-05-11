"""Tests for checkers.py"""

from mrimagetools.validators.checkers import is_a_unit


def test_is_a_unit() -> None:
    """Test that correct unit strings are identified as such"""
    for unit in [
        "meters",
        "m",
        "nm^2/bar",
        "mm**2*s**-1",
        "Nm",
        "gigapascals/kilometer",
        "",  # dimensionless allowed
    ]:
        assert is_a_unit(unit)


def test_incorrect_units() -> None:
    """Test that string that are not units are identified as such"""
    for unit in ["loafofbread", "meeters", "lightspace"]:
        assert not is_a_unit(unit)


def test_invalid_unit_strings() -> None:
    """Test that bad inputs are identified as not units"""
    for unit in ["-", "s-1", 1, Exception()]:
        print(f"testing {unit}")
        assert not is_a_unit(unit)  # type: ignore
