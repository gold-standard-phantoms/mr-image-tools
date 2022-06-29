import pytest

from mrimagetools.validators.checkers import is_a_unit
from mrimagetools.validators.fields import UnitField


def test_units() -> None:
    """Test the unit validator"""
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
