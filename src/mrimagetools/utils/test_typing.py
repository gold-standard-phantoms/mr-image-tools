"""typing tests"""

from typing import Any

import pytest

from mrimagetools.utils.typing import typed


def test_typed() -> None:
    """test the typing"""
    a: Any = 5.0
    _: float = typed(a, float)


def test_typed_fail() -> None:
    """test the typing expecting a fail"""
    a: Any = "a_string"
    with pytest.raises(TypeError):
        _: float = typed(a, float)
