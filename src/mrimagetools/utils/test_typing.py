"""typing tests"""

from typing import Any

import pytest

from mrimagetools.utils.typing import typed


def test_typed() -> None:
    a: Any = 5.0
    v: float = typed(a, float)


def test_typed_fail() -> None:
    a: Any = "a_string"
    with pytest.raises(TypeError):
        v: float = typed(a, float)
