"""safe_divide.py tests"""
from multiprocessing.sharedctypes import Value

import numpy as np
import pytest

from mrimagetools.utils.safe_divide import safe_divide


@pytest.fixture()
def fixture_input_num() -> np.ndarray:
    """A test numerator"""
    return np.array([1, 2, 3, 4], dtype=np.float32)


@pytest.fixture()
def fixture_input_0_num() -> np.ndarray:
    """A test 0 numerator"""
    return np.array([0, 0, 0, 0], dtype=np.float32)


@pytest.fixture()
def fixture_input_div() -> np.ndarray:
    """A test divisor"""
    return np.array([1, 2, 3, 4], dtype=np.float32)


@pytest.fixture()
def fixture_input_0_div() -> np.ndarray:
    """A test 0 divisor"""
    return np.array([0, 0, 0, 0], dtype=np.float32)


def test_simple_div(fixture_input_num: np.ndarray, fixture_input_div: np.ndarray):
    """test normal div"""
    assert np.array_equal(
        safe_divide(fixture_input_num, fixture_input_div), [1.0, 1.0, 1.0, 1.0]
    )


def test_simple_0_div(fixture_input_num: np.ndarray, fixture_input_0_div: np.ndarray):
    """test x/0 div"""
    assert np.array_equal(
        safe_divide(fixture_input_num, fixture_input_0_div), [0, 0, 0, 0]
    )


def test_both_0_div(fixture_input_0_num: np.ndarray, fixture_input_0_div: np.ndarray):
    """test 0/0 div"""
    assert np.array_equal(
        safe_divide(fixture_input_0_num, fixture_input_0_div), [0, 0, 0, 0]
    )


def test_num_0_div(fixture_input_0_num: np.ndarray, fixture_input_div: np.ndarray):
    """test 0/x div"""
    assert np.array_equal(
        safe_divide(fixture_input_0_num, fixture_input_div), [0, 0, 0, 0]
    )


def test_non_broadcastable(fixture_input_div: np.ndarray):
    """test non broadcastable error"""
    with pytest.raises(ValueError):
        safe_divide(np.array([1, 2, 3]), fixture_input_div)
