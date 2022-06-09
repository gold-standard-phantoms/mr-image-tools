"""safe_divide.py tests"""
import numpy as np
import pytest

from mrimagetools.utils.safe_divide import safe_divide


@pytest.fixture()
def fixture_input_num() -> np.ndarray:
    """A test numerator"""
    return np.array([1, 2, 3, 4], dtype=np.float32)


@pytest.fixture()
def fixture_input_num_complex() -> np.ndarray:
    """A test complex numerator"""
    return np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)


@pytest.fixture()
def fixture_input_0_num() -> np.ndarray:
    """A test 0 numerator"""
    return np.array([0, 0, 0, 0], dtype=np.float32)


@pytest.fixture()
def fixture_input_div() -> np.ndarray:
    """A test divisor"""
    return np.array([1, 2, 3, 4], dtype=np.float32)


@pytest.fixture()
def fixture_input_div_complex() -> np.ndarray:
    """A test complex divisor"""
    return np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)


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


def test_div_int(fixture_input_num: np.ndarray):
    """test numpy array division by int"""
    assert np.array_equal(safe_divide(fixture_input_num, 2), [1 / 2, 1, 3 / 2, 2])


def test_num_int(fixture_input_div: np.ndarray):
    """test numpy array division of int by array"""
    assert np.array_equal(safe_divide(12, fixture_input_div), [12, 6, 4, 3])


def test_2_int():
    """test the division of two int"""
    assert safe_divide(8, 2) == 4


def test_complex_div_comple_num(
    fixture_input_num_complex: np.ndarray, fixture_input_div_complex: np.ndarray
):
    """test the division by a complex array with complex numerator"""
    assert np.array_equal(
        safe_divide(fixture_input_num_complex, fixture_input_div_complex),
        fixture_input_num_complex / fixture_input_div_complex,
    )


def test_complex_div(
    fixture_input_num: np.ndarray, fixture_input_div_complex: np.ndarray
):
    """test the division by a complex array with int array numerator"""
    assert np.array_equal(
        safe_divide(fixture_input_num, fixture_input_div_complex),
        fixture_input_num / fixture_input_div_complex,
    )


def test_complex_num(
    fixture_input_num_complex: np.ndarray, fixture_input_div: np.ndarray
):
    """test the division with a complex array numerator and int array divisor"""
    assert np.array_equal(
        safe_divide(fixture_input_num_complex, fixture_input_div),
        fixture_input_num_complex / fixture_input_div,
    )


def test_complex_by_0_stays_complex():
    """test that a complex number divided by 0 stays complex even if still 0"""
    assert safe_divide(4 + 4j, 0) == 0j


def test_wrong_type():
    """test to see if the TypeError is raised if not an InputType is inputed"""
    with pytest.raises(TypeError):
        safe_divide("wrong_input", 1)
