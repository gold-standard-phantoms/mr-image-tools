"""Tests for maths.py"""
import cmath

import numpy as np
import numpy.testing as nptesting
import pytest

from mrimagetools.utils.maths import (
    BadVariableTypeError,
    ExpressionSyntaxError,
    OperatorNotSupportedError,
    UnsupportedNodeError,
    VariableMissingError,
    expression_evaluator,
)


def test_nested_expression() -> None:
    """Test a more complex, nested expression to determine if it can
    be properly evaluated"""
    nptesting.assert_almost_equal(
        expression_evaluator()(
            expression="-(A+B**2.0)/C", A=4.0, B=np.array([2.0, -2.0]), C=-5.0
        ),
        np.array([1.6, 1.6]),
    )


def test_unary_subtract() -> None:
    """Test the unary subtract expression"""
    # Integers
    assert expression_evaluator()(expression="-A", A=4) == -4
    assert expression_evaluator()(expression="-5") == -5

    # Floats
    assert expression_evaluator()(expression="-A", A=-2.5) == 2.5
    assert expression_evaluator()(expression="-5.1") == -5.1

    # Complex
    assert expression_evaluator()(expression="-A", A=4 + 3j) == -4 - 3j
    assert expression_evaluator()(expression="-(1+2j)") == -1 - 2j

    # Numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(expression="-A", A=np.array([5.0, 4.0])),
        np.array([-5.0, -4.0]),
    )

    # Complex numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(expression="-A", A=np.array([1.0 + 1j, 2 + 2j])),
        np.array([-1.0 - 1j, -2 - 2j]),
    )


def test_addition() -> None:
    """Test the addition expression"""
    # Integers
    assert expression_evaluator()(expression="A+B", A=4, B=5) == 9
    assert expression_evaluator()(expression="2+5") == 7

    # Floats
    assert expression_evaluator()(expression="A+B", A=-2.5, B=5) == 2.5
    assert expression_evaluator()(expression="A+5", A=2.5) == 7.5

    # Complex
    assert expression_evaluator()(expression="A+B", A=4 + 4j, B=1 + 2j) == 5 + 6j
    assert expression_evaluator()(expression="A+5", A=4 + 4j) == 9 + 4j

    # Numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(
            expression="A+B", A=np.array([5.0, 4.0]), B=np.array([1.0, 3.0])
        ),
        np.array([6.0, 7.0]),
    )
    nptesting.assert_equal(
        expression_evaluator()(expression="A+5.5", A=np.array([5.0, 4.0])),
        np.array([10.5, 9.5]),
    )

    # Complex numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(expression="A+1+2j", A=np.array([1.0 + 1j, 2 + 2j])),
        np.array([2 + 3j, 3 + 4j]),
    )


def test_subtraction() -> None:
    """Test the subtraction expression"""
    # Integers
    assert expression_evaluator()(expression="A-B", A=4, B=5) == -1
    assert expression_evaluator()(expression="2-5") == -3

    # Floats
    assert expression_evaluator()(expression="A-B", A=-2.5, B=5) == -7.5
    assert expression_evaluator()(expression="A-5", A=2.5) == -2.5

    # Complex
    assert expression_evaluator()(expression="A-B", A=4 + 4j, B=1 + 2j) == 3 + 2j
    assert expression_evaluator()(expression="A-5", A=4 + 4j) == -1 + 4j

    # Numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(
            expression="A-B", A=np.array([5.0, 4.0]), B=np.array([1.0, 3.0])
        ),
        np.array([4.0, 1.0]),
    )
    nptesting.assert_equal(
        expression_evaluator()(expression="A-5.5", A=np.array([5.0, 4.0])),
        np.array([-0.5, -1.5]),
    )

    # Complex numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(expression="A-(1+2j)", A=np.array([1.0 + 1j, 2 + 2j])),
        np.array([0 - 1j, 1 + 0j]),
    )


def test_multiplication() -> None:
    """Test the multiplication expression"""
    # Integers
    assert expression_evaluator()(expression="A*B", A=4, B=5) == 20
    assert expression_evaluator()(expression="2*5") == 10

    # Floats
    assert expression_evaluator()(expression="A*B", A=-2.5, B=5) == -12.5
    assert expression_evaluator()(expression="A*5", A=2.5) == 12.5

    # Complex
    assert expression_evaluator()(expression="A*B", A=4 + 4j, B=1 + 2j) == -4 + 12j
    assert expression_evaluator()(expression="A*5", A=4 + 4j) == (20 + 20j)

    # Numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(
            expression="A*B", A=np.array([5.0, 4.0]), B=np.array([1.0, 3.0])
        ),
        np.array([5.0, 12.0]),
    )
    nptesting.assert_equal(
        expression_evaluator()(expression="A*5.5", A=np.array([5.0, 4.0])),
        np.array([27.5, 22.0]),
    )

    # Complex numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(expression="A*(1+2j)", A=np.array([1.0 + 1j, 2 + 2j])),
        np.array([-1.0 + 3.0j, -2.0 + 6.0j]),
    )


def test_division() -> None:
    """Test the division expression"""
    # Integers
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/B", A=4, B=5), 0.8
    )
    nptesting.assert_almost_equal(expression_evaluator()(expression="2/5"), 0.4)
    assert expression_evaluator()(expression="2/0") == np.Inf
    assert expression_evaluator("safe_divide")(expression="2/0") == 0.0

    # Floats
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/B", A=-2.5, B=5), -0.5
    )
    nptesting.assert_almost_equal(expression_evaluator()(expression="A/5", A=2.5), 0.5)
    assert expression_evaluator()(expression="A/0.0", A=2.5) == np.Inf
    assert expression_evaluator("safe_divide")(expression="A/0.0", A=2.5) == 0.0

    # Complex
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/B", A=4 + 4j, B=1 + 2j), 2.4 - 0.8j
    )
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/5", A=4 + 4j), 0.8 + 0.8j
    )
    assert expression_evaluator()(expression="A/0", A=4 + 4j) == cmath.inf + cmath.infj

    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/(0+1j)", A=4 + 4j), 4 - 4j
    )
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/(1+0j)", A=4 + 4j), 4 + 4j
    )

    # Numpy arrays
    nptesting.assert_almost_equal(
        expression_evaluator()(
            expression="A/B", A=np.array([5.0, 4.0]), B=np.array([1.0, 3.0])
        ),
        np.array([5.0, 4 / 3]),
    )
    nptesting.assert_almost_equal(
        expression_evaluator()(
            expression="A/B", A=np.array([5.0, 4.0]), B=np.array([1.0, 0.0])
        ),
        np.array([5.0, np.Inf]),
    )
    nptesting.assert_almost_equal(
        expression_evaluator("safe_divide")(
            expression="A/B", A=np.array([5.0, 4.0]), B=np.array([1.0, 0.0])
        ),
        np.array([5.0, 0.0]),
    )
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/5.5", A=np.array([5.0, 4.0])),
        np.array([5.0 / 5.5, 4.0 / 5.5]),
    )

    # Complex numpy arrays
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/(1+2j)", A=np.array([1.0 + 1j, 2 + 2j])),
        np.array([1.0 + 1j, 2 + 2j]) / (1 + 2j),
    )
    # Complex numpy arrays
    nptesting.assert_almost_equal(
        expression_evaluator()(expression="A/(0+0j)", A=np.array([1.0 + 2j])),
        np.array([cmath.inf + cmath.infj]),
    )
    nptesting.assert_almost_equal(
        expression_evaluator("safe_divide")(
            expression="A/(0+0j)", A=np.array([1.0 + 2j])
        ),
        np.array([0.0 + 0.0j]),
    )


def test_power() -> None:
    """Test the power expression"""
    # Integers
    assert expression_evaluator()(expression="A**B", A=4, B=5) == 4**5
    assert expression_evaluator()(expression="2**5") == 2**5

    # Floats
    assert expression_evaluator()(expression="A**B", A=-2.5, B=5) == (-2.5) ** 5
    assert expression_evaluator()(expression="A**5", A=2.5) == 2.5**5

    # Complex
    assert expression_evaluator()(expression="A**B", A=4 + 4j, B=1 + 2j) == (
        4 + 4j
    ) ** (1 + 2j)
    assert expression_evaluator()(expression="A**5", A=4 + 4j) == (4 + 4j) ** 5

    # Numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(
            expression="A**B", A=np.array([5.0, 4.0]), B=np.array([1.0, 3.0])
        ),
        np.array([5.0, 4.0]) ** np.array([1.0, 3.0]),
    )
    nptesting.assert_equal(
        expression_evaluator()(expression="A**5.5", A=np.array([5.0, 4.0])),
        np.array([5.0, 4.0]) ** 5.5,
    )

    # Complex numpy arrays
    nptesting.assert_equal(
        expression_evaluator()(expression="A**(1+2j)", A=np.array([1.0 + 1j, 2 + 2j])),
        np.array([1.0 + 1j, 2 + 2j]) ** (1 + 2j),
    )


def test_validation_operator_not_supported_error() -> None:
    """Test that the correct errors are raised when operators are not supported"""
    with pytest.raises(OperatorNotSupportedError):
        expression_evaluator()(expression="A%B", A=1, B=2)
    with pytest.raises(OperatorNotSupportedError):
        expression_evaluator().validate(expression="A%B", A=1, B=2)
    assert expression_evaluator().is_valid(expression="A%B", A=1, B=2) is False


def test_validation_variable_missing_error() -> None:
    """Test that the correct errors are raised when necessary variables are missing"""
    with pytest.raises(VariableMissingError):
        expression_evaluator()(expression="A+B", A=1)
    with pytest.raises(VariableMissingError):
        expression_evaluator().validate(expression="A+B", A=1)
    assert expression_evaluator().is_valid(expression="A+B", A=1) is False


def test_validation_bad_variable_type_error() -> None:
    """Test that the correct errors are raised a variable is unsupported unsupported type"""
    with pytest.raises(BadVariableTypeError):
        expression_evaluator()(expression="A+B", A=1, B="foo")  # type:ignore
    with pytest.raises(BadVariableTypeError):
        expression_evaluator().validate(expression="A+B", A=1, B="foo")  # type:ignore
    assert (
        expression_evaluator().is_valid(
            expression="A+B", A=1, B="foo"  # type:ignore
        )
        is False
    )


def test_validation_unsupported_node_error() -> None:
    """Test that the correct errors are raised when a node is not recognised
    (functions are not yet implemented)"""
    with pytest.raises(UnsupportedNodeError):
        expression_evaluator()(expression="func(A+B)", A=1, B=2)
    assert expression_evaluator().is_valid(expression="func(A+B)", A=1, B=2) is False


def test_validation_bad_syntax_error() -> None:
    """Test that the correct errors are raised when an expression syntax is bad"""
    with pytest.raises(ExpressionSyntaxError):
        expression_evaluator()(expression="//+2A", A=1)

    assert expression_evaluator().is_valid(expression="//+2A", A=1) is False
