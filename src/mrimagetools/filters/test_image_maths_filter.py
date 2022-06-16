"""MathsFilter tests"""
from typing import Tuple, Union

import numpy as np
import numpy.testing as nptesting
import pytest

from mrimagetools.containers.image import BaseImageContainer, NumpyImageContainer
from mrimagetools.filters.basefilter import FilterInputValidationError

from .maths_filter import MathsFilter


def test_simple_expression() -> None:
    """Test a simple expression"""
    image: BaseImageContainer = generate_arange_numpy_image_container(shape=(2, 2, 2))
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "A+1")
    maths_filter.add_input("A", image)
    maths_filter.run()  # should run
    result: BaseImageContainer = maths_filter.outputs["result"]
    nptesting.assert_array_almost_equal(result.image, image.image + 1)


def test_complex_expression() -> None:
    """Test a complex expression"""
    image_a: BaseImageContainer = generate_arange_numpy_image_container(shape=(2, 2, 2))
    image_b: BaseImageContainer = generate_arange_numpy_image_container(
        shape=(2, 2, 2), start=10
    )
    image_c: BaseImageContainer = generate_arange_numpy_image_container(
        shape=(2, 2, 2), start=20
    )
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "-(B*A)/(C+5)")
    maths_filter.add_input("A", image_a)
    maths_filter.add_input("B", image_b)
    maths_filter.add_input("C", image_c)
    maths_filter.run()  # should run
    result: BaseImageContainer = maths_filter.outputs["result"]
    nptesting.assert_array_almost_equal(
        result.image, 0 - (image_b.image * image_a.image) / (image_c.image + 5)
    )
    nptesting.assert_array_equal(result.affine, image_a.affine)


def test_safe_divide() -> None:
    """Test an expression using safe_divide"""
    image_a: BaseImageContainer = generate_arange_numpy_image_container(shape=(2, 2, 2))
    image_b: BaseImageContainer = generate_arange_numpy_image_container(
        shape=(2, 2, 2), start=10
    )
    maths_filter = MathsFilter(config="safe_divide")
    maths_filter.add_input("expression", "A/B")
    maths_filter.add_input("A", image_a)
    maths_filter.add_input("B", image_b)
    maths_filter.run()  # should run
    result: BaseImageContainer = maths_filter.outputs["result"]
    # Check that a regular divisor occurred - except in the case of divide by zero,
    # where the result is zero
    nptesting.assert_array_almost_equal(
        result.image,
        np.true_divide(
            image_a.image,
            image_b.image,
            out=np.zeros(shape=image_a.shape),
            where=image_b.image == 0,
        ),
    )


def test_expression_is_string() -> None:
    """Test that when expression is a valid string, filter runs without error"""
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "A+1")
    maths_filter.add_input("A", 1)
    maths_filter.run()  # should run
    assert maths_filter.outputs["result"] == 2


def test_expression_is_not_string() -> None:
    """Test that when expression is not a valid string, filter validation error"""
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", 1)
    maths_filter.add_input("A", 1)
    with pytest.raises(FilterInputValidationError):
        maths_filter.run()


def test_bad_expression_syntax_validation() -> None:
    """Test that bad expression syntax raises a filter validation error"""
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "/A")
    maths_filter.add_input("A", 1)
    with pytest.raises(FilterInputValidationError):
        maths_filter.run()


def test_bad_expression_validation() -> None:
    """Test that bad expressions raise a filter validation error"""
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "Fun(A)")  # functions not supported (yet!)
    maths_filter.add_input("A", 1)
    with pytest.raises(FilterInputValidationError):
        maths_filter.run()


def test_missing_parameters_validation() -> None:
    """Test that missing parameters raise a filter validation error"""
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "A+B")
    maths_filter.add_input("A", 1)

    with pytest.raises(FilterInputValidationError):
        maths_filter.run()


def generate_arange_numpy_image_container(
    shape: Tuple[int, ...], start: int = 0, affine: Union[np.ndarray, None] = None
) -> NumpyImageContainer:
    """Return a numpy image container with the given shape.
    Values of the array start from 0, and increase by one each step to
    fill the image, writing the elements using C-like index order"""
    if affine is None:
        affine = np.eye(4)
    return NumpyImageContainer(
        image=np.arange(start=start, stop=start + np.prod(shape)).reshape(shape),
        affine=affine,
    )


def test_generate_numpy_image_container() -> None:
    """Test the numpy image generation"""
    image = generate_arange_numpy_image_container(shape=(2, 3, 4), affine=np.eye(4) * 3)
    assert image.shape == (2, 3, 4)
    assert image.image[0, 0, 0] == 0
    assert image.image[0, 0, 1] == 1
    assert image.image[1, 2, 3] == (2 * 3 * 4) - 1
    nptesting.assert_array_equal(image.affine, np.eye(4) * 3)


def test_images_match_shape() -> None:
    """Test that mismatching image shapes throws a validation error"""
    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "A+B")
    maths_filter.add_input("A", generate_arange_numpy_image_container(shape=(2, 3)))
    maths_filter.add_input("B", generate_arange_numpy_image_container(shape=(2, 4)))

    with pytest.raises(FilterInputValidationError):
        maths_filter.run()


def test_images_match_affine() -> None:
    """Test that mismatching affine matrices in the inputs throws a validation error"""

    maths_filter = MathsFilter()
    maths_filter.add_input("expression", "A+B")
    maths_filter.add_input(
        "A", generate_arange_numpy_image_container(shape=(2, 3), affine=np.eye(4))
    )
    maths_filter.add_input(
        "B",
        generate_arange_numpy_image_container(shape=(2, 3), affine=np.zeros((4, 4))),
    )

    with pytest.raises(FilterInputValidationError):
        maths_filter.run()
