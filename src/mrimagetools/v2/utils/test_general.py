# type:ignore
# TODO: remove the above line and fix typing errors
""" general.py tests """
from typing import Any, Dict, Final

import numpy as np
import numpy.testing
import pytest
from numpy.random import default_rng

from mrimagetools.v2.utils.general import (
    SnakeCamelConvertType,
    camel_to_snake,
    camel_to_snake_case_keys_converter,
    generate_random_numbers,
    map_dict,
    snake_to_camel,
)


@pytest.fixture(name="input_dict")
def fixture_input_dict() -> dict:
    """A test dictionary"""
    return {
        "one": "two",
        "three": "four",
        "five": "six",
        "seven": "eight",
        "nine": "ten",
    }


def test_map_dict(input_dict: dict) -> None:
    """Perform a simple dictionary mapping"""
    assert map_dict(
        input_dict=input_dict, io_map={"one": "one_hundred", "five": "five_hundred"}
    ) == {"one_hundred": "two", "five_hundred": "six"}


def test_map_dict_raises_keyerror(input_dict: dict) -> None:
    """Perform a simple dictionary mapping with a missing input dictionary key.
    Check a KeyError is raised"""
    with pytest.raises(KeyError):
        _ = map_dict(
            input_dict=input_dict,
            io_map={"doesnotexist": "one_hundred", "five": "five_hundred"},
        ) == {"one_hundred": "two", "five_hundred": "six"}


def test_map_dict_with_optional(input_dict: dict) -> None:
    """Perform a simple dictionary mapping with a missing input dictionary key,
    and optional flag set True. Check a KeyError is not raised and the correct
    output is created, excluding the io_map which does not exist."""
    assert map_dict(
        input_dict=input_dict,
        io_map={
            "doesnotexist": "one_hundred",
            "five": "five_hundred",
            "nine": "nine_hundred",
        },
        io_map_optional=True,
    ) == {"five_hundred": "six", "nine_hundred": "ten"}


def test_generate_random_numbers() -> None:
    """Checks that generate_random_numbers returns correct values"""
    seed = 12345
    shape_1d = (10,)
    shape_2d = (3, 6)
    shape_3d = (7, 4, 9)

    # test normal distributions
    spec = {"distribution": "gaussian", "mean": 1000.0, "sd": 10.0}
    for shape in [None, shape_1d, shape_2d, shape_3d]:
        rng = default_rng(seed=seed)
        x = rng.normal(1000, 10.0, size=shape)
        y = generate_random_numbers(spec, shape=shape, seed=seed)
        numpy.testing.assert_equal(x, y)

    # test uniform distributions
    spec = {"distribution": "uniform", "max": 150.0, "min": 50.0}
    for shape in [None, shape_1d, shape_2d, shape_3d]:
        rng = default_rng(seed=seed)
        x = rng.uniform(50.0, 150.0, size=shape)
        y = generate_random_numbers(spec, shape=shape, seed=seed)
        numpy.testing.assert_equal(x, y)

    # check that if no specification is supplied then an array of zeros is
    # generated
    # test normal distributions
    spec = {}
    for shape in [shape_1d, shape_2d, shape_3d]:
        x = np.zeros(shape)
        y = generate_random_numbers(spec, shape=shape, seed=seed)
        numpy.testing.assert_equal(x, y)

    # check that errors occur if the specification keywords are missing
    spec = {"distribution": "gaussian"}
    with pytest.raises(KeyError):
        generate_random_numbers(spec)

    spec = {"distribution": "uniform"}
    with pytest.raises(KeyError):
        generate_random_numbers(spec)


def test_camel_to_snake_func() -> None:
    """Test the string camel to snake case functionality"""
    assert camel_to_snake("LabelingDuration") == "labeling_duration"


def test_snake_to_camel() -> None:
    """Test the string camel to snake case functionality"""
    # Shouldn't change
    assert snake_to_camel("labeling_duration") == "LabelingDuration"


def test_camel_to_snake_case_keys() -> None:
    """Check the camel to snake case converter"""
    input_dict: Final[dict[str, Any]] = {
        "CamelCase": "CamelCase",
        "AnotherDict": {
            "FooBar1": 1,
            "foo_bar2": 2,
            "FinalNestingHere": {"Camel": 3, "not_a_camel": 4},
        },
        "snake_case": 3,
    }

    assert camel_to_snake_case_keys_converter(input_dict) == {
        "camel_case": "CamelCase",
        "another_dict": {
            "foo_bar1": 1,
            "foo_bar2": 2,
            "final_nesting_here": {"camel": 3, "not_a_camel": 4},
        },
        "snake_case": 3,
    }


def test_snake_to_camel_case_keys() -> None:
    """Check the snake to camel case converter"""

    assert camel_to_snake_case_keys_converter(
        {
            "camel_case": "CamelCase",
            "another_dict": {
                "foo_bar1": 1,
                "foo_bar2": 2,
                "final_nesting_here": {"camel": 3, "not_a_camel": 4},
            },
            "snake_case": 3,
        },
        SnakeCamelConvertType.SNAKE_TO_CAMEL,
    ) == {
        "CamelCase": "CamelCase",
        "AnotherDict": {
            "FooBar1": 1,
            "FooBar2": 2,
            "FinalNestingHere": {"Camel": 3, "NotACamel": 4},
        },
        "SnakeCase": 3,
    }


def test_camel_to_snake_only_keys() -> None:
    """Ensures that only keys (not values) are converted from camel to snake case"""
    assert camel_to_snake_case_keys_converter(
        {
            "CamelDict": {"FooBar": "FooBar"},
            "CamelList": ["UpperOne", "lower_one"],
            "CamelTuple": ("FooBar", "UpperThree", "foo_baz"),
        }
    ) == {
        "camel_dict": {"foo_bar": "FooBar"},
        "camel_list": ["UpperOne", "lower_one"],
        "camel_tuple": ("FooBar", "UpperThree", "foo_baz"),
    }


def test_snake_to_camel_only_keys() -> None:
    """Ensures that only keys (not values) are converted from snake to camel case"""
    assert camel_to_snake_case_keys_converter(
        {
            "camel_dict": {"foo_bar": "FooBar"},
            "camel_list": ["UpperOne", "lower_one"],
            "camel_tuple": ("FooBar", "UpperThree", "foo_baz"),
        },
        convert_type=SnakeCamelConvertType.SNAKE_TO_CAMEL,
    ) == {
        "CamelDict": {"FooBar": "FooBar"},
        "CamelList": ["UpperOne", "lower_one"],
        "CamelTuple": ("FooBar", "UpperThree", "foo_baz"),
    }
