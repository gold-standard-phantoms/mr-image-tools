""" General utilities """
import os
import re
from collections.abc import Mapping
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.random import default_rng


def map_dict(
    input_dict: Mapping[str, Any],
    io_map: Mapping[str, str],
    io_map_optional: bool = False,
) -> dict[str, Any]:
    """
    Maps a dictionary onto a new dictionary by changing some/all of
    the keys.

    :param input_dict: The input dictionary
    :param io_map: The dictionary used to perform the mapping. All keys
      and values must be strings. For example:

        .. code-block:: python

            {
                "one": "two",
                "three": "four"
            }

      will map inputs keys of "one" to "two" AND "three" to "four".
    :param io_map_optional: If this is False, a KeyError will be raised
      if the keys in the io_map are not found in the input_dict.
    :raises KeyError: if keys required in the mapping are not found in the input_dict
    :return: the remapped dictionary
    """
    # Will raise KeyError if key from io_map is missing in input_dict but
    # only if io_map_optional is False
    return {
        map_to: input_dict[map_from]
        for map_from, map_to in io_map.items()
        if (not io_map_optional) or map_from in input_dict
    }


def splitext(path: str) -> tuple[str, str]:
    """The normal os.path.splitext treats path/example.tar.gz
    as having a filepath of path/example.tar with a .gz
    extension - this fixes it"""
    for ext in [".tar.gz", ".tar.bz2", ".nii.gz"]:
        if path.lower().endswith(ext.lower()):
            return path[: -len(ext)], path[-len(ext) :]
    return os.path.splitext(path)


def generate_random_numbers(
    specification: dict, shape: Union[tuple, None] = None, seed: Optional[int] = None
) -> np.ndarray:
    """Generates a set of numbers according to the prescribed distributions,
    returning them as a list

    :param specification: The distribution to use to generate the parameters:

      * 'gaussian' - normal distribution with mean and standard deviation
      * 'uniform' - rectangular distribution with min and max values
      * 'mean' - mean value of the distribution (gaussian only)
      * 'sd' - standard deviation of the distribution (gaussian only)
      * 'min' - minimum value of the distribution (uniform only)
      * 'max' - maximum value of the distribution (uniform only)

    :type specification: dict
    :param shape: length of the list to return
    :type shape: int or tuple of ints
    :param seed: The random number generator to use, defaults to None
    :type seed: int, optional
    :return: List of the generated numbers
    :rtype: list
    """
    rng = default_rng(seed)
    out: np.ndarray
    if specification.get("distribution") is not None:
        distribution = specification["distribution"]
        if distribution == "gaussian":
            out = rng.normal(specification["mean"], specification["sd"], shape)
        elif distribution == "uniform":
            out = rng.uniform(specification["min"], specification["max"], shape)
        else:
            raise ValueError(f"Distribution {distribution} is not supported")
    else:
        if shape is None:
            raise ValueError("Distribution AND shape have not been specified")
        return np.zeros(shape)

    return out


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case"""
    if not isinstance(name, str):
        raise ValueError("Must be a string")
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    if not isinstance(snake_case, str):
        raise ValueError("Must be a string")
    return snake_case


def snake_to_camel(name: str) -> str:
    """Converts a snake_case string to (upper) CamelCase"""
    return "".join(word.title() for word in name.replace("__", "_").split("_"))


T = TypeVar("T")  # pylint: disable=invalid-name

InputType = Union[T, dict[str, T], tuple[T], list[T]]


class SnakeCamelConvertType(Enum):
    """Describes a conversion between types"""

    SNAKE_TO_CAMEL = auto()
    CAMEL_TO_SNAKE = auto()


def key_value_converter(
    convert_type: SnakeCamelConvertType,
) -> tuple[Callable[[str], str], Callable[[InputType], InputType]]:
    """Get functions for converting keys and values as per the conversion schema."""
    if convert_type == SnakeCamelConvertType.CAMEL_TO_SNAKE:
        return (
            camel_to_snake,
            lambda value: camel_to_snake_case_keys_recursive(value, convert_type),
        )
    if convert_type == SnakeCamelConvertType.SNAKE_TO_CAMEL:
        return (
            snake_to_camel,
            lambda value: camel_to_snake_case_keys_recursive(value, convert_type),
        )
    raise ValueError("Convert type {convert_type} cannot be processed")


def camel_to_snake_case_keys_recursive(
    input_value: InputType, convert_type: SnakeCamelConvertType
) -> InputType:
    """Converts a dict recursively, to ensure that all keys are in correct case"""
    key_conv, value_conv = key_value_converter(convert_type)
    if isinstance(input_value, dict):
        return_dict = {}
        for key, value in input_value.items():
            return_dict[key_conv(key)] = value_conv(value)
        return return_dict
    if isinstance(input_value, tuple):
        return tuple(value_conv(v) for v in input_value)
    if isinstance(input_value, list):
        return [value_conv(v) for v in input_value]
    return input_value


def camel_to_snake_case_keys_converter(
    input_value: dict[str, Any],
    convert_type: SnakeCamelConvertType = SnakeCamelConvertType.CAMEL_TO_SNAKE,
) -> dict[str, Any]:
    """Converts a dictionary where the keys are CamelCase
    into snake_case and vice-versa"""
    key_conv, value_conv = key_value_converter(convert_type)
    return_dict = {}
    for key, value in input_value.items():
        return_dict[key_conv(key)] = value_conv(value)
    return return_dict
