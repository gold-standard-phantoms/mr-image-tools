"""Utility functions for testing filters"""
from __future__ import annotations

from collections.abc import Generator
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pytest
from pydantic import Field, ValidationError, model_validator

from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.validators.parameter_model import GenericParameterModel

DataT = TypeVar("DataT")  # a generic datatype
FilterT = TypeVar("FilterT", bound=BaseFilter)  # A generic BaseFilter


class MissingParameter:
    """Used to represent the parameter is not specified"""


class ParameterPermutation:
    """A generic class for a parameter permutation"""


@dataclass
class ExistingParameterPermutation(ParameterPermutation):
    """Used to specify a parameter where the parameter is provided.
    :param parameter: The parameter value"""

    parameter: Any


@dataclass
class ValidParameterPermutation(ExistingParameterPermutation, Generic[DataT]):
    """Used to specify a single parameter permutation.
    :param parameter: The parameter value"""

    parameter: DataT


@dataclass
class InvalidParameterPermutation(ExistingParameterPermutation):
    """Used to specify a single, invalid parameter permutation.
    :param parameter: The parameter value"""

    parameter: Any


@dataclass
class MissingParameterPermutation(ParameterPermutation):
    """Used to specify whether the parameter is valid if it is not specified (optional).
    :param is_valid: If it is valid to have this parameter not specified."""

    is_valid: bool


class FilterValidationModelParameter(GenericParameterModel, Generic[DataT]):
    """A filter parameter validation model. Contains some optional valid and invalid
    examples, and well as defining if the parameter is required. Can be used:
    ```
    validation = FilterParameterValidationModel[str](
        is_optional=True,
        valid_values=['foo', 'bar'],
        invalid_values=[1, 2]
    )
    ```
    :param is_optional: If the parameter is optional
    :param valid_values: A list of valid values
    :param valid_values: A list of invalid values
    :raises: FilterInputValidationError"""

    is_optional: bool = False
    valid_values: list[DataT] = Field(default_factory=list)
    invalid_values: list[Any] = Field(default_factory=list)

    def permutations(self) -> Generator[ParameterPermutation, None, None]:
        """Generate the possible combinations that the parameter might take.
        :returns: A generator for possible parameters (derived from ParameterPermutation)
        """
        # If the parameter is optional, it is valid if missing
        yield MissingParameterPermutation(is_valid=self.is_optional)
        for value in self.valid_values:
            yield ValidParameterPermutation(parameter=value)
        for value in self.invalid_values:
            yield InvalidParameterPermutation(parameter=value)

    @model_validator(mode="after")
    def validated_parameters(self) -> FilterValidationModelParameter:
        """Is "is_optional" is False, we need at least one
        valid value to use in the testing"""
        if not self.is_optional and len(self.valid_values) == 0:
            raise ValueError("valid_values cannot be empty if is_optional is False")
        return self


def _recursive_parameter_solver(
    validation_tuple_list: list[tuple[str, FilterValidationModelParameter]],
    current_parameters: dict[str, ParameterPermutation],
    depth: int = 0,
) -> list[dict[str, ParameterPermutation]]:
    """A recursive function which generated a list of all permutations
    of the parameter space"""
    # If we're at the end of the rescursion (the end of the list), just
    # return the current dictionary in a list
    if depth >= len(validation_tuple_list):
        return [current_parameters]
    key, value = validation_tuple_list[depth]
    permutation_list: list[dict[str, ParameterPermutation]] = []
    # For all of the possible values that the current parameter might take
    for parameter in value.permutations():
        # Copy the parameter space created so far
        new = copy(current_parameters)
        # Add the current parameter
        new[key] = parameter
        # pass the generator forwards (to the next depth)
        permutation_list.extend(
            _recursive_parameter_solver(validation_tuple_list, new, depth + 1)
        )
    return permutation_list


class FilterValidationModel(GenericParameterModel, Generic[FilterT]):
    """A 'dataclass' (derived from a pydantic BaseModel) to assist with the
    validation of filter, by creating and passing all permutations of
    input parameters and checking if appropriate exceptions are raised (or not).

    NOTE: as the complexity of this functionality is O(n^2), by default, the
    filters are validated, but not run.

    Example:

    ```
    FilterValidationModel(
        filter_type=ProductFilter,
        parameters={
            "input1": FilterValidationModelParameter[StrictFloat](
                is_optional=False, valid_values=[10.0], invalid_values=[20, "str"]
            ),
            "input2": FilterValidationModelParameter[StrictFloat](
                is_optional=False, valid_values=[20.0], invalid_values=[20, "str"]
            ),
            "input3": FilterValidationModelParameter[StrictFloat](
                is_optional=True, valid_values=[15.0], invalid_values=[15, "str"]
            ),
        },
    )
    ```
    NOTE: providing the generic (e.g. StrictFloat) to FilterValidationModelParameter
    is optional, and only helps assist with typechecking your tests. It is recommended to
    use the pydantic StrictFloat, StringStr, etc. where relevant.

    :param filter_type: The type of filter to use
    :param filter_arguments: Optional arguments to the filter __init__
    :param parameters: a dictionary of FilterValidationModelParameter. The key is the
    parameter name, and the value can be viewed by referencing the documentation
    for :class:`.FilterValidationModelParameter`.
    :param run_filter: Whether the filter should be run (or just validated). Default False
    :raises FilterInputValidationError: if the validation is True when it should be False,
    or vice-versa.
    """

    filter_type: type[FilterT]
    filter_arguments: dict[str, Any] = Field(default_factory=dict)
    # keys correspond with the parameter name
    parameters: dict[str, FilterValidationModelParameter] = Field(default_factory=dict)
    run_filter: bool = False

    @model_validator(mode="after")
    def validate_model(self) -> FilterValidationModel:
        """Perform the validation on the filter_type with all permutations of inputs."""
        filter_type = self.filter_type
        parameters = self.parameters
        filter_arguments = self.filter_arguments
        run_filter = self.run_filter

        all_permutations = _recursive_parameter_solver(
            validation_tuple_list=list(parameters.items()), current_parameters={}
        )
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for permutation in all_permutations:
            parameter_set: dict[str, Any] = {}  # To hold the current set of parameters
            is_valid = True  # The parameter should not trigger an exception

            for key, value in permutation.items():
                # The parameter has a value - so set it in the permutation
                if isinstance(value, ExistingParameterPermutation):
                    parameter_set[key] = value.parameter

                if (
                    isinstance(value, InvalidParameterPermutation)
                    or isinstance(value, MissingParameterPermutation)
                    and not value.is_valid
                ):
                    # The parameter should trigger an exception
                    is_valid = False
            if is_valid:
                valid.append(parameter_set)
            else:
                invalid.append(parameter_set)

        # Test all the combinations of valid inputs are accepted
        for parameters in invalid:
            test_filter = filter_type(**filter_arguments)
            test_filter.add_inputs(parameters)
            try:
                test_filter.run(
                    validate_only=not run_filter
                )  # Should raise a validation error
            except FilterInputValidationError:
                continue
            except ValidationError:
                continue
            raise FilterInputValidationError(
                f"Parameters {parameters} did not raise an "
                f"exception with filter: {test_filter.name}"
            )
        for parameters in valid:
            test_filter = filter_type(**filter_arguments)
            test_filter.add_inputs(parameters)
            test_filter.run(validate_only=not run_filter)

        return self


def validate_filter_inputs(
    filter_to_test: type[BaseFilter], validation_data: dict
) -> None:
    """Tests a filter with a validation data dictionary.  Checks that FilterInputValidationErrors
    are raised when data is missing or incorrect.

    :param filter_to_test: the class of the filter to test
    :type filter_to_test: BaseFilter
    :param validation_data: A dictionary, where each key is an input parameter
      for the filter, and the value is a list/tuple where:

        :[0]: is_optional
        :[1]: a value that should pass
        :[2:end]: values that should fail

    :type validation_data: dict
    """
    test_filter = filter_to_test()
    test_data = deepcopy(validation_data)
    # check with inputs that should pass
    for data_key in test_data:
        test_filter.add_input(data_key, test_data[data_key][1])
    test_filter.run()

    for inputs_key in validation_data:
        test_data = deepcopy(validation_data)
        test_filter = filter_to_test()
        is_optional: bool = test_data[inputs_key][0]

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            test_filter.add_input(data_key, test_data[data_key][1])

        # optional inputs should run without issue
        if is_optional:
            test_filter.run()
        else:
            with pytest.raises(FilterInputValidationError):
                test_filter.run()

        # Try data that should fail
        for test_value in validation_data[inputs_key][2:]:
            test_filter = filter_to_test()
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])
            test_filter.add_input(inputs_key, test_value)

            with pytest.raises(FilterInputValidationError):
                test_filter.run()
