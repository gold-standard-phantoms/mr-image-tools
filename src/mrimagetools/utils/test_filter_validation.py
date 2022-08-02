"""tests for filter_validation.py"""

from typing import Dict, Final

import numpy as np
import pytest
from pydantic import StrictFloat

from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.utils.filter_validation import (
    FilterValidationModel,
    FilterValidationModelParameter,
    validate_filter_inputs,
)
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
)


class ProductFilter(BaseFilter):
    """A filter that multiplies its inputs together

    input1: float
    input2: float
    input3: float, optional

    the output is called `product`
    """

    def __init__(self) -> None:
        super().__init__(name="ProductFilter")

    def _run(self) -> None:
        """Multiplies all inputs and creates an `output` with the result"""
        self.outputs["product"] = np.prod(self.inputs.values())  # type:ignore

    def _validate_inputs(self) -> None:
        input_validator = ParameterValidator(
            parameters={
                "input1": Parameter(validators=isinstance_validator(float)),
                "input2": Parameter(validators=isinstance_validator(float)),
                "input3": Parameter(
                    validators=isinstance_validator(float), optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)


def test_validate_filter_inputs_function() -> None:
    """test the inputs validation filter"""
    validation_data = {
        "input1": [False, 10.0, 20, "str"],
        "input2": [False, 20.0, 20, "str"],
        "input3": [True, 15.0, 15, "str"],
    }

    validate_filter_inputs(ProductFilter, validation_data)


def test_non_optional_field_needs_valid_values() -> None:
    """Check that if an input value is not optional, we have some values for it
    (and vice-versa)"""

    FilterValidationModelParameter[str](is_optional=True, invalid_values=[1])
    FilterValidationModelParameter[str](is_optional=True)
    FilterValidationModelParameter[str](
        is_optional=True, valid_values=["foo"], invalid_values=[1]
    )
    FilterValidationModelParameter[str](
        is_optional=False, valid_values=["foo"], invalid_values=[1]
    )
    # We need a valid input as the input is not optional
    with pytest.raises(FilterInputValidationError):
        FilterValidationModelParameter[str](is_optional=False, invalid_values=[1])
    # The default is non-optional, so test that too
    with pytest.raises(FilterInputValidationError):
        FilterValidationModelParameter[str](invalid_values=[1])


def test_generic_filter_validation_parameter() -> None:
    """Test the generic type checking. Use the "Strict" versions to prevent type coercion"""
    with pytest.raises(FilterInputValidationError):
        FilterValidationModelParameter[StrictFloat](
            valid_values=[0.0, "1.0"]  # type:ignore
        )
    FilterValidationModelParameter[StrictFloat](valid_values=[0.0, 1.0])


def test_validate_filter_inputs_function_with_model() -> None:
    """The the general functionality of the `FilterValidationModel`"""
    # valid inputs
    valid_inputs: Final[Dict] = {
        "input1": FilterValidationModelParameter[StrictFloat](
            is_optional=False, valid_values=[10.0], invalid_values=[20, "str"]
        ),
        "input2": FilterValidationModelParameter[StrictFloat](
            is_optional=False, valid_values=[20.0], invalid_values=[20, "str"]
        ),
        "input3": FilterValidationModelParameter[StrictFloat](
            is_optional=True, valid_values=[15.0], invalid_values=[15, "str"]
        ),
    }
    # should pass
    FilterValidationModel(filter_type=ProductFilter, parameters=valid_inputs)

    with pytest.raises(FilterInputValidationError):
        FilterValidationModel(
            filter_type=ProductFilter,
            parameters={
                **valid_inputs,
                "input2": FilterValidationModelParameter[StrictFloat](
                    is_optional=False,
                    valid_values=[20.0],
                    invalid_values=[20.0, "str"],  # 20.0 is valid
                ),
            },
        )
    with pytest.raises(FilterInputValidationError):
        FilterValidationModel(
            filter_type=ProductFilter,
            parameters={
                **valid_inputs,
                "input2": FilterValidationModelParameter[StrictFloat](
                    is_optional=False,
                    valid_values=[20.0, "str"],  # type:ignore # "str" is not valid
                    invalid_values=[20, "str"],
                ),
            },
        )
    with pytest.raises(FilterInputValidationError):
        FilterValidationModel(
            filter_type=ProductFilter,
            parameters={
                **valid_inputs,
                "input3": FilterValidationModelParameter[StrictFloat](
                    is_optional=True,
                    valid_values=[15.0],
                    invalid_values=[15, "str", 10.0],  # 10.0 is not invalid
                ),
            },
        )
    with pytest.raises(FilterInputValidationError):
        FilterValidationModel(
            filter_type=ProductFilter,
            parameters={
                **valid_inputs,
                "input1": FilterValidationModelParameter[StrictFloat](
                    is_optional=True,  # input1 is not optional
                    valid_values=[10.0],
                    invalid_values=[20, "str"],
                ),
            },
        )
