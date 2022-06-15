""" MathsFilter Class"""

from copy import copy
from typing import Dict, Final, Union

import numpy as np

from mrimagetools.containers.image import BaseImageContainer
from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.utils.maths import (
    ExpressionEvaluator,
    ExpressionEvaluatorConfig,
    ExpressionEvaluatorError,
    InputType,
    expression_evaluator,
)
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
    shape_validator,
)


class MathsFilter(BaseFilter):
    """A general expression evalutator that allows calculations to be performed
    using equations. For example:
    "(A+B)/5"
    The variables may be a BaseImageContainer, or be floats, integers or complex.
    NOTE: metadata passing is not handled, this must be done manually

    Currently supports the following unary operations:
    - Unary subtract. e.g. -A

    and the following binary operations:

    - Add. e.g. A+B or delta+4 or image_a+image_b
    - Subtract. e.g. C-D or 5-beta or 1-2
    - Divide. e.g. E/image_a or 5/G or 9/2 (optionally using a safe-divide,
      where division by zero gives 0 - see the __init__ function)
    - Multiply. e.g. H*I or image*7 or 7*9
    - Power. e.g. J**K or image**2 or 2**3

    Complex expressions may be nested, for example:
    `-(image_a+delta**B)/(C-(2+5j))`
    Note that in the above example, `image_a`, `delta`, `B` and `C` would need to be
    defined as parameters.

    The expression will be evaluated and a single output created named `result` with
    the result. For more information, look at the documentation for
    :class:`mrimagetools.utils.maths.expression_evaluator`

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`MathsFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`MathsFilter.KEY_EXPRESSION`

    :param 'expression': The input data image, cannot be a phase image
    :type 'expression': str

    Other parameters should match the variable names in the expression and be either:
    - int
    - float
    - complex
    - BaseImageContainer

    **Outputs**

    :param 'result': The result of the expression
    :type 'result': BaseImageContainer, int, float or complex
    """

    # KEY_CONSTANTS
    KEY_EXPRESSION = "expression"
    KEY_RESULT = "result"

    def __init__(self, config: ExpressionEvaluatorConfig = "default"):
        """Initialise the object. By default, numpy-like operations are used.
        You might want to change the `config` variable to "safe_divide".
        This will mean that all division by zero gives the result of zero
        :raises: ValueError if the config is not supported"""
        super().__init__(name="Maths Filter")
        # Predefine the expression evalutator for use within the function
        self.evalutator: Final[ExpressionEvaluator] = expression_evaluator(config)

    @property
    def variables(self):
        """Return the input variables (every parameter except the expression)"""
        return {
            key: value
            for key, value in self.inputs.items()
            if key != self.KEY_EXPRESSION
        }

    @property
    def parsed_variables(self):
        """Return the input variables, substiting the images for nuumpy arrays.
        This can then be used directly in the ExpressionEvaluator"""
        return {
            key: value.image if isinstance(value, BaseImageContainer) else value
            for key, value in self.variables.items()
        }

    def _run(self):
        """Evaluate the expression using the supplied variables"""

        input_copy = copy(self.inputs)
        expression: str = input_copy.pop("expression")
        variables = input_copy
        # Reconstruct the variables so the imagecontainers are mapped to numpy arrays
        expression_result: InputType = self.evalutator(
            expression=expression, **self.parsed_variables
        )

        result: Union[BaseImageContainer, float, int, complex, None] = None

        if isinstance(expression_result, np.ndarray):
            # Need to put the numpy array back into an image container,
            # so take the first one and clone it
            for value in variables.values():
                if isinstance(value, BaseImageContainer):
                    if not isinstance(expression_result, np.ndarray):
                        # Typeguard
                        raise TypeError(
                            "Trying to assign a number to a numpy array"
                        )  # we should never get here!
                    result = value.clone()
                    result.image = expression_result
                break
        else:
            # The expression did not return a numpy array - so we can just assign it to "result"
            result = expression_result

        if result is None:
            raise TypeError(
                f"Maths filter result ({result}) was not correctly assigned"
            )

        self.outputs[self.KEY_RESULT] = result

    def _validate_inputs(self):
        """Validate the inputs.
        - 'expression' must be a string
        - 'expression' must be a valid expressions
        - all parameters in the expression have been defined
        - all other parameters must be of type BaseImageContainer, float, int or complex
        - all BaseImageContainer parameters must have matching shape and affine matrices
        """
        # expression must be a string
        image_containers: Dict[str, BaseImageContainer] = {}

        # Check that all of the input variables are of a valid type
        for key, value in self.inputs.items():
            if key == self.KEY_EXPRESSION:
                # Ignore the expression - this is not a variable
                continue
            if isinstance(value, BaseImageContainer):
                # OK
                image_containers[key] = value  # We need to check these later
                continue
            if isinstance(value, (BaseImageContainer, float, int, complex)):
                # OK
                continue
            # A supported variable type was not found
            raise FilterInputValidationError(
                f"Input {key} with value {value} is not a supported variable type"
            )

        input_validator = ParameterValidator(
            parameters={
                self.KEY_EXPRESSION: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ]
                )
            },
            post_validators=[
                shape_validator(list(image_containers.keys())),
            ],
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        if image_containers:
            # Check affines match
            first_key, first_value = next(iter(image_containers.items()))
            affine = first_value.affine
            for key, value in image_containers.items():
                if not np.allclose(affine, value.affine):
                    raise FilterInputValidationError(
                        f"Affine for image {first_key} does not match affine for image {key}"
                    )

        # Check the expression/parameters are valid
        try:
            self.evalutator.validate(
                self.inputs[self.KEY_EXPRESSION], **self.parsed_variables
            )
        except ExpressionEvaluatorError as error:
            raise FilterInputValidationError(error) from error
