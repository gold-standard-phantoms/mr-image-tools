"""General image maths functions"""

import ast
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Final, Literal, Type, Union, get_args

import numpy as np

from mrimagetools.utils.safe_divide import safe_divide

InputType = Union[np.ndarray, float, int, complex]

# These will be used as defaults if they are not changed in the ExpressionEvaluator
SUPPORTED_UNARY_OPERATORS: Final[
    dict[type[ast.AST], Callable[[InputType], InputType]]
] = {
    ast.USub: lambda x: np.subtract(0, x),
}

# These will be used as defaults if they are not changed in the ExpressionEvaluator
SUPPORTED_BINARY_OPERATORS: Final[
    dict[type[ast.AST], Callable[[InputType, InputType], InputType]]
] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.Pow: np.power,
}


class ExpressionEvaluatorError(Exception):
    """A an error type for all errors raised in this file"""


class ExpressionSyntaxError(ExpressionEvaluatorError):
    """For syntax errors in parsing the expression"""


class OperatorNotSupportedError(ExpressionEvaluatorError):
    """Raised if an operation in the expression is not supported"""


class VariableMissingError(ExpressionEvaluatorError):
    """Raised if an variable in the expression is missing"""


class BadVariableTypeError(ExpressionEvaluatorError):
    """Raised if a variable is not a supported type (int, float, complex, numpy array)"""


class UnsupportedNodeError(ExpressionEvaluatorError):
    """The AST has arrived at node that is unknown"""


def _conditional_raiser(
    raise_exception: bool,
) -> Callable[[ExpressionEvaluatorError], bool]:
    """Returns a function that raises the provided exception, only if
    `raise_exception` is True, other False is always returned.
    Used to help reduce boilerplate code"""

    def _raise_or_false(error: ExpressionEvaluatorError) -> bool:
        """Raises the error if raise_exception is True, otherwise return False"""
        if raise_exception:
            raise error
        return False

    return _raise_or_false


@dataclass
class ExpressionEvaluator:
    """A general expression evalutator that allows calculations to be performed
    using equations.
    The expression evalutator must first be created, then it can be used. For example:
    `result = ExpressionEvaluator()(expression="(A+B)/5", A=np.array([1.0, 2.0]), B=1.0)`
    you can also change the operations supported, for example:
    ```
    e = ExpressionEvaluator(unary_operators={ast.USub: lambda x: np.subtract(0, x)})
    e.binary_operators[ast.Mult] = np.divide
    result = e(expression="-(A*B)/5", A=np.array([1.0, 2.0]), B=1.0)`
    ```

    This allows extensibility and future-proofs the expression evaluator

    To save operation time, an expression and its variable can be validated
    before the tree is parsed and evaluated, using the `is_valid()` function.

    The variables may be a numpy array, or be floats, integers or complex.
    The expression will be evaluated using an abstract syntax tree.

    We want to be able to evaluate the following unary operations:
    - Unary subtract. e.g. -A, using numpy.subtract with the default optional arguments,
    and 0 set as the first argument (using a lambda function).

    and the following binary operations:

    - Add. e.g. A+B or delta+4 or 123+1, using numpy.add with the default optional arguments
    - Subtract. e.g. C-D or 5-beta or 1-2, using numpy.subtract with the default optional arguments
    - Divide. e.g. E/F or 5/G or 9/2 using numpy.divide. NOTE: this can be a "safe_divide",
    by setting the "safe_divide" config flag. This will ensure that, when the divisor is a numpy
    array, a division by zero = zero)
    - Multiply. e.g. H*I or image*7 or 7*9, using numpy.multiply with the default optional arguments
    - Power. e.g. J**K or image**2 or 2**3, using numpy.power with the default optional arguments

    Operations should work on both numpy arrays, float, integers, complexes,
    and a combination of all the above.
    Operations can be nested in expressions and parenthesis used, for example, (A**B+2)/C
    Once run, the filter will return the result, which may be a numpy array, float, int or complex.
    """

    unary_operators: dict[type[ast.AST], Callable[[InputType], InputType]] = field(
        default_factory=lambda: copy(SUPPORTED_UNARY_OPERATORS)
    )

    binary_operators: dict[
        type[ast.AST], Callable[[InputType, InputType], InputType]
    ] = field(default_factory=lambda: copy(SUPPORTED_BINARY_OPERATORS))

    def _eval(self, node: ast.AST, variables: dict[str, InputType]) -> InputType:
        """Evaluates the node using the supplied variables and the supported operators
        :param node: The ast node
        :param variables: The variables to be substituted into the expression
        :return: The result of the node (solving the child tree)
        :raises: a subclass of :class:`ExpressionEvaluatorError` if an exception is encountered
        """
        # Check the tree and raise exceptions is any issues are found
        self._valid(node=node, variables=variables, raise_exception=True)
        if isinstance(node, ast.Num):  # Node is a number (int, float, complex)
            return node.n
        if isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return self.unary_operators[type(node.op)](
                self._eval(node=node.operand, variables=variables)
            )
        if isinstance(node, ast.BinOp):  # <left operand> <operator> <right operand>
            return self.binary_operators[type(node.op)](
                self._eval(node=node.left, variables=variables),
                self._eval(node=node.right, variables=variables),
            )
        if isinstance(
            node, ast.Name
        ):  # Node is a variable (throws a KeyError if not found)
            return variables[node.id]

        raise UnsupportedNodeError(f"The node {node} is unknown and cannot be parsed")

    # pylint: disable=too-many-return-statements
    def _valid(
        self, node: ast.AST, variables: dict[str, InputType], raise_exception=False
    ) -> bool:
        """Determines whether the expression can be evaluated with the given variables.
        Optionally, raises an exception if the expression cannot be evaluated
        Evaluates the node using the supplied variables and the supported operators
        :param node: The ast node
        :param variables: The variables to be substituted into the expression
        :param raise_exception: Raise an exception if an error is found (will be a subclass of
         :class:`ExpressionEvaluatorError`)
        :return: if the node can be evaluated with the given variables
        :raises: a subclass of :class:`ExpressionEvaluatorError` if raise_exception is True
        """
        conditional_raise = _conditional_raiser(raise_exception)
        if isinstance(node, ast.Num):  # Node is a number (int, float, complex)
            return True  # This node is implicitly evaluated
        if isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            if type(node.op) in self.unary_operators:
                return self._valid(node=node.operand, variables=variables)
            # Operation is not supported
            return conditional_raise(
                OperatorNotSupportedError(
                    f"The {node.op} unary operation is not supported"
                )
            )
        if isinstance(node, ast.BinOp):  # <left operand> <operator> <right operand>
            if type(node.op) in self.binary_operators:
                return self._valid(node=node.left, variables=variables) and self._valid(
                    node=node.right, variables=variables
                )
            # Operation is not supported
            return conditional_raise(
                OperatorNotSupportedError(
                    f"The {node.op} binary operation is not supported"
                )
            )
        if isinstance(node, ast.Name):  # Node is a variable
            if node.id not in variables:
                # The variable is missing
                return conditional_raise(
                    VariableMissingError(
                        f"The variable `{node.id}` has not been defined"
                    )
                )
            if not isinstance(variables[node.id], get_args(InputType)):
                # The variable is an unsupported type
                return conditional_raise(
                    BadVariableTypeError(
                        f"The variable `{node.id}` is an unsupported type"
                    )
                )
            # The variable exists and is a supported type, and therefore, can be used
            return True
        return conditional_raise(
            UnsupportedNodeError(f"The node {node} is unknown and cannot be parsed")
        )

    @staticmethod
    def _generate_tree(expression: str) -> ast.Expression:
        """Generate the abstract syntax tree
        :raises ExpressionSyntaxError: if the syntax is bad"""
        try:
            return ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ExpressionSyntaxError(error) from error

    def is_valid(self, expression: str, **variables: InputType) -> bool:
        """Determines whether the expression can be evaluated with the given variables.
        Evaluates the node using the supplied variables and the supported operators
        :param expression: The expression to validate
        :param variables: The variables to be substituted into the expression
        :param raise_exception: Raise an exception if an error is found (will be a subclass of
         :class:`ExpressionEvaluatorError`)
        :return: if the node can be evaluated with the given variables
        """
        try:
            tree = self._generate_tree(expression=expression)
        except ExpressionSyntaxError:
            return False
        return self._valid(node=tree.body, variables=variables)

    def validate(self, expression: str, **variables: InputType) -> None:
        """Determines whether the expression can be evaluated with the given variables.
        Raises an exception if the expression cannot be evaluated
        Evaluates the node using the supplied variables and the supported operators
        :param expression: The expression to validate
        :param variables: The variables to be substituted into the expression
        :param raise_exception: Raise an exception if an error is found (will be a subclass of
         :class:`ExpressionEvaluatorError`)
        """
        tree = self._generate_tree(expression=expression)
        self._eval(node=tree.body, variables=variables)

    def __call__(self, expression: str, **variables: InputType) -> InputType:
        """Operations should work on both numpy arrays, float, integers, complexes,
        and a combination of all the above.
        Operations can be nested in expressions and parenthesis used, for example, (A**B+2)/C
        Once run, the filter will return the result, which may be a numpy array, float,
        int or complex.

        :param 'expression': A string representation of the expression to be evaluated,
        e.g. `(A+B)+5`
        :param 'variables' : The remaining variables, which must be in the expression. These are
        numpy arrays, floats, ints or complex types.

        :return: A numpy array, float, int or complex with the result of the expression."""
        tree = self._generate_tree(expression=expression)
        return self._eval(node=tree.body, variables=variables)


ExpressionEvaluatorConfig = Literal["default", "safe_divide"]


def expression_evaluator(
    config: ExpressionEvaluatorConfig = "default",
) -> ExpressionEvaluator:
    """A general expression evalutator that allows calculations to be performed
    using equations.
    The expression evalutator must first be created, then it can be used. For example:
    `expression_evaluator()(expression=(A+B)/5, A=np.array([1.0, 2.0]), B=1.0)`
    you can also use different default, for example:
    `expression_evaluator('save_divide')(expression=(A+B)/5, A=np.array([1.0, 2.0]), B=1.0)`
    Will perform a safe divide IF the divisor is a numpy array (see below for more details)

    The variables may be a numpy array, or be floats, integers or complex.
    The expression will be evaluated using an abstract syntax tree.

    We want to be able to evaluate the following unary operations:
    - Unary subtract. e.g. -A, using numpy.subtract with the default optional arguments,
    and 0 set as the first argument (using a lambda function).

    and the following binary operations:

    - Add. e.g. A+B or delta+4 or 123+1, using numpy.add with the default optional arguments
    - Subtract. e.g. C-D or 5-beta or 1-2, using numpy.subtract with the default optional arguments
    - Divide. e.g. E/F or 5/G or 9/2 using numpy.true_divide. NOTE: this can be a "safe_divide",
    by setting the "safe_divide" config flag, a division by zero = zero
    - Multiply. e.g. H*I or image*7 or 7*9, using numpy.multiply with the default optional arguments
    - Power. e.g. J**K or image**2 or 2**3, using numpy.power with the default optional arguments

    Operations should work on both numpy arrays, float, integers, complexes,
    and a combination of all the above.
    Operations can be nested in expressions and parenthesis used, for example, (A**B+2)/C
    Once run, the filter will return the result, which may be a numpy array, float, int or complex.

    :param config: can be set to "safe_divide" to perform a safe division operation
    :return: an ExpressionEvaluator. Can be called with the following parameters:

    The parameters of the return of this function (a callable) are:
    :param 'expression': A string representation of the expression to be evaluated, e.g. `(A+B)+5`
    :param 'variables' : The remaining variables, which must be in the expression. These are
    numpy arrays, floats, ints or complex types.

    :return: A numpy array, float, int or complex with the result of the expression."""

    evaluator = ExpressionEvaluator()

    if config == "default":
        return evaluator

    if config == "safe_divide":
        evaluator.binary_operators[ast.Div] = safe_divide
        return evaluator

    raise ValueError(f"{config} is not a supported expression evalutator config")
