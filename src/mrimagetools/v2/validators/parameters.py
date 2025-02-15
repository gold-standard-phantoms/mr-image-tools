"""
Used to perform parameter validation. The most useful documentation can
be found on the class 'ParameterValidator'
"""
import os
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from mrimagetools.v2.containers.image import BaseImageContainer


class ValidationError(Exception):
    """Used to indicate that a dictionary is invalid"""


class Validator:
    """All _validator functions return an object of this class.
    The object can be called with a value to check whether the value
    is valid. A string method is also available to display the validator's
    criteria message.
    """

    def __init__(self, func: Callable[[Any], bool], criteria_message: str) -> None:
        """
        :param func: the callable (must take a single value) which returns
        True or False depending on whether the validation criteria have been met
        :param criteria_message: the message that can be used to display the
        validation criteria
        """
        self.func = func
        self.criteria_message = criteria_message

    def __call__(self, value: Any) -> bool:
        return self.func(value)

    def __str__(self) -> str:
        return self.criteria_message


def isfile_validator(
    extensions: Union[str, list[str], None] = None, must_exist: bool = False
) -> Validator:
    """
    Validates that a given value is a path to a file.
    :param extension: string of file extensions that must match
    :param must_exist: the file must exist on disk"""

    if extensions is not None:
        for extension in extensions:
            if not isinstance(extension, str):
                raise TypeError("All extensions must be strings")

    def validate(value: str) -> bool:
        # check that value is a str
        if isinstance(value, str):
            # check that the value isn't a path
            is_not_dir = not os.path.isdir(value)
            # check that the path actually exists
            root_folder_exists = os.path.exists(os.path.dirname(value))

            # check that the file has a valid extension
            if extensions is not None:
                valid_extension = any(
                    value.lower().endswith(extension.lower())
                    for extension in extensions
                )
            else:
                valid_extension = True
            if must_exist:
                return (
                    is_not_dir
                    and root_folder_exists
                    and valid_extension
                    and os.path.exists(value)
                )
            return is_not_dir and root_folder_exists and valid_extension
        return False

    return Validator(
        validate,
        (
            "Value must be a valid filename" + f"with extension(s) {extensions}"
            if extensions is not None
            else "" + "and must exist" if must_exist else ""
        ),
    )


def isinstance_validator(a_type: Union[type, tuple[type, ...]]) -> Validator:
    """
    Validates that a given value is an instance of the given type(s) (or or derived from).
    a_type: a type e.g. str, or a tuple of types e.g. (int, str)
    """
    if not isinstance(a_type, type):
        if not (
            isinstance(a_type, tuple)
            and all(isinstance(single_type, type) for single_type in a_type)
        ):
            raise TypeError(f"{a_type} is not a type or list of types")
    msg = "Value must be of type "
    if isinstance(a_type, tuple):
        msg += " or ".join([x.__name__ for x in a_type])
    else:
        msg += a_type.__name__

    return Validator(lambda value: isinstance(value, a_type), msg)


def range_exclusive_validator(start, end) -> Validator:
    """
    Validate that a given value is between a given range (excluding the start and end values).
    Can be used with int, float or BaseImageContainer.
    :param start: the start value
    :param end: the end value
    """
    if start >= end:
        raise ValueError(f"Start ({start} must be less than end ({end})")

    def validate(value: Union[float, int, BaseImageContainer]) -> bool:
        if isinstance(value, (float, int)):
            return start < value < end
        if isinstance(value, BaseImageContainer):
            return (value.image > start).all() and (value.image < end).all()
        return False

    return Validator(
        validate, f"Value(s) must be between {start} and {end} (exclusive)"
    )


def range_inclusive_validator(start, end) -> Validator:
    """
    Validate that a given value is between a given range (including the start and end values)
    Can be used with int, float or BaseImageContainer.
    :param start: the start value
    :param end: the end value
    """
    if start > end:
        raise ValueError(f"Start ({start} must be less than or equal to end ({end})")

    def validate(value: Union[float, int, BaseImageContainer]) -> bool:
        if isinstance(value, (float, int)):
            return start <= value <= end
        if isinstance(value, BaseImageContainer):
            return (value.image >= start).all() and (value.image <= end).all()
        return False

    return Validator(
        validate, f"Value(s) must be between {start} and {end} (inclusive)"
    )


def greater_than_equal_to_validator(start) -> Validator:
    """
    Validate that a given value is greater or equal to a number.
    Can be used with int, float or BaseImageContainer.
    :param start: the value to be greater than or equal to
    """
    if not isinstance(start, (float, int)):
        raise TypeError(f"Start ({start}) must be a number type")

    def validate(value: Union[float, int, BaseImageContainer]) -> bool:
        if isinstance(value, (float, int)):
            return start <= value
        if isinstance(value, BaseImageContainer):
            return bool((start <= value.image).all())
        return False

    return Validator(validate, f"Value(s) must be greater than or equal to {start}")


def greater_than_validator(start) -> Validator:
    """
    Validate that a given value is greater than a number.
    Can be used with int, float or BaseImageContainer.
    :param start: the value to be greater than
    """
    if not isinstance(start, (float, int)):
        raise TypeError(f"Start ({start}) must be a number type")

    def validate(value: Union[float, int, BaseImageContainer]) -> bool:
        if isinstance(value, (float, int)):
            return start < value
        if isinstance(value, BaseImageContainer):
            return bool((start < value.image).all())
        return False

    return Validator(validate, f"Value(s) must be greater than {start}")


def from_list_validator(
    options: Union[list, tuple], case_insensitive: bool = False
) -> Validator:
    """
    Validates that a given value is from a list.
    :param option: the list of options, one of which the value must match
    :param case_insensitive: perform a case-insensitive matching
    """
    if not isinstance(options, (list, tuple)):
        raise TypeError(f"Input must be a list, is {options}")
    if case_insensitive:
        lowercase_options = [x.lower() if isinstance(x, str) else x for x in options]
        return Validator(
            lambda value: (
                value.lower() in lowercase_options
                if isinstance(value, str)
                else value in lowercase_options
            ),
            f"Value must be in {options} (ignoring case)",
        )
    return Validator(lambda value: value in options, f"Value must be in {options}")


def of_length_validator(length: int) -> Validator:
    """
    Validates that a given value has a given length.
    Might be, for example, a list or a string.
    :param length: the required length of the value
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Required length must be a positive integer")

    msg = f"Value (string or list) must have length {length}"

    return Validator(
        lambda value: hasattr(value, "__len__") and len(value) == length, msg
    )


def list_of_type_validator(a_type: Union[type, tuple[type, ...]]) -> Validator:
    """
    Validates that a given value is a list of the given type(s).
    a_type: a type e.g. str, or a tuple of types e.g. (int, str)
    """
    if not isinstance(a_type, type):
        if not (
            isinstance(a_type, tuple)
            and all(isinstance(single_type, type) for single_type in a_type)
        ):
            raise TypeError(f"{a_type} is not a type or list of types")
    msg = "Value must be a list of type "
    if isinstance(a_type, tuple):
        msg += " or ".join([x.__name__ for x in a_type])
    else:
        msg += a_type.__name__

    return Validator(
        lambda value: all(isinstance(single, a_type) for single in value), msg
    )


def non_empty_list_or_tuple_validator() -> Validator:
    """Validates that a value is a list or tuple and is non-empty"""
    return Validator(
        lambda value: isinstance(value, (list, tuple)) and len(value) > 0,
        "Value must be a non-empty list",
    )


def regex_validator(pattern: str, case_insensitive: bool = False) -> Validator:
    """
    Validates that a value matches the given regex pattern
    :param pattern: the regex pattern to match
    :param case_insensitive: perform a case-insensitive matching
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{pattern} is not a valid python regex pattern") from exc
    return Validator(
        lambda value: (
            bool(
                re.match(pattern, value, flags=re.IGNORECASE if case_insensitive else 0)
            )
            if isinstance(value, str)
            else False
        ),
        (
            "Value must match pattern"
            f" {pattern}{' (ignoring case)' if case_insensitive else ''}"
        ),
    )


def reserved_string_list_validator(
    strings: list[str], delimiter: str = " ", case_insensitive: bool = False
) -> Validator:
    """
    Validates that the value is a string which is comprised only of the list of given strings,
    separated by the delimiter. The strings may be repeated multiple times and in any order,
    although the value must not be the empty string.
    e.g. with strings=['foo','bar'] and delimiter='_', this would match:
    "foo_bar_foo", "foo", "bar", "bar_bar_bar_bar_foo"
    but would not match:
    "", "FOO", "anythingelse", "foo__bar", "bar foo"
    :param strings: a list of strings
    :param delimiter: a delimiter (defaults to space)
    :param case_insensitive: perform a case-insensitive matching
    """
    if not isinstance(strings, list):
        raise TypeError(f"string must be a list, is {strings}")
    if len(strings) == 0:
        raise ValueError("strings list cannot be empty")
    for string in strings:
        if not isinstance(string, str):
            raise ValueError(
                f"strings list must only contains strings. Contains {string}"
            )

    concat_strings = "|".join(strings)
    pattern = rf"^({concat_strings})({delimiter}({concat_strings}))*$"
    return Validator(
        regex_validator(pattern=pattern, case_insensitive=case_insensitive).func,
        (
            f"Value must be a string combination of {strings} separated by "
            f"'_'{' (ignoring case)' if case_insensitive else ''}"
        ),
    )


def for_each_validator(item_validator: Validator) -> Validator:
    """
    Validates that the value must be iteratable and each of its items are
    valid based on the item_validator.
    e.g.
    validator=for_each_validator(greater_than_validator(0.7))
    would create a validator that check all items in a last are > 0.7
    :param item_validator: a validator to apply to each item in a list
    """

    if not isinstance(item_validator, Validator):
        raise TypeError("First argument of `for_each_validator` must be a validator")
    return Validator(
        lambda value: isinstance(value, (list, tuple))
        and all([item_validator(v) for v in value]),
        f"Must be a list or tuple and for each value in the list: {item_validator}",
    )


def has_attribute_value_validator(
    attribute_name: str, attribute_value: Any
) -> Validator:
    """Validates that the parameter has an attribute with the given name
    and also that the attribute value matches a given value.
    e.g.
    has_attribute_value_validator("a_property", 500.0)
    would create a validators that check that an object (`obj') has a property
    `a_property` that matches 500.0. i.e. `obj.a_property == 500.0`
    :param attribute_name: the attribute name to compare
    :param attribute_value: the value of the attribute to compare against
    """
    if not isinstance(attribute_name, str):
        raise TypeError("The attribute_name must be a string")

    return Validator(
        lambda value: hasattr(value, attribute_name)
        and getattr(value, attribute_name) == attribute_value,
        f"Value must have an attribute {attribute_name} with value {attribute_value}",
    )


def or_validator(validators: list[Validator]) -> Validator:
    """Boolean OR between supplied validators.
    If any of the supplied validators evaluate as True, this validator
    evaluates as True.

    :param validators: A list (or tuple) of validators
    :type validators: Union[List[Validator], Tuple[Validator, ...]]
    """
    if not isinstance(validators, (list, tuple)):
        raise TypeError("The argument to `or_validator` must be a list or tuple")
    for validator in validators:
        if not isinstance(validator, Validator):
            raise TypeError("Each element of input must be of type Validator")

    return Validator(
        func=lambda value: any([validator(value) for validator in validators]),
        criteria_message=" OR ".join([str(validator) for validator in validators]),
    )


def and_validator(validators: list[Validator]) -> Validator:
    """Boolean AND between supplied validators.
    If all of the supplied validators evaluate as True, this validator
    evaluates as True.

    :param validators: A list (or tuple) of validators
    :type validators: Union[List[Validator], Tuple[Validator, ...]]
    """
    if not isinstance(validators, (list, tuple)):
        raise TypeError("The argument to `and_validator` must be a list or tuple")
    for validator in validators:
        if not isinstance(validator, Validator):
            raise TypeError("Each element of input must be of type Validator")

    return Validator(
        func=lambda value: all([validator(value) for validator in validators]),
        criteria_message=" AND ".join([str(validator) for validator in validators]),
    )


def shape_validator(
    keys: Union[list[str], tuple[str]], maxdim: Union[int, None] = None
) -> Validator:
    """Checks that all of the keys have the same shape
    If all the supplied inputs have matching shapes, this validator
    evaluates as True.
    If the supplied input key list is empty, this validator evaluates as True.

    Note this is intended to be used as a post validator as the argument
    for the validator is a dictionary.

    :param keys: A list (or tuple) of strings
    :type keys: Union[List[str], Tuple[str]]
    """

    if not isinstance(keys, (list, tuple)):
        raise TypeError(
            "The argument 'keys' to 'shape_validator' must be a list or tuple"
        )

    for key in keys:
        if not isinstance(key, str):
            raise TypeError("Each element of input must be of type str")
    if maxdim is not None:
        if not isinstance(maxdim, int):
            raise TypeError("The argument 'maxdim' to 'shape_validator' must be a int")

    def validate(d: dict) -> bool:
        if not keys:
            # list is empty
            return True
        # check the keys exist in d
        keys_exist = all([key in d for key in keys])
        # check all values have the attribute `shape`
        if keys_exist:
            have_shape = all([hasattr(d[key], "shape") for key in keys])
        else:
            have_shape = False
        # check the shapes match with the shape of the first key
        if have_shape:
            if maxdim is not None:
                shapes_match = [d[key].shape[:maxdim] for key in keys].count(
                    d[keys[0]].shape[:maxdim]
                ) == len(keys)
            else:
                shapes_match = [d[key].shape for key in keys].count(
                    d[keys[0]].shape
                ) == len(keys)

        else:
            shapes_match = False
        return keys_exist and have_shape and shapes_match

    return Validator(
        validate,
        criteria_message=f"{keys} must all have the same shapes",
    )


class Parameter:
    # pylint: disable=too-few-public-methods
    """A description of a parameter which is to be validated against"""

    def __init__(
        self,
        validators: Union[Callable[..., bool], list[Callable[..., bool]]],
        default_value=None,
        optional=False,
    ):
        """
        :param validators: a single validators, or a list of validators. The
        validators must be initialised with their parameters. Examples might be:
        validators=greater_than_validator(0.7)
        validators=[range_inclusive_validator(1, 2), from_list_validator([1.5, 1.6])]
        :param default_value: a default value. If set, when an input dictionary is validated,
        if a given parameter is missing, it will be given this value
        :param optional: if False, this parameter must be supplied.
        """
        if isinstance(validators, list):
            self.validators = validators
        else:
            self.validators = [validators]
        self.default_value = default_value
        self.optional = True if default_value is not None else optional

        # Must ensure the default value is valid
        if default_value is not None:
            for validator in self.validators:
                if not validator(default_value):
                    raise ValueError(f"Default value of {default_value} is not valid")


class ParameterValidator:
    """Used to validate a dictionary of parameters specified with the Parameter class against
    an input dictionary. Will also insert any default values that are missing from the input
    dictionary.
    """

    def __init__(
        self,
        parameters: dict[str, Parameter],
        post_validators: Optional[list[Validator]] = None,
    ):
        """
        :param parameters: a dictionary of input parameters. An example might be:
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(non_empty_list_validator(), default_value=[1, 2, 3]),
        }
        :param post_validators: a list of additional validators, which will be executed
        after the parameters have been validated and defaults inserted.
        """
        if post_validators is None:
            post_validators = []

        for parameter in parameters.values():
            if not isinstance(parameter, Parameter):
                raise TypeError(
                    "All values passed to InputParameters must be of Parameter type,"
                    " {key} is not"
                )

        for post_validator in post_validators:
            if not isinstance(post_validator, Validator):
                raise TypeError("All items in post_validators must be a Validator")

        self.parameters: dict[str, Parameter] = parameters
        self.post_validators: list[Validator] = post_validators

    def get_defaults(self) -> dict:
        """Return a dictionary of default values for each of the parameters
        in the ParameterValidator. If a parameter does not have a default value,
        it is excluded from the dictionary
        :return: a dictionary of default parameter values
        """
        defaults = {}
        for parameter_key, parameter_value in self.parameters.items():
            if parameter_value.default_value is not None:
                defaults[parameter_key] = parameter_value.default_value
        return defaults

    def validate(self, d: dict, error_type: type[Exception] = ValidationError) -> dict:
        """
        Validate an input dictionary, replacing missing dictionary entries with default values.
        If any of the dictionary entries are invalid w.r.t. any of the validators, a
        ValidationError will be raised (unless error_type is defined, see parameter docs).
        :param d: the input dictionary. e.g.: {"foo": "bar foo bar"}
        :param error_type: the type of Exception to be raised.
        :return: the dictionary with any defaults filled. e.g.
        {
            "foo": "bar foo bar",
            "bar": [1, 2, 3],
        }
        """
        # error_type must derived from Exception
        if not issubclass(error_type, Exception):
            raise TypeError("error_type must be a subclass of Exception")
        return_dict = deepcopy(d)
        errors = []
        # Check all non-optional parameters are present
        for parameter_name, parameter in self.parameters.items():
            # If a parameter is missing
            if d.get(parameter_name, None) is None:
                # If the parameter is optional, set it to the
                if parameter.optional and parameter.default_value is not None:
                    return_dict[parameter_name] = parameter.default_value
                if not parameter.optional:
                    errors.append(
                        f"{parameter_name} is a required parameter and is "
                        "not in the input dictionary"
                    )
            else:
                # We need to validate the parameter against its validators
                for validator in parameter.validators:
                    if not validator(d[parameter_name]):
                        errors.append(
                            f"Parameter {parameter_name} with value"
                            f" {d[parameter_name]} does not meet the following"
                            f" criterion: {validator}"
                        )

        # Check all of the post_validators
        for post_validator in self.post_validators:
            if not post_validator(return_dict):
                errors.append(str(post_validator))

        if errors:
            raise error_type(". ".join(errors))
        return return_dict
