# type:ignore
# TODO: remove the above line and fix typing errors
""" parameters.py tests """
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from mrimagetools.v2.containers.image import BaseImageContainer, NumpyImageContainer
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    ValidationError,
    and_validator,
    for_each_validator,
    from_list_validator,
    greater_than_equal_to_validator,
    greater_than_validator,
    has_attribute_value_validator,
    isfile_validator,
    isinstance_validator,
    list_of_type_validator,
    non_empty_list_or_tuple_validator,
    of_length_validator,
    or_validator,
    range_exclusive_validator,
    range_inclusive_validator,
    regex_validator,
    reserved_string_list_validator,
    shape_validator,
)


def test_range_inclusive_validator_creator() -> None:
    """Check the inclusive validator creator raises
    errors when start >end"""

    with pytest.raises(ValueError):
        range_inclusive_validator(2, 1)


def test_range_inclusive_validator() -> None:
    """Test the inclusive validator with some values"""
    validator = range_inclusive_validator(-1, 1)
    assert str(validator) == "Value(s) must be between -1 and 1 (inclusive)"
    assert validator(-0.99)
    assert validator(0.99)
    assert validator(-1)
    assert validator(1)
    assert not validator(-1.01)
    assert not validator(1.01)
    assert not validator("not a number")


def test_range_inclusive_validator_image_container() -> None:
    """Test the inclusive validator with an image container"""
    validator = range_inclusive_validator(-1, 1)
    assert str(validator) == "Value(s) must be between -1 and 1 (inclusive)"

    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [0.1, -0.9]]))
    assert validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[-1.0, 0.2], [0.1, -0.9]]))
    assert validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [1, -0.9]]))
    assert validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [1.1, -0.9]]))
    assert not validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[-1.1, 0.2], [1, -0.9]]))
    assert not validator(image_container)


def test_range_exclusive_validator_creator() -> None:
    """Check the exclusive validator creator raises
    errors when start >= end"""
    with pytest.raises(ValueError):
        range_exclusive_validator(1, 1)
    with pytest.raises(ValueError):
        range_exclusive_validator(2, 1)


def test_range_exclusive_validator() -> None:
    """Test the exclusive validator with some values"""
    validator = range_exclusive_validator(-1, 1)
    assert str(validator) == "Value(s) must be between -1 and 1 (exclusive)"
    assert validator(-0.99)
    assert validator(0.99)
    assert not validator(-1)
    assert not validator(1)
    assert not validator("not a number")


def test_range_exclusive_validator_image_container() -> None:
    """Test the exclusive validator with an image container"""
    validator = range_exclusive_validator(-1, 1)
    assert str(validator) == "Value(s) must be between -1 and 1 (exclusive)"

    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [0.1, -0.9]]))
    assert validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[-1.0, 0.2], [0.1, -0.9]]))
    assert not validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [1, -0.9]]))
    assert not validator(image_container)


def test_greater_than_validator_creator() -> None:
    """Check the greater_than_validator creator raises
    errors when start is not a number type"""
    with pytest.raises(TypeError):
        greater_than_validator("str")
    with pytest.raises(TypeError):
        greater_than_validator([])


def test_greater_than_validator() -> None:
    """Test the greater_than_validator with some values"""
    validator = greater_than_validator(100)
    assert str(validator) == "Value(s) must be greater than 100"
    assert validator(101)
    assert validator(1000)
    assert validator(float("inf"))
    assert not validator(99)
    assert not validator(100)
    assert not validator(float("-inf"))
    assert not validator("not a number")


def test_greater_than_validator_image_container() -> None:
    """Test the greater than validator with an image container"""
    validator = greater_than_validator(1.5)
    assert str(validator) == "Value(s) must be greater than 1.5"

    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [0.1, -0.9]]))
    assert not validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[1.5, 2.2], [1.7, 90]]))
    assert not validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[1.51, 2.2], [1.7, 90]]))
    assert validator(image_container)


def test_greater_than_equal_to_validator_creator() -> None:
    """Check the greater_than_equal_to_validator creator raises
    errors when start is not a number type"""
    with pytest.raises(TypeError):
        greater_than_equal_to_validator("str")
    with pytest.raises(TypeError):
        greater_than_equal_to_validator([])


def test_greater_than_equal_to_validator() -> None:
    """Test the greater_than_equal_to_validator with some values"""
    validator = greater_than_equal_to_validator(100)
    assert str(validator) == "Value(s) must be greater than or equal to 100"
    assert validator(101)
    assert validator(1000)
    assert validator(float("inf"))
    assert not validator(99)
    assert validator(100)
    assert not validator(float("-inf"))
    assert not validator("not a number")


def test_greater_than_equal_to_validator_image_container() -> None:
    """Test the greater than equal to validator with an image container"""
    validator = greater_than_equal_to_validator(1.5)
    assert str(validator) == "Value(s) must be greater than or equal to 1.5"

    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [0.1, -0.9]]))
    assert not validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[1.5, 2.2], [1.7, 90]]))
    assert validator(image_container)
    image_container = NumpyImageContainer(image=np.array([[1.51, 2.2], [1.7, 90]]))
    assert validator(image_container)


def test_from_list_validator_creator() -> None:
    """The the from list validator creation"""
    with pytest.raises(TypeError):
        from_list_validator("foo")
    with pytest.raises(TypeError):
        from_list_validator(1)
    with pytest.raises(TypeError):
        from_list_validator({})


def test_from_list_validator() -> None:
    """Test the from list validator"""
    validator = from_list_validator(["FOO", "BAR", "foo", 2])
    assert str(validator) == "Value must be in ['FOO', 'BAR', 'foo', 2]"
    assert validator("FOO")
    assert validator("BAR")
    assert validator("foo")
    assert validator(2)
    assert not validator("bar")
    assert not validator(["FOO"])
    assert not validator({})
    assert not validator(1)

    validator = from_list_validator(["BAR", "foo", 2], case_insensitive=True)
    assert str(validator) == "Value must be in ['BAR', 'foo', 2] (ignoring case)"
    assert validator("FOO")
    assert validator("foo")
    assert validator("BAR")
    assert validator("bAr")
    assert validator(2)
    assert not validator("baz")
    assert not validator(["FOO"])
    assert not validator({})
    assert not validator(1)


def test_isinstance_validator_creator() -> None:
    """The the isinstance validator creation"""
    isinstance_validator(int)  # ok
    isinstance_validator((int, float, str))  # ok
    with pytest.raises(TypeError):
        isinstance_validator(1)
    with pytest.raises(TypeError):
        isinstance_validator("foo")
    with pytest.raises(TypeError):
        isinstance_validator([])
    with pytest.raises(TypeError):
        isinstance_validator({})


def test_isinstance_validator() -> None:
    """Test the isinstance_validator"""
    validator = isinstance_validator(str)
    assert str(validator) == "Value must be of type str"
    assert validator("foo")
    assert not validator([])
    assert not validator({})
    assert not validator(1)
    assert not validator(2.0)

    validator = isinstance_validator(int)
    assert str(validator) == "Value must be of type int"
    assert validator(1)
    assert not validator("foo")
    assert not validator(1.0)
    assert not validator([])
    assert not validator({})

    validator = isinstance_validator(BaseImageContainer)
    assert str(validator) == "Value must be of type BaseImageContainer"
    image_container = NumpyImageContainer(image=np.array([[-0.5, 0.2], [0.1, -0.9]]))
    assert validator(image_container)
    assert not validator("foo")
    assert not validator(1)
    assert not validator([])

    validator = isinstance_validator((float, int))
    assert str(validator) == "Value must be of type float or int"
    assert validator(1.0)
    assert validator(2)
    assert not validator([1])
    assert not validator([1, 2])
    assert not validator("foo")
    assert not validator(["foo", "bar"])


def test_of_length_validator_creator() -> None:
    """Test the of length validator creator"""
    of_length_validator(1)
    of_length_validator(100)
    with pytest.raises(ValueError):
        of_length_validator(0)
    with pytest.raises(ValueError):
        of_length_validator(-1)
    with pytest.raises(ValueError):
        of_length_validator(100.1)
    with pytest.raises(ValueError):
        of_length_validator("foo")


def test_of_length_validator() -> None:
    """Test the of length validator"""
    validator = of_length_validator(5)
    assert str(validator) == "Value (string or list) must have length 5"
    assert not validator(5)
    assert not validator(1.0)
    assert not validator("foo")
    assert not validator([1, 2, 3])
    assert not validator([1.0, 2.0, 3.0, 4.0])
    assert not validator([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert validator("fooba")
    assert validator(["foobar", "foobar", "foobar", "foobar", "foobar"])
    assert validator([1, 2, 3, 4, 5])
    assert validator([1.0, 2.0, 3.0, 4.0, 5.0])


def test_list_of_type_validator_creator() -> None:
    """Test the list of validator creation"""
    list_of_type_validator(int)  # ok
    list_of_type_validator((int, float, str))  # ok
    with pytest.raises(TypeError):
        list_of_type_validator(1)
    with pytest.raises(TypeError):
        list_of_type_validator("foo")
    with pytest.raises(TypeError):
        list_of_type_validator([])
    with pytest.raises(TypeError):
        list_of_type_validator({})


def test_list_of_type_validator() -> None:
    """Test the list of type validator"""
    validator = list_of_type_validator(str)
    assert str(validator) == "Value must be a list of type str"
    assert validator([])
    assert validator(["foo"])
    assert validator(["foo", "bar"])
    assert not validator([1])
    assert not validator([1, 2])

    validator = list_of_type_validator(int)
    assert str(validator) == "Value must be a list of type int"
    assert validator([])
    assert validator([1])
    assert validator([1, 2])
    assert not validator(["foo"])
    assert not validator(["foo", "bar"])
    assert not validator([1.0, 2.0])

    validator = list_of_type_validator(float)
    assert str(validator) == "Value must be a list of type float"
    assert validator([])
    assert validator([1.0])
    assert validator([1.0, 2.0])
    assert not validator(["foo"])
    assert not validator(["foo", "bar"])
    assert not validator([1, 2])
    assert not validator([1])

    validator = list_of_type_validator((float, int))
    assert str(validator) == "Value must be a list of type float or int"
    assert validator([])
    assert validator([1.0])
    assert validator([1.0, 2.0])
    assert validator([1, 2])
    assert validator([1])
    assert not validator(["foo"])
    assert not validator(["foo", "bar"])


def test_non_empty_list_validator() -> None:
    """Test the non-empty list validator"""
    validator = non_empty_list_or_tuple_validator()
    assert str(validator) == "Value must be a non-empty list"
    assert validator([1, 2, 3])
    assert validator(["foo", "bar"])
    assert not validator(1)
    assert not validator("foo")
    assert not validator([])


def test_regex_validator_creator() -> None:
    """Test the regex validator creator"""
    with pytest.raises(ValueError):
        regex_validator("[")  # bad regex


def test_regex_validator() -> None:
    """Test the regex validator"""
    validator = regex_validator(r"^M*(s|t)$")
    assert str(validator) == "Value must match pattern ^M*(s|t)$"
    assert validator("s")
    assert validator("t")
    assert validator("MMMMMMt")
    assert validator("Ms")
    assert not validator("foo")
    assert not validator("")

    validator = regex_validator(r"^COW|dog|CaT$", case_insensitive=True)
    assert validator("CAT")
    assert validator("cAT")
    assert validator("cOw")
    assert not validator("Cog")
    assert str(validator) == "Value must match pattern ^COW|dog|CaT$ (ignoring case)"

    assert not validator(1)


def test_reserved_string_list_validator_creator() -> None:
    """Test the reserved string list validator creator"""
    with pytest.raises(TypeError):
        reserved_string_list_validator(1)

    with pytest.raises(ValueError):
        reserved_string_list_validator([])

    with pytest.raises(ValueError):
        reserved_string_list_validator(["foo", 1])


def test_reserved_string_list_validator() -> None:
    """Test the reserved string list validator"""
    validator = reserved_string_list_validator(
        strings=["M0", "CONTROL", "LABEL"], delimiter="_"
    )
    assert (
        str(validator)
        == "Value must be a string combination of ['M0', 'CONTROL', 'LABEL'] separated"
        " by '_'"
    )
    assert validator("M0")
    assert validator("CONTROL")
    assert validator("LABEL_CONTROL")
    assert validator("M0_LABEL_CONTROL")
    assert not validator("foo")
    assert not validator("M0_foo")

    validator = reserved_string_list_validator(
        strings=["M0", "CONTROL", "LABEL"], delimiter="_", case_insensitive=True
    )
    assert (
        str(validator)
        == "Value must be a string combination of ['M0', 'CONTROL', 'LABEL'] "
        "separated by '_' (ignoring case)"
    )
    assert validator("m0")
    assert validator("M0")
    assert validator("CONTROL")
    assert validator("coNTrol")
    assert validator("m0_LaBEl_ConTROL")
    assert not validator("foo")
    assert not validator("M0_foo")


def test_for_each_validator_creator() -> None:
    """Test the for each validator creator"""
    with pytest.raises(TypeError):
        for_each_validator(item_validator="not a validator")

    # An uninitialised validator
    with pytest.raises(TypeError):
        for_each_validator(item_validator=range_inclusive_validator)


def test_for_each_validator() -> None:
    """Test the for each validator"""

    validator = for_each_validator(greater_than_validator(0.5))
    assert (
        str(validator)
        == "Must be a list or tuple and for each value in the "
        "list: Value(s) must be greater than 0.5"
    )
    assert validator([0.6, 0.7, 0.8])
    assert validator((0.7, 0.51, 0.6))
    assert not validator([0.5, 0.6, 0.7])
    assert not validator([0.1, 100, 0.9])
    assert not validator((0.7, 0.1, 0.6))
    assert not validator("not a list or tuple")


def test_has_attribute_value_validator_creator() -> None:
    """Test the has_attribute_value_validator creator"""

    # attribute_name must be a string
    with pytest.raises(TypeError):
        has_attribute_value_validator(attribute_name=5, attribute_value=5)


def test_has_attribute_value_validator() -> None:
    """Test the has_attribute_value_validator"""

    validator = has_attribute_value_validator(
        attribute_name="a_property", attribute_value="foobar"
    )
    assert str(validator) == "Value must have an attribute a_property with value foobar"
    # has the correct argument with the correct value
    good_obj = SimpleNamespace(a_property="foobar", b_property=100)
    # has the correct argument with the incorrect value
    bad_obj = SimpleNamespace(a_property="notfoobar", b_property=3.0)
    # has the incorrect argument
    worse_obj = SimpleNamespace(not_a_property="notfoobar", b_property=1.0)

    assert validator(good_obj)
    assert not validator(bad_obj)
    assert not validator(worse_obj)
    # some other objects that won't have the `foobar` attribute
    assert not validator("foobar")
    assert not validator(1)
    assert not validator([])


def test_or_validator_creator() -> None:
    """Test the or_validator creator"""

    with pytest.raises(TypeError):
        or_validator(validators="not a list")

    with pytest.raises(TypeError):
        or_validator(validators=[range_inclusive_validator(0, 1), "not a validator"])

    or_validator(
        validators=[
            range_inclusive_validator(0, 1),
            range_exclusive_validator(100, 200),
        ]
    )


def test_or_validator() -> None:
    """Test the or_validator"""
    validator = or_validator(
        [
            range_inclusive_validator(0, 1),
            range_exclusive_validator(100, 200),
        ]
    )
    assert (
        str(validator)
        == "Value(s) must be between 0 and 1 (inclusive) OR "
        "Value(s) must be between 100 and 200 (exclusive)"
    )
    assert not validator(-0.001)
    assert validator(0)
    assert validator(0.5)
    assert validator(1)
    assert not validator(1.1)
    assert not validator(50)
    assert not validator(100)
    assert validator(100.1)
    assert validator(150)
    assert validator(199.9)
    assert not validator(200)


def test_and_validator_creator() -> None:
    """Test the and_validator creator"""

    with pytest.raises(TypeError):
        and_validator(validators="not a list")

    with pytest.raises(TypeError):
        and_validator(validators=[range_inclusive_validator(0, 1), "not a validator"])

    and_validator(
        validators=[
            range_inclusive_validator(0, 1),
            range_exclusive_validator(100, 200),
        ]
    )


def test_and_validator() -> None:
    """Test the and_validator"""
    validator = and_validator(
        [
            range_inclusive_validator(0, 10),
            range_exclusive_validator(5, 15),
        ]
    )

    assert (
        str(validator)
        == "Value(s) must be between 0 and 10 (inclusive) AND "
        "Value(s) must be between 5 and 15 (exclusive)"
    )

    assert not validator(-0.01)
    assert not validator(0)
    assert not validator(5)
    assert validator(5.1)
    assert validator(9.9)
    assert validator(10)
    assert not validator(10.1)
    assert not validator(15)
    assert not validator(15.1)


def test_shape_validator_creator() -> None:
    """Tests the image_shape_validator creator"""
    test_image = NumpyImageContainer(np.ones((8, 8, 8)))

    with pytest.raises(TypeError):
        shape_validator(images="not a list")

    with pytest.raises(TypeError):
        shape_validator(images=[np.zeros(5), "str", test_image])

    with pytest.raises(TypeError):
        shape_validator(["image_1", "image_2", "image_3"], 4.5)

    shape_validator(["image_1", "image_2", "image_3"])
    shape_validator(("image_1", "image_2", "image_3"))
    shape_validator(["image_1", "image_2", "image_3"], 2)


def test_shape_validator() -> None:
    """Tests the image_shape_validator"""

    image_1 = NumpyImageContainer(np.ones((5, 3, 8)))
    image_2 = NumpyImageContainer(np.ones((5, 3, 8)))
    image_3 = NumpyImageContainer(np.ones((2, 7, 9, 11)))
    image_4 = NumpyImageContainer(np.ones((2, 7, 9, 11)))
    image_5 = NumpyImageContainer(np.ones((5, 3, 8, 10)))
    array_1 = np.ones((5, 3, 8))
    array_2 = np.ones((2, 7, 9, 11))
    float_1 = 17.5
    d = {
        "image_1": image_1,
        "image_2": image_2,
        "image_3": image_3,
        "image_4": image_4,
        "array_1": array_1,
        "array_2": array_2,
        "float_1": float_1,
        "image_5": image_5,
    }
    # matching shapes
    validator = shape_validator(["image_1", "image_2", "array_1"])
    assert validator(d)
    validator = shape_validator(["image_3", "image_4", "array_2"])
    assert validator(d)
    validator = shape_validator(["image_1"])
    assert validator(d)

    # non-matching shapes
    validator = shape_validator(["image_1", "image_3", "array_1"])
    assert not validator(d)

    # no shape attribute
    validator = shape_validator(["float_1"])
    assert not validator(d)

    # with maxdim
    validator = shape_validator(["image_1", "image_2", "array_1", "image_5"])
    assert not validator(d)
    validator = shape_validator(["image_1", "image_2", "array_1", "image_5"], 3)
    assert validator(d)


def test_parameter_validator_valid() -> None:
    """Test the parameter validator with some valid example data"""
    parameter_validator = ParameterValidator(
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(
                non_empty_list_or_tuple_validator(), default_value=[1, 2, 3]
            ),
        }
    )
    assert parameter_validator.validate({"foo": "bar foo bar"}) == {
        "foo": "bar foo bar",
        "bar": [1, 2, 3],
    }  # no ValidationError raised


def test_parameter_validator_valid_with_optional_parameters() -> None:
    """Test the parameter validator with some valid example data
    including a (missing) optional parameter"""
    parameter_validator = ParameterValidator(
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(non_empty_list_or_tuple_validator(), optional=True),
        }
    )
    assert parameter_validator.validate({"foo": "bar foo bar"}) == {
        "foo": "bar foo bar"
    }  # no ValidationError raised


def test_parameter_validator_missing_required() -> None:
    """Test the parameter validator with some valid example data"""
    parameter_validator = ParameterValidator(
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(non_empty_list_or_tuple_validator()),
        }
    )

    with pytest.raises(
        ValidationError,
        match="bar is a required parameter and is not in the input dictionary",
    ):
        parameter_validator.validate({"foo": "bar foo bar"})


def test_parameter_validator_multiple_validators() -> None:
    """Test the parameter validator with multiple validators"""
    parameter_validator = ParameterValidator(
        {
            "a_number": Parameter(
                [range_inclusive_validator(1, 2), from_list_validator([1.5, 1.6])]
            )
        }
    )
    parameter_validator.validate({"a_number": 1.5})  # OK

    with pytest.raises(
        ValidationError,
        match=r"Parameter a_number with value 1.7 does not meet the following "
        r"criterion: Value must be in \[1.5, 1.6\]",
    ):
        parameter_validator.validate({"a_number": 1.7})


def test_parameter_validator_multiple_errors() -> None:
    """Test that multiple errors are correctly reported by the validator"""
    parameter_validator = ParameterValidator(
        {
            "a_number": Parameter(
                [range_inclusive_validator(1, 2), from_list_validator([1.5, 1.6])]
            ),
            "b_number": Parameter(list_of_type_validator(str)),
        }
    )
    parameter_validator.validate({"a_number": 1.5, "b_number": ["foo", "bar"]})  # OK

    with pytest.raises(
        ValidationError,
        match=(
            r"^Parameter a_number with value 0.9 does not meet the following"
            r" criterion: "
            r"Value\(s\) must be between 1 and 2 \(inclusive\)\. Parameter a_number"
            r" with value 0.9 "
            r"does not meet the following criterion: Value must be in \[1.5, 1.6\]\. "
            r"Parameter b_number with value \[1, 2\] does not meet the following"
            r" criterion: "
            r"Value must be a list of type str$"
        ),
    ):
        parameter_validator.validate({"a_number": 0.9, "b_number": [1, 2]})


def test_parameter_validator_bad_error_type() -> None:
    """Test that a TypeError is raised if a bad error type is given to the validator"""

    parameter_validator = ParameterValidator({})
    with pytest.raises(TypeError):
        parameter_validator.validate({}, error_type="foo")
    with pytest.raises(TypeError):
        parameter_validator.validate({}, error_type="foo")

    parameter_validator.validate(
        {}, error_type=ValidationError
    )  # sanity check - should be allowed


def test_parameter_validator_change_error_type() -> None:
    """Test that the appropriate error type is raised when user-specified"""

    parameter_validator = ParameterValidator(
        {"b_number": Parameter(list_of_type_validator(str))}
    )

    with pytest.raises(RuntimeError):
        parameter_validator.validate({}, error_type=RuntimeError)


def test_isfile_validator_creator() -> None:
    """Check the isfile validator raises errors when extension is
    not a string or list of strings
    """
    with pytest.raises(TypeError):
        isfile_validator(1)
        isfile_validator(
            [
                1,
                1,
            ]
        )
        isfile_validator([".ext", ".nii", 1])


def test_isfile_validator() -> None:
    """Tests the isfile validator with some values"""

    with TemporaryDirectory() as temp_dir:
        file_name = os.path.join(temp_dir, "file.txt")
        validator = isfile_validator()
        assert not validator(1)  # non-string
        assert not validator("a string")  # a string that isn't a path
        assert validator(file_name)  # valid filename
        assert not validator(temp_dir)  # just a directory
        assert not validator(
            os.path.join(temp_dir, "a_folder", "file.txt")
        )  # dir doesn't exist
        validator = isfile_validator(extensions=".json")
        assert not validator(file_name)  # wrong extension
        validator = isfile_validator(extensions=[".json", ".txt"])
        assert validator(file_name)  # correct extension
        validator = isfile_validator(must_exist=True)
        assert not validator(file_name)

        f = open(file_name, "w")
        f.write("some text")
        f.close()
        assert validator(file_name)
