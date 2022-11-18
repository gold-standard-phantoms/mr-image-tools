"""Tests from GroundTruthParser"""
from copy import deepcopy
from typing import Final, List

import numpy as np
import pytest
from nibabel import Nifti1Image

from mrimagetools.containers.image import NiftiImageContainer, NumpyImageContainer
from mrimagetools.filters.basefilter import FilterInputValidationError
from mrimagetools.filters.ground_truth_parser import GroundTruthParser, Quantity
from mrimagetools.utils.filter_validation import (
    FilterValidationModel,
    FilterValidationModelParameter,
)
from mrimagetools.validators.fields import NiftiDataTypeField, UnitField
from mrimagetools.validators.parameter_model import ParameterModel


@pytest.fixture(name="valid_dict_input")
def fixture_valid_inputs() -> dict:
    """A basic input to the filter.
    Contains two (2x2x2) datasets. The first has values
    set to one, the second has values set to two."""
    data = np.ones((2, 2, 2, 1, 3))
    data[:, :, :, :, 1] *= 2.0

    return {
        "config": {
            "quantities": [
                {"name": "t1", "units": "s", "cast_to": None},
                {"name": "adc", "units": "mm**2/s", "cast_to": None},
                {"name": "segmentation_a", "cast_to": "uint8"},
            ],
            "segmentation_labels": {
                "segmentation_a": {
                    "background": 0,
                    "grey_matter": 1,
                    "white_matter": 2,
                    "csf": 3,
                }
            },
        },
        "image": NiftiImageContainer(nifti_img=Nifti1Image(data, affine=np.eye(4))),
    }


@pytest.fixture(name="complex_dict_input")
def fixture_complex_dict_input() -> dict:
    """A more complex input.
    Contains three (2x2x2x2(t)) datasets. The first has values
    set to one, the second has values set to two, the third
    is set to three, but cast as a uint64.
    There is also a calculated quantity and a set of parameter"""
    data = np.ones((2, 2, 2, 2, 3))
    data[:, :, :, :, 1] *= 2.0
    data[:, :, :, :, 2] *= 3.0
    return {
        "config": {
            "quantities": [
                {"name": "t1", "units": "s"},
                {"name": "adc", "units": "mm**2/s"},
                {"name": "inty", "cast_to": "uint64"},
            ],
            "parameters": {"foo": "bar", "arr": [1, 2, 3]},
        },
        "image": NiftiImageContainer(nifti_img=Nifti1Image(data, affine=np.eye(4))),
    }


def test_basic_validation_and_run(valid_dict_input: dict) -> None:
    """Test the filter runs with some basic inputs. Test the output is as expected."""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    parser.add_input("config", valid_dict_input["config"])
    parser.run()
    assert parser.parsed_inputs.image == valid_dict_input["image"]
    assert parser.parsed_inputs.config.quantities == [
        Quantity(name="t1", units=UnitField("s"), cast_to=None),
        Quantity(name="adc", units=UnitField("mm**2/s"), cast_to=None),
        Quantity(name="segmentation_a", cast_to=NiftiDataTypeField("uint8")),
    ]
    np.testing.assert_array_equal(parser.outputs["t1"].image, np.ones((2, 2, 2)))
    np.testing.assert_array_equal(parser.outputs["adc"].image, 2.0 * np.ones((2, 2, 2)))
    np.testing.assert_array_equal(
        parser.outputs["segmentation_a"].image, np.ones((2, 2, 2))
    )
    assert (
        parser.outputs["segmentation_labels"]
        == valid_dict_input["config"]["segmentation_labels"]
    )


def test_complex_validation_and_run(complex_dict_input: dict) -> None:
    """Test the filter run with some complex input. Test the output is as expected."""
    parser = GroundTruthParser()
    parser.add_input("image", complex_dict_input["image"])
    parser.add_input("config", complex_dict_input["config"])
    parser.run()
    assert parser.parsed_inputs.image == complex_dict_input["image"]
    assert parser.parsed_inputs.config.quantities == [
        Quantity(name="t1", units=UnitField("s")),
        Quantity(name="adc", units=UnitField("mm**2/s")),
        Quantity(name="inty", cast_to=NiftiDataTypeField("uint64")),
    ]
    np.testing.assert_array_equal(parser.outputs["t1"].image, np.ones((2, 2, 2, 2)))
    np.testing.assert_array_equal(
        parser.outputs["adc"].image, 2.0 * np.ones((2, 2, 2, 2))
    )
    np.testing.assert_array_equal(
        parser.outputs["inty"].image, 3 * np.ones((2, 2, 2, 2))
    )
    # Check the casting
    assert parser.outputs["inty"].image.dtype == np.uint64


def test_basic_validation_with_parameters(valid_dict_input: dict) -> None:
    """Test the filter runs with some basic inputs and parameters"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    parser.add_input(
        "config",
        {
            "quantities": valid_dict_input["config"]["quantities"],
            "parameters": {"foo": "bar"},
        },
    )
    parser.run()
    assert parser.parsed_inputs.config.parameters == {"foo": "bar"}


def test_parameter_validation(valid_dict_input: dict) -> None:
    """Test that we can define and use a parameter validation model"""

    class ParaModel(ParameterModel):
        """Simple model"""

        a: int
        b: str

    parser = GroundTruthParser(parameter_model=ParaModel)
    parser.add_inputs(valid_dict_input)
    with pytest.raises(FilterInputValidationError):
        parser.run()  # missing all required parameters

    valid_dict_input["config"]["parameters"] = {"a": "bar", "b": "foo"}
    parser = GroundTruthParser(parameter_model=ParaModel)
    parser.add_inputs(valid_dict_input)
    with pytest.raises(FilterInputValidationError):
        parser.run()  # a cannot be coerced into an int

    valid_dict_input["config"]["parameters"] = {"a": 1, "b": "foo"}
    parser = GroundTruthParser(parameter_model=ParaModel)
    parser.add_inputs(valid_dict_input)
    parser.run()  # parameters are valid w.r.t. ParaModel


def test_duplicate_quantity_names_validation(valid_dict_input: dict) -> None:
    """duplicate quantity names are not allowed"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    parser.add_input(
        "config",
        {
            "quantities": [
                {"name": "t1", "units": "s", "cast_to": "float64"},
                {"name": "t1", "units": "mm**2/s", "cast_to": "float64"},
                {"name": "t1", "units": "mm", "cast_to": "float64"},
            ]
        },
    )
    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_negative_segmentation_label(valid_dict_input: dict) -> None:
    """test an error is thrown if negative label"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    parser.add_input(
        "config",
        {
            "quantities": [
                {"name": "segmentation_a", "cast_to": "uint8"},
                {"name": "t1", "units": "s", "cast_to": "float64"},
                {"name": "t2", "units": "s", "cast_to": "float64"},
            ],
            "segmentation_labels": {
                "segmentation_a": {
                    "background": -10,
                    "grey_matter": 1,
                    "white_matter": 2,
                    "csf": 3,
                }
            },
        },
    )

    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_duplicate_segmentation_label(valid_dict_input: dict) -> None:
    """test an error is thrown if duplicate lavel"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    parser.add_input(
        "config",
        {
            "quantities": [
                {"name": "segmentation_a", "cast_to": "uint8"},
                {"name": "t1", "units": "s", "cast_to": "float64"},
                {"name": "t2", "units": "s", "cast_to": "float64"},
            ],
            "segmentation_labels": {
                "segmentation_a": {
                    "background": 0,
                    "grey_matter": 0,
                    "white_matter": 2,
                    "csf": 3,
                }
            },
        },
    )

    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_bad_image_type_validation() -> None:
    """The `image` is not an image"""
    parser = GroundTruthParser()
    image_container = "not_an_image_container"
    parser.add_input("image", image_container)
    parser.add_input("config", {"quantities": []})
    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_missing_config_validation(valid_dict_input: dict) -> None:
    """The config is missing"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_bad_shape_image_validation() -> None:
    """Test a 4d image (not 5d) throws an exception"""
    parser = GroundTruthParser()
    parser.add_input("image", NumpyImageContainer(image=np.ones(shape=(2, 2))))
    parser.add_input("config", {"quantities": []})
    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_duplicate_quantity_calculated_names_validation(valid_dict_input: dict) -> None:
    """duplicate names between quantities and calculated quantities are not allowed"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    config: Final[dict] = {
        "quantities": [
            {"name": "t1", "units": "s", "cast_to": "float64"},
            {
                "name": "adc",
                "units": "mm**2/s",
                "cast_to": "float64",
            },  # duplicate
            {"name": "t2", "units": "s", "cast_to": "float64"},
        ],
        "calculated_quantities": [
            {
                "name": "adc",
                "units": "mm**2/s",
                "cast_to": "float64",
                "expression": "t1*1",
            },  # duplicate
        ],
    }

    parser.add_input("config", deepcopy(config))
    with pytest.raises(FilterInputValidationError):
        parser.run()

    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    config["calculated_quantities"][0][
        "name"
    ] = "t2_aug"  # change the name to a unique one
    parser.set_input("config", deepcopy(config))
    parser.run()  # should now run


def test_corresponding_entries_segmentation_label(valid_dict_input: dict) -> None:
    """test an error is raised if no entry is corresponding between the
    segmentation label and the quantity or calculated quantity"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    config: Final[dict] = {
        "quantities": [
            {"name": "t1", "units": "s", "cast_to": "float64"},
            {
                "name": "adc",
                "units": "mm**2/s",
                "cast_to": "float64",
            },
            {"name": "t2", "units": "s", "cast_to": "float64"},
        ],
    }
    parser.add_input("config", deepcopy(config))
    parser.add_input(
        "segmentation_labels",
        {
            "segmentation_a": {
                "background": 0,
                "grey_matter": 1,
                "white_matter": 2,
                "csf": 3,
            }
        },
    )
    with pytest.raises(FilterInputValidationError):
        parser.run()


def test_missing_required_quantities(valid_dict_input: dict) -> None:
    """Test that if any of the `required_quantities` are missing, a validation
    error is raised"""
    parser = GroundTruthParser()
    parser.add_input("image", valid_dict_input["image"])
    config: Final[dict] = {
        "quantities": [
            {"name": "t1", "units": "s", "cast_to": "float64"},
            {
                "name": "adc",
                "units": "mm**2/s",
                "cast_to": "float64",
            },
            {"name": "t2", "units": "s", "cast_to": "float64"},
        ],
        "calculated_quantities": [
            {
                "name": "calc",
                "units": "mm",
                "cast_to": "float64",
                "expression": "t1*1",
            },
        ],
    }
    parser.add_input("config", deepcopy(config))

    # Generate the invalid_configs based on removed one of the required_quantities
    # each time
    invalid_configs: list[dict] = []
    for quantity_type in ("quantities", "calculated_quantities"):
        for index, _ in enumerate(config[quantity_type]):
            conf = deepcopy(config)
            conf[quantity_type].pop(index)
            invalid_configs.append(conf)

    FilterValidationModel(
        filter_type=GroundTruthParser,
        filter_arguments={"required_quantities": ("t1", "t2", "adc", "calc")},
        parameters={
            "image": FilterValidationModelParameter(
                is_optional=False, valid_values=[valid_dict_input["image"]]
            ),
            "config": FilterValidationModelParameter(
                is_optional=False,
                valid_values=[deepcopy(config)],
                invalid_values=invalid_configs,
            ),
        },
    )
