# type:ignore
# TODO: remove the above line and fix typing errors
""" Tests some user inputs to the model to make sure the validation is performed correctly """
# pylint: disable=redefined-outer-name
from copy import deepcopy

import numpy
import pytest
from numpy.random import default_rng

from mrimagetools.v2.data.filepaths import GROUND_TRUTH_DATA
from mrimagetools.v2.validators.parameters import ValidationError
from mrimagetools.v2.validators.user_parameter_input import (
    ASL,
    ASL_POST_VALIDATOR,
    DEFAULT_GROUND_TRUTH,
    GROUND_TRUTH,
    IMAGE_TYPE_VALIDATOR,
    STRUCTURAL,
    generate_parameter_distribution,
    get_example_input_params,
    validate_input_params,
)


def test_user_input_valid() -> None:
    """Tests a valid set of inputs"""
    d = {
        "label_type": "PASL",
        "label_duration": 2.0,
        "signal_time": 2.5,
        "label_efficiency": 0.8,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 85,
        "m0": 0.7,
        "asl_context": "m0scan control label control label",
        "echo_time": [0, 1, 2, 3, 4],
        "repetition_time": [3, 4.5, 5, 6.4, 1.2],
        "rot_z": [-180, 180, 0, 0, 0],
        "rot_y": [0.0, 180.0, 0.0, 180.0, 1.2],
        "rot_x": [-180.0, 0, 0.2, 3.0, 1.3],
        "transl_x": [-1000, 0.0, 5.6, 6.7, 7.8],
        "transl_y": [0.0, 1000.0, 0.3, 100.6, 2.3],
        "transl_z": [5.6, 1.3, 1.2, 1.3, 1.2],
        "desired_snr": 5.0,
        "acq_matrix": [8, 9, 10],
        "acq_contrast": "se",
        "random_seed": 123_871_263,
        "excitation_flip_angle": 35.6,
        "inversion_flip_angle": 164.0,
        "inversion_time": 1.0,
        "interpolation": "linear",
        "background_suppression": False,
        "output_image_type": "complex",
        "gkm_model": "whitepaper",
    }
    assert d == IMAGE_TYPE_VALIDATOR[ASL].validate(
        d
    )  # the same dictionary should be returned


def test_asl_user_input_defaults_created() -> None:
    """Test default values for the asl image type"""
    correct_defaults = {
        "label_type": "pcasl",
        "asl_context": "m0scan control label",
        "echo_time": {
            "m0scan": 0.01,
            "control": 0.01,
            "label": 0.01,
        },
        "repetition_time": {
            "m0scan": 10.0,
            "control": 5.0,
            "label": 5.0,
        },
        "rot_z": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        "rot_y": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        "rot_x": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        "transl_x": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        "transl_y": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        "transl_z": {"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
        "label_duration": 1.8,
        "signal_time": 3.6,
        "label_efficiency": 0.85,
        "desired_snr": 1000.0,
        "acq_matrix": [64, 64, 40],
        "acq_contrast": "se",
        "random_seed": 0,
        "excitation_flip_angle": 90.0,
        "inversion_flip_angle": 180.0,
        "inversion_time": 1.0,
        "interpolation": "linear",
        "background_suppression": {
            "sat_pulse_time": 4.0,
            "sat_pulse_time_opt": 3.98,
            "pulse_efficiency": "ideal",
            "num_inv_pulses": 4,
            "apply_to_asl_context": ["label", "control"],
        },
        "output_image_type": "magnitude",
        "gkm_model": "full",
    }

    # Validation should include inputs
    assert IMAGE_TYPE_VALIDATOR[ASL].validate({}) == correct_defaults
    # Get the defaults directly
    assert IMAGE_TYPE_VALIDATOR[ASL].get_defaults() == correct_defaults


def test_structural_user_input_defaults_created() -> None:
    """Test default values for the structural image type"""
    correct_defaults = {
        "echo_time": 0.005,
        "repetition_time": 0.3,
        "rot_z": 0.0,
        "rot_y": 0.0,
        "rot_x": 0.0,
        "transl_x": 0.0,
        "transl_y": 0.0,
        "transl_z": 0.0,
        "acq_matrix": [197, 233, 189],
        "acq_contrast": "se",
        "excitation_flip_angle": 90.0,
        "inversion_flip_angle": 180.0,
        "inversion_time": 1.0,
        "desired_snr": 100.0,
        "random_seed": 0,
        "output_image_type": "magnitude",
        "modality": "T1w",
        "interpolation": "linear",
    }

    # Validation should include inputs
    assert IMAGE_TYPE_VALIDATOR[STRUCTURAL].validate({}) == correct_defaults
    # Get the defaults directly
    assert IMAGE_TYPE_VALIDATOR[STRUCTURAL].get_defaults() == correct_defaults


def test_ground_truth_user_input_defaults_created() -> None:
    """Test default values for the ground_truth image type"""
    correct_defaults = {
        "rot_z": 0.0,
        "rot_y": 0.0,
        "rot_x": 0.0,
        "transl_x": 0.0,
        "transl_y": 0.0,
        "transl_z": 0.0,
        "acq_matrix": [64, 64, 40],
        "interpolation": ["linear", "nearest"],
    }

    # Validation should include inputs
    assert IMAGE_TYPE_VALIDATOR[GROUND_TRUTH].validate({}) == correct_defaults
    # Get the defaults directly
    assert IMAGE_TYPE_VALIDATOR[GROUND_TRUTH].get_defaults() == correct_defaults


def test_mismatch_asl_context_array_sizes() -> None:
    """Check that if the length of any of:
    - echo_time
    - repetition_time
    - rot_z
    - rot_y
    - rot_x
    - transl_x
    - transl_y
    - transl_z
    does not match the number of items in asl_context, a ValidationError
    will be raised with an appropriate error message
    """
    good_input = {
        "label_type": "PASL",
        "asl_context": "m0scan control label",
        "echo_time": [0.01, 0.01, 0.01],
        "repetition_time": [10.0, 5.0, 5.0],
        "rot_z": [0.0, 0.0, 0.0],
        "rot_y": [0.0, 0.0, 0.0],
        "rot_x": [0.0, 0.0, 0.0],
        "transl_x": [0.0, 0.0, 0.0],
        "transl_y": [0.0, 0.0, 0.0],
        "transl_z": [0.0, 0.0, 0.0],
    }
    ASL_POST_VALIDATOR.validate(good_input)  # no exception

    for param in [
        "echo_time",
        "repetition_time",
        "rot_x",
        "rot_y",
        "rot_z",
        "transl_z",
        "transl_y",
        "transl_x",
    ]:
        d = deepcopy(good_input)
        d[param] = [0.1, 0.2, 0.3, 0.4]  # wrong number of parameters

        with pytest.raises(
            ValidationError,
            match=(
                f"{param} must be present and have the same number of entries as"
                " asl_context"
            ),
        ):
            ASL_POST_VALIDATOR.validate(d)


def test_generate_parameter_distribution_list_input() -> None:
    """Check that if param is not a dict then it is returned"""
    assert generate_parameter_distribution([0, 1, 2, 3, 4, 5]) == [0, 1, 2, 3, 4, 5]


def test_generate_parameter_distribution_gaussian() -> None:
    """Check that a ValidationError is raised if the parameters for the
    distribution validator are incorrect"""
    # good input
    test_input = {
        "distribution": "gaussian",
        "mean": 4.5,
        "sd": 2.5,
        "seed": 12345,
    }
    out = generate_parameter_distribution(test_input, 8)
    assert len(out) == 8
    assert isinstance(out, list)

    # remove 'distribution'
    d = deepcopy(test_input)
    d.pop("distribution")
    with pytest.raises(ValidationError):
        generate_parameter_distribution(d, 8)

    # remove 'mean'
    d = deepcopy(test_input)
    d.pop("mean")
    with pytest.raises(ValidationError):
        generate_parameter_distribution(d, 8)

    # remove 'sd'
    d = deepcopy(test_input)
    d.pop("sd")
    with pytest.raises(ValidationError):
        generate_parameter_distribution(d, 8)

    # remove 'seed', there shouldn't be an error
    d = deepcopy(test_input)
    d.pop("seed")
    generate_parameter_distribution(d, 8)


def test_generate_parameter_distribution_uniform() -> None:
    """Check that a ValidationError is raised if the parameters for the
    distribution validator are incorrect"""
    # good input
    test_input = {
        "distribution": "uniform",
        "min": 2.5,
        "max": 4.5,
        "seed": 12345,
    }
    out = generate_parameter_distribution(test_input, 8)
    assert len(out) == 8
    assert isinstance(out, list)

    # remove 'distribution'
    d = deepcopy(test_input)
    d.pop("distribution")
    with pytest.raises(ValidationError):
        generate_parameter_distribution(d, 8)

    # remove 'min'
    d = deepcopy(test_input)
    d.pop("min")
    with pytest.raises(ValidationError):
        generate_parameter_distribution(d, 8)

    # remove 'max'
    d = deepcopy(test_input)
    d.pop("max")
    with pytest.raises(ValidationError):
        generate_parameter_distribution(d, 8)

    # remove 'seed', there shouldn't be an error
    d = deepcopy(test_input)
    d.pop("seed")
    generate_parameter_distribution(d, 8)


def test_autogenerate_array_params() -> None:
    """Check that if the array parameters are correctly generated"""
    good_input = get_example_input_params()
    good_input["image_series"][0]["series_parameters"][
        "asl_context"
    ] = "m0scan m0scan control label control label control label"
    good_input["image_series"][0]["series_parameters"]["echo_time"] = [0.01] * 8
    good_input["image_series"][0]["series_parameters"]["repetition_time"] = [5] * 8

    for param in [
        "rot_x",
        "rot_y",
        "rot_z",
        "transl_z",
        "transl_y",
        "transl_x",
    ]:
        good_input["image_series"][0]["series_parameters"][param] = [0.0] * 8
    for n, param in enumerate(
        [
            "rot_x",
            "rot_y",
            "rot_z",
            "transl_z",
            "transl_y",
            "transl_x",
        ]
    ):
        # test gaussian distribution
        d = deepcopy(good_input)
        series_params = d["image_series"][0]["series_parameters"]

        series_params[param] = {
            "distribution": "gaussian",
            "mean": 4.5,
            "sd": 2.5,
            "seed": n,
        }
        rng = default_rng(n)

        d = validate_input_params(d)
        numpy.testing.assert_array_equal(
            d["image_series"][0]["series_parameters"][param],
            rng.normal(4.5, 2.5, 8).round(decimals=4),
        )

        # test uniform distribution
        d = deepcopy(good_input)
        series_params = d["image_series"][0]["series_parameters"]
        series_params[param] = {
            "distribution": "uniform",
            "min": 2.5,
            "max": 4.5,
            "seed": n,
        }
        rng = default_rng(n)
        d = validate_input_params(d)
        numpy.testing.assert_array_equal(
            d["image_series"][0]["series_parameters"][param],
            rng.uniform(2.5, 4.5, 8).round(decimals=4),
        )

    good_input["image_series"][0]["series_parameters"][
        "asl_context"
    ] = "m0scan control label"
    # remove and check defaults
    d = deepcopy(good_input)
    # remove translation parameters so defaults are generated
    _ = [
        d["image_series"][0]["series_parameters"].pop(param)
        for param in [
            "rot_x",
            "rot_y",
            "rot_z",
            "transl_z",
            "transl_y",
            "transl_x",
        ]
    ]
    d["image_series"][0]["series_parameters"].pop("echo_time")
    d["image_series"][0]["series_parameters"].pop("repetition_time")
    d = validate_input_params(d)
    assert d["image_series"][0]["series_parameters"]["echo_time"] == [
        0.01,
        0.01,
        0.01,
    ]
    assert d["image_series"][0]["series_parameters"]["repetition_time"] == [
        10.0,
        5.0,
        5.0,
    ]

    d["image_series"][0]["series_parameters"]["repetition_time"] = {
        "m0scan": 6.479,
        "control": 0.78,
        "label": 3.24,
    }
    d["image_series"][0]["series_parameters"]["echo_time"] = {
        "m0scan": 0.01,
        "control": 0.02,
        "label": 0.5,
    }
    d = validate_input_params(d)
    assert d["image_series"][0]["series_parameters"]["echo_time"] == [
        0.01,
        0.02,
        0.5,
    ]
    assert d["image_series"][0]["series_parameters"]["repetition_time"] == [
        6.479,
        0.78,
        3.24,
    ]

    # check with the example in the series-asl.rst documentation
    new_params = {
        "asl_context": "m0scan m0scan control label control label control label",
        "echo_time": {
            "m0scan": 0.012,
            "control": 0.012,
            "label": 0.012,
        },
        "repetition_time": {
            "m0scan": 10.0,
            "control": 4.5,
            "label": 4.5,
        },
        "rot_x": {
            "distribution": "gaussian",
            "mean": 1.0,
            "sd": 0.1,
            "seed": 12345,
        },
        "transl_y": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0,
            "seed": 12345,
        },
    }
    d = deepcopy(good_input)
    series_params = d["image_series"][0]["series_parameters"]
    d["image_series"][0]["series_parameters"] = {**series_params, **new_params}
    d = validate_input_params(d)


@pytest.fixture
def input_params() -> None:
    """A valid input parameter config"""
    return {
        "global_configuration": {
            "ground_truth": "hrgt_icbm_2009a_nls_3t",
            "image_override": {"m0": 5.0},
            "parameter_override": {"lambda_blood_brain": 0.85},
            "ground_truth_modulate": {
                "t1": {"scale": 0.5},
                "t2": {"offset": 2},
                "m0": {"scale": 2, "offset": 1.5},
            },
        },
        "image_series": [
            {
                "series_type": "asl",
                "series_description": "user description for asl",
                "series_parameters": {
                    "asl_context": "m0scan control label",
                    "label_type": "pcasl",
                    "acq_matrix": [64, 64, 40],
                },
            },
            {
                "series_type": "structural",
                "series_description": "user description for structural scan",
                "series_parameters": {
                    "acq_contrast": "ge",
                    "echo_time": 0.05,
                    "repetition_time": 0.3,
                    "acq_matrix": [256, 256, 128],
                },
            },
            {
                "series_type": "ground_truth",
                "series_description": "user description for ground truth",
                "series_parameters": {"acq_matrix": [64, 64, 40]},
            },
        ],
    }


@pytest.fixture(name="expected_parsed_input")
def fixture_expected_parsed_input() -> None:
    """create fixture for test"""
    return {
        "global_configuration": {
            "ground_truth": {
                "nii_file": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["nii_file"],
                "json_file": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["json_file"],
            },
            "image_override": {"m0": 5.0},
            "parameter_override": {"lambda_blood_brain": 0.85},
            "ground_truth_modulate": {
                "t1": {"scale": 0.5},
                "t2": {"offset": 2},
                "m0": {"scale": 2, "offset": 1.5},
            },
            "subject_label": "001",
        },
        "image_series": [
            {
                "series_type": "asl",
                "series_description": "user description for asl",
                "series_parameters": {
                    "asl_context": "m0scan control label",
                    "label_type": "pcasl",
                    "acq_matrix": [64, 64, 40],
                    "echo_time": [0.01, 0.01, 0.01],
                    "repetition_time": [10.0, 5.0, 5.0],
                    "rot_z": [0.0, 0.0, 0.0],
                    "rot_y": [0.0, 0.0, 0.0],
                    "rot_x": [0.0, 0.0, 0.0],
                    "transl_x": [0.0, 0.0, 0.0],
                    "transl_y": [0.0, 0.0, 0.0],
                    "transl_z": [0.0, 0.0, 0.0],
                    "label_duration": 1.8,
                    "signal_time": 3.6,
                    "label_efficiency": 0.85,
                    "desired_snr": 1000.0,
                    "acq_contrast": "se",
                    "random_seed": 0,
                    "excitation_flip_angle": 90.0,
                    "inversion_flip_angle": 180.0,
                    "inversion_time": 1.0,
                    "interpolation": "linear",
                    "background_suppression": {
                        "sat_pulse_time": 4.0,
                        "sat_pulse_time_opt": 3.98,
                        "pulse_efficiency": "ideal",
                        "num_inv_pulses": 4,
                        "apply_to_asl_context": ["label", "control"],
                    },
                    "output_image_type": "magnitude",
                    "gkm_model": "full",
                },
            },
            {
                "series_type": "structural",
                "series_description": "user description for structural scan",
                "series_parameters": {
                    "echo_time": 0.05,
                    "repetition_time": 0.3,
                    "rot_z": 0.0,
                    "rot_y": 0.0,
                    "rot_x": 0.0,
                    "transl_x": 0.0,
                    "transl_y": 0.0,
                    "transl_z": 0.0,
                    "acq_matrix": [256, 256, 128],
                    "acq_contrast": "ge",
                    "excitation_flip_angle": 90.0,
                    "inversion_flip_angle": 180.0,
                    "inversion_time": 1.0,
                    "desired_snr": 100.0,
                    "random_seed": 0,
                    "output_image_type": "magnitude",
                    "modality": "T1w",
                    "interpolation": "linear",
                },
            },
            {
                "series_type": "ground_truth",
                "series_description": "user description for ground truth",
                "series_parameters": {
                    "acq_matrix": [64, 64, 40],
                    "rot_z": 0.0,
                    "rot_y": 0.0,
                    "rot_x": 0.0,
                    "transl_x": 0.0,
                    "transl_y": 0.0,
                    "transl_z": 0.0,
                    "interpolation": ["linear", "nearest"],
                },
            },
        ],
    }


def test_valid_input_params(input_params: dict, expected_parsed_input: dict) -> None:
    """Test that a valid input parameter file is parsed without
    raising an exception and that the appropriate defaults are inserted"""
    # Should not raise an exception
    parsed_input = validate_input_params(input_params)

    assert parsed_input == expected_parsed_input

    # Also, try changing the ground_truth to the nifti file
    # in the HRGT data (JSON file assumed same name)
    input_params["global_configuration"]["ground_truth"] = GROUND_TRUTH_DATA[
        "hrgt_icbm_2009a_nls_3t"
    ]["nii_file"]
    # Should not raise an exception
    parsed_input = validate_input_params(input_params)

    assert parsed_input == expected_parsed_input

    # Also, try changing the ground_truth to the nifti file/json file
    # in the HRGT data
    input_params["global_configuration"]["ground_truth"] = {
        "nii_file": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["nii_file"],
        "json_file": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["json_file"],
    }
    # Should not raise an exception
    parsed_input = validate_input_params(input_params)

    assert parsed_input == expected_parsed_input


def test_invalid_data_input_params(input_params: dict) -> None:
    """Tests that bad ground_truth data set in the input parameters
    raises appropriate Expections (should always be
    mrimagetools.v2.validators.parameters.ValidationError)"""

    input_params["global_configuration"]["ground_truth"] = "i_dont_exist"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)
    input_params["global_configuration"].pop("ground_truth")

    input_params["global_configuration"]["image_override"] = "a_string"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)

    input_params["global_configuration"]["image_override"] = {"m0": "a_string"}
    with pytest.raises(ValidationError):
        validate_input_params(input_params)
    input_params["global_configuration"].pop("image_override")

    input_params["global_configuration"]["subject_label"] = "invalid_characters!$%^"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)


def test_bad_series_type_input_params(input_params: dict) -> None:
    """Tests that bad series_type data set in the input parameters
    raises appropriate Expections (should always be
    mrimagetools.v2.validators.parameters.ValidationError)"""

    input_params["image_series"][0]["series_type"] = "magic"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)


def test_missing_series_parameters_inserts_defaults(input_params: dict) -> None:
    """Tests that if series_parameters are completely missing for
    an image series, the defaults are inserted"""

    input_params["image_series"][0].pop("series_parameters")

    # The default series parameters should be added
    assert validate_input_params(input_params)["image_series"][0] == {
        "series_type": "asl",
        "series_description": "user description for asl",
        "series_parameters": {
            "asl_context": "m0scan control label",
            "label_type": "pcasl",
            "acq_matrix": [64, 64, 40],
            "echo_time": [0.01, 0.01, 0.01],
            "repetition_time": [10.0, 5.0, 5.0],
            "rot_z": [0.0, 0.0, 0.0],
            "rot_y": [0.0, 0.0, 0.0],
            "rot_x": [0.0, 0.0, 0.0],
            "transl_x": [0.0, 0.0, 0.0],
            "transl_y": [0.0, 0.0, 0.0],
            "transl_z": [0.0, 0.0, 0.0],
            "label_duration": 1.8,
            "signal_time": 3.6,
            "label_efficiency": 0.85,
            "desired_snr": 1000.0,
            "acq_contrast": "se",
            "random_seed": 0,
            "excitation_flip_angle": 90.0,
            "inversion_flip_angle": 180.0,
            "inversion_time": 1.0,
            "interpolation": "linear",
            "background_suppression": {
                "sat_pulse_time": 4.0,
                "sat_pulse_time_opt": 3.98,
                "pulse_efficiency": "ideal",
                "num_inv_pulses": 4,
                "apply_to_asl_context": ["label", "control"],
            },
            "output_image_type": "magnitude",
            "gkm_model": "full",
        },
    }


def test_example_input_params_valid() -> None:
    """Test that the generated example input parameters pass
    the validation (validated internally)"""
    p = get_example_input_params()
    validate_input_params(p)
    assert p["global_configuration"]["ground_truth"] == DEFAULT_GROUND_TRUTH


def test_user_parameter_input_background_suppression() -> None:
    """Tests the background suppression parameters"""
    p = get_example_input_params()
    # check empty "background_suppression"  dict inserts defaults according to the
    # "pulse_times_omitted" validator
    p["image_series"][0]["series_parameters"]["background_suppression"] = {}
    d = validate_input_params(p)
    assert d["image_series"][0]["series_parameters"]["background_suppression"] == {
        "sat_pulse_time": 4.0,
        "pulse_efficiency": "ideal",
        "num_inv_pulses": 4,
        "apply_to_asl_context": ["label", "control"],
    }
    # check "background_suppression" == None inserts the default values
    p["image_series"][0]["series_parameters"]["background_suppression"] = None
    d = validate_input_params(p)
    assert d["image_series"][0]["series_parameters"]["background_suppression"] == {
        "sat_pulse_time": 4.0,
        "sat_pulse_time_opt": 3.98,
        "pulse_efficiency": "ideal",
        "num_inv_pulses": 4,
        "apply_to_asl_context": ["label", "control"],
    }

    # check True "background_suppression" inserts defaults
    p["image_series"][0]["series_parameters"]["background_suppression"] = True
    d = validate_input_params(p)
    assert d["image_series"][0]["series_parameters"]["background_suppression"] == {
        "sat_pulse_time": 4.0,
        "sat_pulse_time_opt": 3.98,
        "pulse_efficiency": "ideal",
        "num_inv_pulses": 4,
        "apply_to_asl_context": ["label", "control"],
    }

    # check False "background_suppression" does nothing
    p["image_series"][0]["series_parameters"]["background_suppression"] = False
    d = validate_input_params(p)
    assert d["image_series"][0]["series_parameters"]["background_suppression"] is False

    # check inversion times supplied
    p["image_series"][0]["series_parameters"]["background_suppression"] = {
        "inv_pulse_times": [0.05, 1.0, 1.5]
    }
    d = validate_input_params(p)
    assert d["image_series"][0]["series_parameters"]["background_suppression"] == {
        "sat_pulse_time": 4.0,
        "pulse_efficiency": "ideal",
        "inv_pulse_times": [0.05, 1.0, 1.5],
        "apply_to_asl_context": ["label", "control"],
    }


def test_user_parameter_input_signal_time_list() -> None:
    """Tests the background suppression parameters"""
    p = get_example_input_params()
    # set "signal_time" to a valid list of times
    lab_dur = 1.8
    pld = [0.25, 0.5, 0.75, 1.0]
    p["image_series"][0]["series_parameters"]["signal_time"] = [
        t + lab_dur for t in pld
    ]
    validate_input_params(p)

    # try numbers that are out of range
    p["image_series"][0]["series_parameters"]["signal_time"] = [-1.0, 0, 1.0]
    with pytest.raises(ValidationError):
        validate_input_params(p)
