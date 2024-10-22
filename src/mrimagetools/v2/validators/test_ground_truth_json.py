""" Tests for the ground truth JSON validator """
import pytest
from jsonschema.exceptions import ValidationError

from mrimagetools.v2.validators.ground_truth_json import validate_input

pytest.skip(allow_module_level=True)


def test_valid_ground_truth_json() -> None:
    """Test valid grouth truth raises no error"""
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    validate_input(input_dict=json)


def test_ground_truth_typos_a() -> None:
    """Test ground with typos in required fields raises error"""
    json = {
        "quantties": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_typos_b() -> None:
    """Test ground with typos in required fields raises error"""
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "seg": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_typos_c() -> None:
    """Test ground with typos in required fields raises error"""
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "params": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_json_missing_quantities() -> None:
    """Test missing 'quantities' property raises error"""
    json = {
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_json_missing_segmentation() -> None:
    """Test missing 'segmentation' property raises error"""
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_missing_parameters() -> None:
    """Test missing 'parameters' property raises error"""
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_missing_units() -> None:
    """Test missing 'parameters' property raises error"""
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)
