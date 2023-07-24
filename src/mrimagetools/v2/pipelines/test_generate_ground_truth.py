"""Test for generate_ground_truth.py"""
import json
import os
from copy import deepcopy
from tempfile import TemporaryDirectory

import jsonschema
import jsonschema.exceptions
import nibabel as nib
import numpy as np
import pytest

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.filters.ground_truth_parser import GroundTruthConfig
from mrimagetools.v2.pipelines.generate_ground_truth import generate_hrgt
from mrimagetools.v2.validators.schemas.index import load_schemas


@pytest.fixture(name="validation_data")
def input_data_fixture() -> dict:
    """Fixture with test data"""

    return {
        "seg_mask_container": NiftiImageContainer(
            nib.Nifti2Image(
                np.stack([i * np.ones((2, 2, 3), dtype=np.uint16) for i in range(4)]),
                np.eye(4),
            )
        ),
        "seg_mask_float_container": NiftiImageContainer(
            nib.Nifti2Image(
                np.stack(
                    [0.5 * i * np.ones((2, 2, 3), dtype=np.float64) for i in range(7)]
                ),
                np.eye(4),
            )
        ),
        "hrgt_params": {
            "label_values": [0, 1, 2, 3],
            "label_names": ["reg0", "reg1", "reg2", "reg3"],
            "quantities": {
                "quant1": [0.0, 1.0, 2.0, 3.0],
                "quant2": [0.0, 2.0, 4.0, 3.0],
            },
            "units": ["ml/m", "m/s^2"],
            "parameters": {
                "t1_arterial_blood": 1.65,
                "lambda_blood_brain": 0.9,
                "magnetic_field_strength": 3.0,
            },
        },
    }


@pytest.fixture(name="dwi_validation_data")
def dwi_data_fixture() -> dict:
    """Fixture with DWI test data"""

    return {
        "seg_mask_container": NiftiImageContainer(
            nib.Nifti2Image(
                np.stack([i * np.ones((2, 2, 3), dtype=np.uint16) for i in range(4)]),
                np.eye(4),
            )
        ),
        "hrgt_params": {
            "label_values": [0, 1, 2, 3],
            "label_names": ["background", "vial1", "vial2", "vial3"],
            "quantities": {
                "adc_x": [0.0, 0.5e-3, 1.5e-3, 3e-3],
                "adc_y": [0.0, 0.5e-3, 1.5e-3, 3e-3],
                "adc_z": [0.0, 0.5e-3, 1.5e-3, 3e-3],
                "t1": [0.0, 0.4, 0.7, 1.0],
                "t2": [0.0, 0.1, 0.35, 0.7],
                "m0": [0.0, 0.5, 0.75, 1.0],
            },
            "units": ["mm^2/s", "mm^2/s", "mm^2/s", "s", "s", None],
            "parameters": {
                "magnetic_field_strength": 3.0,
            },
        },
    }


def test_dwi_hrgt_params_schema(dwi_validation_data: dict) -> None:
    """Check that the example dwi hrgt_params passes the json schema"""
    jsonschema.validate(
        dwi_validation_data["hrgt_params"], load_schemas()["generate_dwi_hrgt_params"]
    )

    # check it fails when 'magnetic_field_strength' is missing from 'parameters'
    d = deepcopy(dwi_validation_data["hrgt_params"])
    d["parameters"].pop("magnetic_field_strength")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(d, load_schemas()["generate_asl_hrgt_params"])

    # try something that should fail - swap type for one of the arrays
    d = deepcopy(dwi_validation_data["hrgt_params"])
    d["label_names"] = d["label_values"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(d, load_schemas()["generate_dwi_hrgt_params"])


def test_dwi_generate_hrgt(dwi_validation_data: dict) -> None:
    """Test generate_hrgt function with DWI ground truth data"""
    with TemporaryDirectory() as temp_dir:
        json_filename = os.path.join(temp_dir, "hrgt_params.json")
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(dwi_validation_data["hrgt_params"], json_file, indent=4)

        nifti_filename = os.path.join(temp_dir, "seg_mask.nii.gz")
        nib.save(dwi_validation_data["seg_mask_container"].nifti_image, nifti_filename)

        results = generate_hrgt(
            hrgt_params_filename=json_filename,
            seg_mask_filename=nifti_filename,
            schema_name="generate_dwi_hrgt_params",
            output_dir=temp_dir,
        )

        with open(os.path.join(temp_dir, "hrgt.json"), encoding="utf-8") as json_file:
            saved_json = json.load(json_file)

        assert results.config == GroundTruthConfig(
            **saved_json
        )  # should not produce an error


def test_hrgt_params_schema(validation_data: dict) -> None:
    """Check that the example hrgt_params passes the json schema"""
    jsonschema.validate(
        validation_data["hrgt_params"], load_schemas()["generate_asl_hrgt_params"]
    )

    # check it passes when 'lambda_blood_brain' is missing from 'parameters'
    d = deepcopy(validation_data["hrgt_params"])
    d["parameters"].pop("lambda_blood_brain")
    jsonschema.validate(d, load_schemas()["generate_asl_hrgt_params"])

    # try something that should fail - swap type for one of the arrays
    d = deepcopy(validation_data["hrgt_params"])
    d["label_names"] = d["label_values"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(d, load_schemas()["generate_asl_hrgt_params"])


def test_generate_hrgt(validation_data: dict) -> None:
    """Test generate_hrgt function"""
    with TemporaryDirectory() as temp_dir:
        json_filename = os.path.join(temp_dir, "hrgt_params.json")
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(validation_data["hrgt_params"], json_file, indent=4)

        nifti_filename = os.path.join(temp_dir, "seg_mask.nii.gz")
        nib.save(validation_data["seg_mask_container"].nifti_image, nifti_filename)

        results = generate_hrgt(
            hrgt_params_filename=json_filename,
            seg_mask_filename=nifti_filename,
            schema_name="generate_asl_hrgt_params",
            output_dir=temp_dir,
        )

        with open(os.path.join(temp_dir, "hrgt.json"), encoding="utf-8") as json_file:
            saved_json = json.load(json_file)

        assert results.config == GroundTruthConfig(
            **saved_json
        )  # should not produce an error


def test_generate_hrgt_float_seg_mask(validation_data: dict) -> None:
    """Test generate_hrgt function with float seg_mask data"""
    with TemporaryDirectory() as temp_dir:
        json_filename = os.path.join(temp_dir, "hrgt_params.json")
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(validation_data["hrgt_params"], json_file, indent=4)

        nifti_filename = os.path.join(temp_dir, "seg_mask.nii.gz")
        nib.save(
            validation_data["seg_mask_float_container"].nifti_image, nifti_filename
        )

        results = generate_hrgt(
            hrgt_params_filename=json_filename,
            seg_mask_filename=nifti_filename,
            schema_name="generate_asl_hrgt_params",
            output_dir=temp_dir,
        )

        with open(os.path.join(temp_dir, "hrgt.json"), encoding="utf-8") as json_file:
            saved_json = json.load(json_file)

        assert results.config == GroundTruthConfig(
            **saved_json
        )  # should not produce an error
