"""Tests for the DWI pipeline"""
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from mrimagetools.pipelines.dwi_pipeline import dwi_pipeline

TEST_DATA_INPUT_PARAM = {
    "b_values": [500, 1000],
    "b_vectors": [[1, 0, 1], [1, 1, 1]],
    "snr": 100.0,
    "repetition_time": 2.0,
    "echo_time": 0.02,
}

TEST_DATA_INPUT_JSON = {
    "quantities": [
        {"name": "adc_x", "units": "mm^2/s"},
        {"name": "adc_y", "units": "mm^2/s"},
        {"name": "adc_z", "units": "mm^2/s"},
        {"name": "t1", "units": "s"},
        {"name": "t2", "units": "s"},
        {"name": "m0"},
        {"name": "segmentation", "cast_to": "uint8"},
    ],
    "segmentation_labels": {
        "segmentation": {"background": 0, "grey_matter": 1, "white_matter": 2, "csf": 3}
    },
    "parameters": {"magnetic_field_strength": 3},
}

GROUND_TRUTH_INPUT = np.zeros((3, 3, 3, 1, 7), dtype=np.float64)

# segmentation
GROUND_TRUTH_INPUT[:, :, 0, 0, 6] = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
GROUND_TRUTH_INPUT[:, :, 1, 0, 6] = [[0, 2, 0], [2, 3, 2], [0, 2, 0]]
GROUND_TRUTH_INPUT[:, :, 2, 0, 6] = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]

# m0
GROUND_TRUTH_INPUT[:, :, 0, 0, 5] = [
    [0, 74.62, 0],
    [74.62, 64.73, 74.62],
    [0, 74.62, 0],
]
GROUND_TRUTH_INPUT[:, :, 1, 0, 5] = [
    [0, 64.73, 0],
    [64.73, 68.06, 64.73],
    [0, 64.73, 0],
]
GROUND_TRUTH_INPUT[:, :, 2, 0, 5] = [
    [0, 74.62, 0],
    [74.62, 64.73, 74.62],
    [0, 74.62, 0],
]

# t2
GROUND_TRUTH_INPUT[:, :, 0, 0, 4] = [[0, 0.08, 0], [0.08, 0.11, 0.08], [0, 0.08, 0]]
GROUND_TRUTH_INPUT[:, :, 1, 0, 4] = [[0, 0.11, 0], [0.11, 0.3, 0.11], [0, 0.11, 0]]
GROUND_TRUTH_INPUT[:, :, 2, 0, 4] = [[0, 0.08, 0], [0.08, 0.11, 0.08], [0, 0.08, 0]]

# t1
GROUND_TRUTH_INPUT[:, :, 0, 0, 3] = [[0, 1.33, 0], [1.33, 0.83, 1.33], [0, 1.33, 0]]
GROUND_TRUTH_INPUT[:, :, 1, 0, 3] = [[0, 0.83, 0], [0.83, 3.0, 0.83], [0, 0.83, 0]]
GROUND_TRUTH_INPUT[:, :, 2, 0, 3] = [[0, 1.33, 0], [1.33, 0.83, 1.33], [0, 1.33, 0]]

# 1-1.1e-09 m^2/s
# WM 0.67-0.8E-09
# GM 0.8-1.0E-09
# CSF 3.0-3.4E-09

# adc_z = 1.1*adc_y
GROUND_TRUTH_INPUT[:, :, 0, 0, 2] = [
    [0, 1.0e-09, 0],
    [1.0e-09, 0.8e-09, 1.0e-09],
    [0, 1.0e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, 1, 0, 2] = [
    [0, 0.8e-09, 0],
    [0.8e-09, 3.0e-09, 0.8e-09],
    [0, 0.8e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, 2, 0, 2] = [
    [0, 1.0e-09, 0],
    [1.0e-09, 0.8e-09, 1.0e-09],
    [0, 1.0e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, :, 0, 2] = np.multiply(
    GROUND_TRUTH_INPUT[:, :, :, 0, 2], 1.1e4
)

# adc_y
GROUND_TRUTH_INPUT[:, :, 0, 0, 1] = [
    [0, 1.0e-09, 0],
    [1.0e-09, 0.8e-09, 1.0e-09],
    [0, 1.0e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, 1, 0, 1] = [
    [0, 0.8e-09, 0],
    [0.8e-09, 3.0e-09, 0.8e-09],
    [0, 0.8e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, 2, 0, 1] = [
    [0, 1.0e-09, 0],
    [1.0e-09, 0.8e-09, 1.0e-09],
    [0, 1.0e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, :, 0, 1] = np.multiply(GROUND_TRUTH_INPUT[:, :, :, 0, 2], 1e4)

# adc_x = 0.9*adc_y
GROUND_TRUTH_INPUT[:, :, 0, 0, 0] = [
    [0, 1.0e-09, 0],
    [1.0e-09, 0.8e-09, 1.0e-09],
    [0, 1.0e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, 1, 0, 0] = [
    [0, 0.8e-09, 0],
    [0.8e-09, 3.0e-09, 0.8e-09],
    [0, 0.8e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, 2, 0, 0] = [
    [0, 1.0e-09, 0],
    [1.0e-09, 0.8e-09, 1.0e-09],
    [0, 1.0e-09, 0],
]
GROUND_TRUTH_INPUT[:, :, :, 0, 0] = np.multiply(
    GROUND_TRUTH_INPUT[:, :, :, 0, 0], 0.9e4
)


def test_dwi_pipeline():
    """Test that the pipeline outputs something"""
    with tempfile.TemporaryDirectory() as temp_dir:
        print(temp_dir)
        input_para_name = Path(temp_dir, "input_para_name.json")
        input_name = Path(temp_dir, "input_name.json")
        input_para_name.write_text(json.dumps(TEST_DATA_INPUT_PARAM), encoding="utf-8")
        input_name.write_text(json.dumps(TEST_DATA_INPUT_JSON), encoding="utf-8")
        # create the nifti file
        image = nib.Nifti1Image(
            GROUND_TRUTH_INPUT,
            affine=np.eye(4),
        )
        nib.save(image, Path(temp_dir, "input_name.nii"))
        results = dwi_pipeline(str(input_name), str(input_para_name))

        assert results.image.image is not None
