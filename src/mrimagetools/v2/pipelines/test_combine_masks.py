"""Tests for combine_masks.py"""

import json
import os
from copy import deepcopy
from tempfile import TemporaryDirectory

import jsonschema
import jsonschema.exceptions
import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.pipelines.combine_masks import combine_fuzzy_masks
from mrimagetools.v2.utils.io import nifti_reader
from mrimagetools.v2.validators.schemas.index import load_schemas


@pytest.fixture(name="validation_data")
def input_data_fixture() -> dict:
    """Fixture with test data"""

    mask_data = np.array(
        (
            (1.0, 0.25, 0.50, 0.50),
            (0.25, 0.50, 1.0, 0.0),
            (0.50, 1.0, 0.0, 0.25),
            (0.0, 1.0, 0.25, 0.50),
        )
    )
    return {
        "masks": [
            NiftiImageContainer(nib.Nifti1Image(mask_data[i, :], np.eye(4)))
            for i in range(4)
        ],
        "test_params": {
            "mask_files": ["str", "str", "str"],
            "region_values": [1, 2, 3, 4],
            "region_priority": [4, 3, 2, 1],
            "threshold": 0.25,
        },
    }


def test_combine_masks_params_schema(validation_data: dict) -> None:
    """Check that the example test_params passes the json schema"""
    jsonschema.validate(validation_data["test_params"], load_schemas()["combine_masks"])

    # check it passes when 'threshold' is missing from 'parameters'
    d = deepcopy(validation_data["test_params"])
    d.pop("threshold")

    # try something that should fail - swap type for one of the arrays
    d = deepcopy(validation_data["test_params"])
    d["region_values"] = d["mask_files"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(d, load_schemas()["combine_masks"])


def test_combine_masks_mock_data(validation_data: dict) -> None:
    """test combine_masks function"""
    with TemporaryDirectory() as temp_dir:
        # save the test images
        nifti_filenames = []
        for i, image in enumerate(validation_data["masks"]):
            nifti_filenames.append(os.path.join(temp_dir, f"image_{i}.nii.gz"))
            nib.nifti2.save(image.nifti_image, nifti_filenames[i])

        # put the filenames into the dictionary then save as a json
        validation_data["test_params"]["mask_files"] = nifti_filenames
        json_filename = os.path.join(temp_dir, "params.json")
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(validation_data["test_params"], json_file, indent=4)

        output_filename = os.path.join(temp_dir, "combined_mask.nii.gz")
        results = combine_fuzzy_masks(json_filename, output_filename)
        saved_nifti = nifti_reader(output_filename)

        numpy.testing.assert_array_equal(saved_nifti.dataobj, results.image)
