"""Tests for load bids filter"""

import os
from shutil import copyfile

import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.basefilter import FilterInputValidationError
from mrimagetools.v2.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.v2.filters.load_bids_filter import LoadBidsFilter
from mrimagetools.v2.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="test_data")
def data_fixture(tmp_path) -> dict:
    """Returns a dictionary with data for testing"""
    test_dims = (4, 4, 4)
    test_data = np.ones(test_dims)
    test_nifti = NiftiImageContainer(nib.Nifti1Image(test_data, np.eye(4)))
    test_sidecar = {
        "SeriesDescription": "some text",
        "SeriesNumber": 123,
        "PostLabelingDelay": [1, 0, 0, 0, 0],
    }
    nifti_filename = os.path.join(tmp_path, "data.nii.gz")
    json_filename = os.path.join(tmp_path, "data.json")
    nib.nifti2.save(test_nifti.nifti_image, nifti_filename)
    wrong_ext_filename = os.path.join(tmp_path, "data.gz")
    copyfile(nifti_filename, wrong_ext_filename)

    BidsOutputFilter.save_json(test_sidecar, json_filename)
    return {
        "nifti": test_nifti,
        "nifti_filename": nifti_filename,
        "wrong_ext_filename": wrong_ext_filename,
        "sidecar_filename": json_filename,
        "sidecar": test_sidecar,
    }


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture(tmp_path, test_data) -> dict:
    """Returns a dictionary with data for input validation testing"""

    return {
        "nifti_filename": [
            False,
            test_data["nifti_filename"],
            os.path.join(tmp_path, test_data["wrong_ext_filename"]),
            "a string",
            1,
        ]
    }


def test_load_bids_filter_validate_inputs(validation_data) -> None:
    """Checks a FilterInputValidationError is raised when the inputs to the
    LoadBidsFilter are incorrect or missing"""
    validate_filter_inputs(LoadBidsFilter, validation_data)


def test_load_bids_filter_mock_data(test_data) -> None:
    """Tests the LoadBidsFilter with some mock data"""
    load_bids_filter = LoadBidsFilter()
    load_bids_filter.add_input("nifti_filename", test_data["nifti_filename"])
    load_bids_filter.run()

    numpy.testing.assert_array_equal(
        load_bids_filter.outputs["image"].image, test_data["nifti"].image
    )

    assert (
        ImageMetadata.from_bids(test_data["sidecar"])
        == load_bids_filter.outputs["image"].metadata
    )
    assert load_bids_filter.inputs["json_filename"] == test_data["sidecar_filename"]


def test_load_bids_filter_empty() -> None:
    """test empty filter running"""
    load_bids_filter = LoadBidsFilter()
    with pytest.raises(FilterInputValidationError):
        load_bids_filter.run()
