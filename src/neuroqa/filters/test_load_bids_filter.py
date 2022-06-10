"""Tests for load bids filter"""

import os
from json import load
from shutil import copyfile
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import numpy.testing
import pytest
from asldro.containers.image import NiftiImageContainer
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.utils.filter_validation import validate_filter_inputs

from neuroqa.filters.load_bids_filter import LoadBidsFilter


@pytest.fixture(name="test_data")
def test_data_fixture(tmp_path):
    """Returns a dictionary with data for testing"""
    test_dims = (4, 4, 4)
    test_data = np.ones(test_dims)
    test_nifti = NiftiImageContainer(nib.Nifti1Image(test_data, np.eye(4)))
    test_sidecar = {
        "BidsTextField": "some text",
        "BidsNumericField": 123,
        "BidsArrayField": [1, 0, 0, 0, 0],
    }
    nifti_filename = os.path.join(tmp_path, "data.nii.gz")
    json_filename = os.path.join(tmp_path, "data.json")
    nib.save(test_nifti.nifti_image, nifti_filename)
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
def input_validation_dict_fixture(tmp_path, test_data):
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


def test_load_bids_filter_validate_inputs(validation_data):
    """Checks a FilterInputValidationError is raised when the inputs to the
    LoadBidsFilter are incorrect or missing"""
    validate_filter_inputs(LoadBidsFilter, validation_data)


def test_load_bids_filter_mock_data(test_data):
    """Tests the LoadBidsFilter with some mock data"""
    load_bids_filter = LoadBidsFilter()
    load_bids_filter.add_input("nifti_filename", test_data["nifti_filename"])
    load_bids_filter.run()

    numpy.testing.assert_array_equal(
        load_bids_filter.outputs["image"].image, test_data["nifti"].image
    )

    assert test_data["sidecar"] == load_bids_filter.outputs["image"].metadata
    assert load_bids_filter.inputs["json_filename"] == test_data["sidecar_filename"]


def test_load_bids_filter_empty():
    load_bids_filter = LoadBidsFilter()
    with pytest.raises(FilterInputValidationError):
        load_bids_filter.run()
