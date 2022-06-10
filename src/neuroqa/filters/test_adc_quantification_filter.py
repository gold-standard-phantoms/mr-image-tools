"""Test for AdcQuantificationFilter"""

import nibabel as nib
import numpy as np
import numpy.testing
import pytest
from asldro.containers.image import NiftiImageContainer
from asldro.utils.filter_validation import validate_filter_inputs

from neuroqa.filters.adc_quantification_filter import AdcQuantificationFilter


@pytest.fixture(name="test_data")
def test_data_fixture():
    """Returns a dictionary with data for testing"""
    test_dims = (4, 4, 1)
    test_seg_mask = np.arange(16).reshape(test_dims)
    adc = test_seg_mask * (3e-3 / 15)  # scale between 0 and 3e-3 mm^2/s
    b_values = [0, 500, 500, 1000, 1000]
    b_vectors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
    dwi = NiftiImageContainer(
        nib.Nifti1Image(
            np.stack([np.exp(-b_val * adc) for b_val in b_values], axis=3),
            affine=np.eye(4),
        ),
        metadata={
            "series_type": "structural",
            "modality": "DWI",
            "series_number": 1,
        },
    )
    return {
        "dwi": dwi,
        "b_values": b_values,
        "b_vectors": b_vectors,
        "adc": adc,
    }


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture(test_data):
    """Returs a dictionary with data for input validation"""
    im_wrong_size = NiftiImageContainer(
        nib.Nifti1Image(np.ones((3, 3, 3, 1)), np.eye(4))
    )
    nifti = nib.Nifti1Image(np.ones((3, 3, 3, 1)), np.eye(4))
    b_vectors_wrong_size = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    return {
        "dwi": [
            False,
            test_data["dwi"],
            im_wrong_size,
            nifti,
            np.ones((3, 3, 3, 1)),
            1.0,
            "str",
        ],
        "b_values": [
            False,
            test_data["b_values"],
            [100, 200, 300],
            [0, 100, 200, 300],
            1.0,
            "str",
        ],
        "b_vectors": [False, test_data["b_vectors"], b_vectors_wrong_size, 1.0, "str"],
    }


def test_adc_quantification_filter_validate_inputs(validation_data):
    """Check a FilterInputValidationError is raised when the inputs to the
    AdcQuantificationFilter are incorrect or missing"""

    validate_filter_inputs(AdcQuantificationFilter, validation_data)


def test_adc_quantification_filter_mock_data(test_data):
    """Test the AdcQuantificationFilter with some mock data"""
    adc_quantification_filter = AdcQuantificationFilter()
    adc_quantification_filter.add_inputs(test_data)
    adc_quantification_filter.run()

    numpy.testing.assert_array_almost_equal(
        adc_quantification_filter.outputs["adc"].image[:, :, :, 0], test_data["adc"]
    )
    numpy.testing.assert_array_almost_equal(
        adc_quantification_filter.outputs["adc"].image[:, :, :, 1], test_data["adc"]
    )
    # check metadata
    assert adc_quantification_filter.outputs["adc"].metadata == {
        "modality": "ADCmap",
        "Quantity": "ADC",
        "Units": "mm^2/s",
        "ImageType": ["DERIVED", "PRIMARY", "ADCmap", "None"],
        "series_type": "structural",
        "series_number": 1,
        "b_values": test_data["b_values"],
        "b_vectors": test_data["b_vectors"],
    }

    # try reversing the order of the data
    test_data["b_values"].reverse()
    test_data["b_vectors"].reverse()
    test_data["dwi"].image = np.flip(test_data["dwi"].image, axis=3)
    adc_quantification_filter = AdcQuantificationFilter()
    adc_quantification_filter.add_inputs(test_data)
    adc_quantification_filter.run()
