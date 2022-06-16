"""Tests for MtrQuantificationFilter"""


import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from mrimagetools.containers.image import NiftiImageContainer
from mrimagetools.filters.mtr_quantification_filter import MtrQuantificationFilter
from mrimagetools.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="test_data")
def test_data_fixture() -> dict:
    """Returns a dictionary with data for testing"""
    test_dims = (4, 4, 1)
    test_seg_mask = np.arange(16).reshape(test_dims)
    mtr = test_seg_mask * (100 / 15)  # scale betwen 0 and 100%
    s_nosat = np.ones(test_dims)
    s_sat = s_nosat * (1 - mtr / 100)

    return {
        "image_nosat": NiftiImageContainer(
            nib.Nifti1Image(s_nosat, np.eye(4)),
            metadata={
                "series_type": "structural",
                "modality": "T1w",
                "series_number": 1,
            },
        ),
        "image_sat": NiftiImageContainer(
            nib.Nifti1Image(s_sat, np.eye(4)),
            metadata={
                "series_type": "structural",
                "modality": "T1w",
                "series_number": 1,
            },
        ),
        "image_mtr": NiftiImageContainer(
            nib.Nifti1Image(mtr, np.eye(4)),
            metadata={
                "series_type": "structural",
                "modality": "T1w",
                "series_number": 1,
            },
        ),
    }


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture(test_data) -> dict:

    im_wrong_size = NiftiImageContainer(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)))
    im_wrong_affine = NiftiImageContainer(
        nib.Nifti1Image(np.ones((4, 4, 1)), 3 * np.eye(4))
    )

    return {
        "image_nosat": [
            False,
            test_data["image_nosat"],
            im_wrong_size,
            im_wrong_affine,
            np.ones((4, 4, 1)),
            1.0,
            "str",
        ],
        "image_sat": [
            False,
            test_data["image_sat"],
            im_wrong_size,
            im_wrong_affine,
            np.ones((4, 4, 1)),
            1.0,
            "str",
        ],
    }


def test_mtr_quantification_filter_validate_inputs(validation_data) -> None:
    """Check a FilterInputValidationError is raised when the
    inputs to the MtrQuantificationFilter are incorrect or missing.
    """

    validate_filter_inputs(
        MtrQuantificationFilter,
        validation_data,
    )


def test_mtr_quantification_filter_mock_data(test_data) -> None:
    """Test the MtrQuantificationFilter with some mock data"""

    mtr_quantification_filter = MtrQuantificationFilter()

    mtr_quantification_filter.add_inputs(test_data)
    mtr_quantification_filter.run()

    numpy.testing.assert_array_almost_equal(
        mtr_quantification_filter.outputs["mtr"].image, test_data["image_mtr"].image
    )

    # Check the metadata has been added

    assert mtr_quantification_filter.outputs["mtr"].metadata == {
        "modality": "MTRmap",
        "Quantity": "MTR",
        "Units": "pu",
        "ImageType": ["DERIVED", "PRIMARY", "MTRmap", "None"],
        "series_type": "structural",
        "series_number": 1,
    }
