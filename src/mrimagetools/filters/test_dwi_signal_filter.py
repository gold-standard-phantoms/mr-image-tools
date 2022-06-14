"""Test DwiSignalFilter"""

import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from mrimagetools.containers.image import NiftiImageContainer
from mrimagetools.filters.dwi_signal_filter import DwiSignalFilter
from mrimagetools.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="test_data")
def test_data_fixture() -> dict:
    """Returns a dictionary with data for testing"""
    b_values = [1]
    b_vectors = [[1, 1, 1]]  # what if 0 values in b_vector

    test_dims = (2, 2, 2, 1)  # shape (x, x, x, len(b_values))
    image = np.arange(8, dtype=np.float32).reshape(test_dims)
    # this is the different attenuation coefficient we simulate
    adc_img_val = np.zeros((2, 2, 2, 3))
    # supposidely you can get adc only with one of the 3D element of image
    # hence why I only used one b_value
    for i in range(0, 3):
        adc_img_val[:, :, :, i] = i
    adc = NiftiImageContainer(
        nib.Nifti1Image(
            adc_img_val,
            affine=np.eye(4),
        )
    )  # shape (x, x, x, 3)
    image[:, :, :, 0] = [
        [
            [0.04978706836, 0.04978706836],
            [0.04978706836, 0.04978706836],
        ],
        [
            [0.04978706836, 0.04978706836],
            [0.04978706836, 0.04978706836],
        ],
    ]

    return {
        "adc": adc,
        "b_values": b_values,
        "b_vectors": b_vectors,
        "image": image,
    }


@pytest.fixture(name="validation_data")
def validation_data(test_data) -> dict:
    """returns validation data to test input validation"""
    im_wrong_size = NiftiImageContainer(
        nib.Nifti1Image(np.ones((3, 3, 3, 1)), np.eye(4))
    )
    nifti = nib.Nifti1Image(np.ones((3, 3, 3, 5)), np.eye(4))
    b_vectors_wrong_size = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0, 0]]

    return {
        "adc": [
            False,
            test_data["adc"],
            im_wrong_size,
            nifti,
            np.ones((3, 3, 3, 1)),
            1.0,
            4 + 4j,
            "str",
        ],
        "b_values": [
            False,
            test_data["b_values"],
            [100, 200, 300],
            [0, 100, 200, 300],
            1.0,
            4 + 4j,
            "str",
        ],
        "b_vectors": [
            False,
            test_data["b_vectors"],
            b_vectors_wrong_size,
            1.0,
            4 + 4j,
            "str",
        ],
    }


def test_value(test_data) -> None:
    """test formula"""
    dwi_signal_filter = DwiSignalFilter()
    dwi_signal_filter.add_inputs(test_data)
    dwi_signal_filter.run()

    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["image"].image, test_data["image"]
    )
    # check metadata
    assert dwi_signal_filter.outputs["image"].metadata == {
        "ImageFlavor": "DWI",
        "b_values": test_data["b_values"],
        "b_vectors": test_data["b_vectors"],
    }


def test_dwi_signal_filter_validate_inputs(validation_data) -> None:
    """Check a FilterInputValidationError is raised when the inputs to the
    DwiSignalFilter are incorrect or missing"""

    validate_filter_inputs(DwiSignalFilter, validation_data)
