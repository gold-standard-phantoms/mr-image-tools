""" Acquire MRI Image Filter tests """
# pylint: disable=duplicate-code

import logging

import nibabel as nib
import numpy as np
import numpy.testing

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.filters.acquire_mri_image_filter import AcquireMriImageFilter
from mrimagetools.v2.utils.filter_validation import validate_filter_inputs

logger = logging.getLogger(__name__)

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = NiftiImageContainer(
    nifti_img=nib.Nifti2Image(
        np.ones(TEST_VOLUME_DIMENSIONS),
        affine=np.array(
            (
                (1, 0, 0, -16),
                (0, 1, 0, -16),
                (0, 0, 1, -16),
                (0, 0, 0, 1),
            )
        ),
    )
)
TEST_NIFTI_ONES.header.set_xyzt_units("mm", "sec")


# each entry, [0] = optional (True/False), [1] = pass, [2:] = fail
INPUT_DICT = {
    "t1": (False, TEST_NIFTI_ONES, "str", np.ones(TEST_VOLUME_DIMENSIONS)),
    "t2": (False, TEST_NIFTI_ONES, "str", np.ones(TEST_VOLUME_DIMENSIONS)),
    "t2_star": (False, TEST_NIFTI_ONES, "str", np.ones(TEST_VOLUME_DIMENSIONS)),
    "m0": (False, TEST_NIFTI_ONES, "str", np.ones(TEST_VOLUME_DIMENSIONS)),
    "acq_contrast": (False, "ge", 1, TEST_NIFTI_ONES),
    "echo_time": (False, 0.1, "str", TEST_NIFTI_ONES),
    "repetition_time": (False, 0.1, "str", TEST_NIFTI_ONES),
    "snr": (False, 100.0, -1, "str", TEST_NIFTI_ONES),
    "mag_enc": (True, TEST_NIFTI_ONES, "str", np.ones(TEST_VOLUME_DIMENSIONS)),
    "excitation_flip_angle": (False, 0.1, "str", TEST_NIFTI_ONES),
    "inversion_flip_angle": (True, 0.1, "str", TEST_NIFTI_ONES),
    "inversion_time": (True, 0.1, "str", TEST_NIFTI_ONES),
    "image_flavor": (True, "PERFUSION", 1, TEST_NIFTI_ONES),
    "target_shape": (True, (32, 32, 32), 0.1, "str", TEST_NIFTI_ONES),
    "rotation": (True, (5.0, 34.1, 82.56), 0.1, "str", TEST_NIFTI_ONES),
    "rotation_origin": (True, (5.0, 0.0, 0.0), 0.1, "str", TEST_NIFTI_ONES),
    "translation": (True, (1.0, 3.0, 5.56), 0.1, "str", TEST_NIFTI_ONES),
    "interpolation": (True, "nearest", "str", 1),
    "reference_image": (True, TEST_NIFTI_ONES, "str", np.ones(TEST_VOLUME_DIMENSIONS)),
}


def test_acquire_mri_image_filter_validate_inputs_required() -> None:
    """Check a FilterInputValidationError is raised when the inputs
    to the AcquireMriImageFilter are incorrect or missing"""

    validate_filter_inputs(AcquireMriImageFilter, INPUT_DICT)


def test_acquire_mri_image_filter_mocked_filter_run() -> None:
    """Test the AcquireMriImageFilter with some mock data"""
    params = {
        "t1": TEST_NIFTI_ONES,
        "t2": TEST_NIFTI_ONES,
        "t2_star": TEST_NIFTI_ONES,
        "m0": TEST_NIFTI_ONES,
        "acq_contrast": "ge",
        "echo_time": 0.01,
        "repetition_time": 10.0,
        "snr": 0.0,
        "excitation_flip_angle": 90.0,
        "inversion_flip_angle": 0.0,
        "inversion_time": 0.0,
        "image_flavor": "PERFUSION",
        "target_shape": (32, 32, 32),
        "rotation": (0.0, 0.0, 0.0),
        "rotation_origin": (0.0, 0.0, 0.0),
        "translation": (0.0, 0.0, 0.0),
        "reference_image": TEST_NIFTI_ONES,
    }

    acquire_mri_image_filter = AcquireMriImageFilter()
    for data_key in params:
        acquire_mri_image_filter.add_input(data_key, params[data_key])

    acquire_mri_image_filter.run()

    # compare the output image with a manually calculated array
    numpy.testing.assert_array_almost_equal(
        acquire_mri_image_filter.outputs["image"].image,
        np.ones(TEST_VOLUME_DIMENSIONS) * (1 - np.exp(-10)) * np.exp(-0.01),
    )
