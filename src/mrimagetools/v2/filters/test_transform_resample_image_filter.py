# type: ignore
""" Transform Resample image Filter Tests """
# pylint: disable=duplicate-code

import logging

import nibabel as nib
import nilearn as nil
import numpy as np
import numpy.testing
import pytest

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.filters.transform_resample_image_filter import (
    TransformResampleImageFilter,
)
from mrimagetools.v2.utils.filter_validation import validate_filter_inputs
from mrimagetools.v2.utils.resampling import transform_resample_image

logger = logging.getLogger(__name__)

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = NiftiImageContainer(
    nifti_img=nib.Nifti2Image(np.zeros(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
)
TEST_NIFTI_ONES.header.set_xyzt_units("mm", "sec")

INPUT_VALIDATION_DICTIONARY = {
    "image": (False, TEST_NIFTI_ONES, np.ones(TEST_VOLUME_DIMENSIONS), "str", 1.0),
    "target_shape": (
        True,
        TEST_VOLUME_DIMENSIONS,
        (16.0, 16.0, 16.0),
        (1, 2, 3, 4),
        "str",
        1,
        [32, 32, 32],
    ),
    "rotation": (
        True,
        (0.0, 0.0, 0.0),
        1.0,
        (181.0, -181.0, 234.2),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
    "rotation_origin": (True, (1.0, 2.0, 3.0), 1.0, (int(1), int(2), int(3)), "str"),
    "translation": (
        True,
        (1.0, 2.0, 3.0),
        1.0,
        (int(1), int(2), int(3)),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
    "interpolation": (True, "nearest", 1),
}


@pytest.mark.parametrize("validation_data", [INPUT_VALIDATION_DICTIONARY])
def test_transform_resample_image_filter_validate_inputs(validation_data: dict) -> None:
    """Check a FilterInputValidationError is raised when the
    inputs to the TransformResampleImageFilter are incorrect or missing
    """
    validate_filter_inputs(TransformResampleImageFilter, validation_data)


def test_transform_resample_image_filter_mock_data() -> None:
    """Test the transform_resampe_image_filter with some mock data"""

    # Create some synthetic data
    grid = np.mgrid[0:128, 0:128]
    circle = (
        np.sum((grid - np.array([32, 32])[:, np.newaxis, np.newaxis]) ** 2, axis=0)
        < 256
    )
    diamond = (
        np.sum(np.abs(grid - np.array([75, 32])[:, np.newaxis, np.newaxis]), axis=0)
        < 16
    )
    rectangle = (
        np.max(np.abs(grid - np.array([64, 64])[:, np.newaxis, np.newaxis]), axis=0)
        < 16
    )
    image = np.zeros_like(circle)
    image = image + circle + 2.0 * rectangle + 3.0 * diamond + np.eye(128)

    image = numpy.expand_dims(image, axis=2)

    # define world coordinate origin (x,y,z) = (0,0,0) at (i,j,k) = (64,64,1)
    # and 1 voxel == 1mm isotropically
    # therefore according to RAS+:
    source_affine = np.array(
        ((1, 0, 0, -64), (0, 1, 0, -64), (0, 0, 1, -0.5), (0, 0, 0, 1))
    )

    nifti_image = nib.Nifti2Image(image, affine=source_affine)
    nifti_image.header.set_xyzt_units("mm", "sec")

    rotation = (0.0, 0.0, 45.0)
    rotation_origin = tuple(
        np.array(nil.image.coord_transform(75, 32, 0, source_affine)).astype(float)
    )
    # rotation_origin = (0.0, 0.0, 0.0)
    translation = (0.0, 10.0, 0.0)
    target_shape = (64, 64, 1)

    image[
        tuple(
            np.rint(
                nil.image.coord_transform(
                    rotation_origin[0],
                    rotation_origin[1],
                    rotation_origin[2],
                    np.linalg.inv(source_affine),
                )
            ).astype(np.int32)
        )
    ] = 5.0

    # create NiftiImageContainer of this image
    nifti_image_container = NiftiImageContainer(nifti_image)

    xr_obj_filter = TransformResampleImageFilter()
    xr_obj_filter.add_input(
        TransformResampleImageFilter.KEY_IMAGE, nifti_image_container
    )
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_ROTATION, rotation)
    xr_obj_filter.add_input(
        TransformResampleImageFilter.KEY_ROTATION_ORIGIN, rotation_origin
    )
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_TRANSLATION, translation)
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_TARGET_SHAPE, target_shape)

    xr_obj_filter.run()
    new_nifti_container: NiftiImageContainer = xr_obj_filter.outputs[
        TransformResampleImageFilter.KEY_IMAGE
    ]

    ### function called here
    str_nifti, _ = transform_resample_image(
        nifti_image, translation, rotation, rotation_origin, target_shape  # type:ignore
    )

    # data should match
    numpy.testing.assert_array_equal(str_nifti.dataobj, new_nifti_container.image)
    # Affines should match
    numpy.testing.assert_array_equal(str_nifti.affine, new_nifti_container.affine)

    # confirm the voxel_size is calculated correctly.
    numpy.testing.assert_array_equal(
        new_nifti_container.metadata.voxel_size,
        nib.affines.voxel_sizes(new_nifti_container.affine),
    )

    ## Check interpolation is working correctly
    xr_obj_filter = TransformResampleImageFilter()
    xr_obj_filter.add_input(
        TransformResampleImageFilter.KEY_IMAGE, nifti_image_container
    )
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_ROTATION, rotation)
    xr_obj_filter.add_input(
        TransformResampleImageFilter.KEY_ROTATION_ORIGIN, rotation_origin
    )
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_TRANSLATION, translation)
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_TARGET_SHAPE, target_shape)
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_INTERPOLATION, "nearest")

    xr_obj_filter.run()

    numpy.testing.assert_array_equal(
        np.unique(xr_obj_filter.outputs["image"].image), (0, 1, 2, 3)
    )
