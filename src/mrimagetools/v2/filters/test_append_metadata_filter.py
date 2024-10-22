""" Test for AppendMetadataFilter"""
# pylint: disable=duplicate-code

from copy import deepcopy
from typing import Final

import nibabel as nib
import numpy as np
import pytest

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.append_metadata_filter import AppendMetadataFilter
from mrimagetools.v2.filters.basefilter import FilterInputValidationError

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = nib.nifti2.Nifti2Image(
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
TEST_NIFTI_CON_ONES = NiftiImageContainer(nifti_img=TEST_NIFTI_ONES)

TEST_METADATA: Final[ImageMetadata] = ImageMetadata(echo_time=5.0, quantity="m0")

# test data dictionary,
# each entry, [0] = optional (True/False), [1] = pass, [2:] = fail
INPUT_VALIDATION_DICT = {
    "image": (False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1, "str"),
    "metadata": (False, TEST_METADATA, 1, "str"),
}


def test_append_metadata_filter_validate_inputs() -> None:
    """Check a FilterInputValidationError is raised when the inputs to the
    AppendMetadataFilter are incorrect or missing"""

    test_filter = AppendMetadataFilter()
    test_data = deepcopy(INPUT_VALIDATION_DICT)

    # check with inputs that should pass
    for data_key in test_data:
        test_filter.add_input(data_key, test_data[data_key][1])

    for inputs_key in INPUT_VALIDATION_DICT:
        test_data = deepcopy(INPUT_VALIDATION_DICT)
        test_filter = AppendMetadataFilter()
        is_optional: bool = test_data[inputs_key][0]  # type: ignore

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            test_filter.add_input(data_key, test_data[data_key][1])

        # optional inputs should run without issue
        if is_optional:
            test_filter.run()
        else:
            with pytest.raises(FilterInputValidationError):
                test_filter.run()

        # Try data that should fail
        for test_value in INPUT_VALIDATION_DICT[inputs_key][2:]:
            test_filter = AppendMetadataFilter()
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])
            test_filter.add_input(inputs_key, test_value)

            with pytest.raises(FilterInputValidationError):
                test_filter.run()


def test_append_metadata_filter_new_metadata() -> None:
    """Tests the AppendMetadataFilter with an image that has no metadata to begin with"""

    append_metadata_filter = AppendMetadataFilter()
    append_metadata_filter.add_input("image", TEST_NIFTI_CON_ONES)
    append_metadata_filter.add_input("metadata", TEST_METADATA)
    append_metadata_filter.run()

    # output image's metadata should be equal to TEST_METADATA
    assert append_metadata_filter.outputs["image"].metadata == TEST_METADATA


def test_append_metadata_filter_merge_metadata() -> None:
    """Tests the AppendMetadataFilter with an image that existing metadata"""
    test_image = TEST_NIFTI_CON_ONES.clone()
    test_image.metadata = ImageMetadata(modality="t1")
    new_metadata = ImageMetadata(
        **{
            **test_image.metadata.dict(exclude_none=True),
            **TEST_METADATA.dict(exclude_none=True),
        }
    )
    append_metadata_filter = AppendMetadataFilter()
    append_metadata_filter.add_input("image", test_image)
    append_metadata_filter.add_input("metadata", TEST_METADATA)
    append_metadata_filter.run()

    # output image's metadata should be equal to new_metadata
    assert append_metadata_filter.outputs["image"].metadata == new_metadata


def test_append_metadata_filter_output_image_is_input_image() -> None:
    """Tests that the output image of the AppendMetaDataFilter is a reference of the input
    image"""
    append_metadata_filter = AppendMetadataFilter()
    append_metadata_filter.add_input("image", TEST_NIFTI_CON_ONES)
    append_metadata_filter.add_input("metadata", TEST_METADATA)
    append_metadata_filter.run()

    # output should point to TEST_NIFTI_CON_ONES
    assert append_metadata_filter.outputs["image"] is TEST_NIFTI_CON_ONES
