""" CombineTimeSeriesFilter tests """
import os
from collections.abc import Mapping, MutableMapping
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import numpy.testing
import pytest
from nibabel.nifti1 import Nifti1Image

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.basefilter import FilterInputValidationError
from mrimagetools.v2.filters.combine_time_series_filter import CombineTimeSeriesFilter
from mrimagetools.v2.utils.io import nifti_reader


@pytest.fixture(name="image_containers")
def fixture_image_containers() -> dict:
    """Return a dict of (nifti) image containers, where the keys are
    valid "image_nnnnn" string where nnnnnn==0 to 9. The number of
    prepending zeros is equal to the index.
    Each container is size 2x3x4. The value of each voxel is
    equal to the index of the container. The indices range from 0-9.
    There are two metadata fields:
    `to_list` is equal to the index of the container.
    `to_singleton` is always equal to 10
    `even` is equal to the index of the container, but only present in
    odd indices.
    `odd` is always equal to 1.5, but is only present in even indices,
    the final image container has an extra `foo` metadata field.
    """
    # create a dummy nifti image for the header
    dummy_nifti = nib.Nifti1Image(dataobj=np.ones((2, 3, 4)), affine=np.eye(4))
    dummy_nifti.header.set_xyzt_units("mm", "sec")
    num_image_containers = 10
    image_containers = {
        f"image_{''.join(['0' for _ in range(i)]) + str(i)}": (
            NiftiImageContainer(
                nifti_img=nib.Nifti1Image(
                    dataobj=np.ones((2, 3, 4)) * i,
                    affine=np.eye(4),
                    header=dummy_nifti.header,
                ),
                metadata=ImageMetadata(
                    repetition_time_preparation=i, m0=10, post_label_delay=i
                ),
            )
            if i % 2 == 0
            else NiftiImageContainer(
                nifti_img=nib.Nifti1Image(
                    dataobj=np.ones((2, 3, 4)) * i,
                    affine=np.eye(4),
                    header=dummy_nifti.header,
                ),
                metadata=ImageMetadata(
                    repetition_time_preparation=i, m0=10, label_efficiency=1.5
                ),
            )
        )
        for i in range(num_image_containers)
    }
    # Have a metadata field that only reside in a single image container
    image_containers["image_0000000009"].metadata.quantity = "foobar"
    return image_containers


def test_combine_time_series_filter_good_input(
    image_containers: Mapping[str, NiftiImageContainer]
):
    """Run the CombineTimeSeriesFilter with valid inputs, and check the
    output image is as expected"""
    a_filter = CombineTimeSeriesFilter()
    a_filter.add_inputs(image_containers)
    a_filter.run()
    # Check the meta-data has been created correctly
    assert a_filter.outputs["image"].metadata == ImageMetadata(
        repetition_time_preparation=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        m0=10,
        post_label_delay=[0, None, 2, None, 4, None, 6, None, 8, None],
        quantity="foobar",
        label_efficiency=1.5,
    )

    # Test the image has been created correctly - checking voxel values
    container = a_filter.outputs["image"]
    assert container.image.shape == (2, 3, 4, 10)
    for i in range(10):
        numpy.testing.assert_array_equal(
            container.image[:, :, :, i], np.ones((2, 3, 4)) * i
        )

    # Check the image_type has been set correctly
    assert container.image_type == image_containers["image_0"].image_type

    # Check the nifti's xyzt_units have been preserved
    assert container.nifti_image.header.get_xyzt_units() == ("mm", "sec")


def test_combine_time_series_filter_repeat_index(
    image_containers: MutableMapping[str, NiftiImageContainer]
):
    """Run the CombineTimeSeriesFilter where we have a repeat input index.
    We should get a FilterInputValidationError error"""
    image_containers["image_7"] = NiftiImageContainer(
        nifti_img=nib.Nifti1Image(dataobj=np.ones((2, 3, 4)) * 7, affine=np.eye(4)),
        metadata=ImageMetadata(repetition_time_preparation=7, m0=10),
    )

    a_filter = CombineTimeSeriesFilter()
    a_filter.add_inputs(image_containers)
    with pytest.raises(FilterInputValidationError):
        a_filter.run()


def test_combine_time_series_filter_missing_index(
    image_containers: MutableMapping[str, NiftiImageContainer]
):
    """Run the CombineTimeSeriesFilter where we have a missing input index.
    We should get a FilterInputValidationError error"""
    image_containers.pop("image_0")
    a_filter = CombineTimeSeriesFilter()
    a_filter.add_inputs(image_containers)
    with pytest.raises(FilterInputValidationError):
        a_filter.run()


def test_combine_time_series_filter_mismatched_image_dimensions(
    image_containers: MutableMapping[str, NiftiImageContainer]
):
    """Run the CombineTimeSeriesFilter where all of the images
    have mismatched dimensions - check we get a FilterInputValidationError"""
    image_containers["image_0000006"].image = np.ones((3, 3, 3))
    a_filter = CombineTimeSeriesFilter()
    a_filter.add_inputs(image_containers)
    with pytest.raises(FilterInputValidationError):
        a_filter.run()


def test_combine_time_series_filter_non_image(
    image_containers: MutableMapping[str, NiftiImageContainer]
):
    """Run the CombineTimeSeriesFilter where one of the image inputs
    is not an imagecontainer - check we get a FilterInputValidationError"""
    image_containers["image_0000006"] = "surprise!!!"  # type: ignore
    a_filter = CombineTimeSeriesFilter()
    a_filter.add_inputs(image_containers)
    with pytest.raises(FilterInputValidationError):
        a_filter.run()


def test_combine_time_series_filter_with_complex_images() -> None:
    """Test the combine time series filter with more advanced images"""
    num_image_containers = 10
    image_containers = {
        f"image_{''.join(['0' for _ in range(i)]) + str(i)}": (
            NiftiImageContainer(
                nifti_img=nib.Nifti1Image(
                    dataobj=np.ones((1, 1, 1)) * (i + 1j), affine=np.eye(4)
                ),
                metadata=ImageMetadata(
                    repetition_time_preparation=i, m0=10, post_label_delay=i
                ),
            )
            if i % 2 == 0
            else NiftiImageContainer(
                nifti_img=nib.Nifti1Image(
                    dataobj=np.ones((1, 1, 1)) * (i + 1j), affine=np.eye(4)
                ),
                metadata=ImageMetadata(
                    repetition_time_preparation=i, m0=10, label_efficiency=1.5
                ),
            )
        )
        for i in range(num_image_containers)
    }
    a_filter = CombineTimeSeriesFilter()
    a_filter.add_inputs(image_containers)
    a_filter.run()

    numpy.testing.assert_array_equal(
        a_filter.outputs["image"].image,
        np.stack(
            [np.ones((1, 1, 1)) * (i + 1j) for i in range(num_image_containers)], axis=3
        ),
    )
    with TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "image.nii.gz")
        nib.nifti2.save(a_filter.outputs["image"].nifti_image, filename)
        loaded_image = nifti_reader(filename)

        numpy.testing.assert_array_equal(
            loaded_image.dataobj,
            np.stack(
                [np.ones((1, 1, 1)) * (i + 1j) for i in range(num_image_containers)],
                axis=3,
            ),
        )
