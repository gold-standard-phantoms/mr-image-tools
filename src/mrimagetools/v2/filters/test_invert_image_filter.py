""" InvertImageFilter tests """
import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mrimagetools.v2.containers.image import NiftiImageContainer, NumpyImageContainer
from mrimagetools.v2.filters.basefilter import FilterInputValidationError
from mrimagetools.v2.filters.invert_image_filter import InvertImageFilter


def test_invert_image_filter_outputs() -> None:
    """Test the invert image filter validator throws appropriate errors"""
    invert_image_filter = InvertImageFilter()

    invert_image_filter.add_input("image", 123)  # wrong image input type
    with pytest.raises(FilterInputValidationError):
        invert_image_filter.run()


def test_invert_image_filter_with_nifti() -> None:
    """Test the invert image filter works correctly with NiftiImageContainer"""
    invert_image_filter = InvertImageFilter()
    array = np.ones(shape=(3, 3, 3, 1), dtype=np.float32)

    img = nib.Nifti2Image(dataobj=array, affine=np.eye(4))
    nifti_image_container = NiftiImageContainer(nifti_img=img)

    invert_image_filter.add_input("image", nifti_image_container)
    invert_image_filter.run()

    assert_array_equal(invert_image_filter.outputs["image"].image, -array)


def test_invert_image_filter_with_numpy() -> None:
    """Test the invert image filter works correctly with NumpyImageContainer"""
    invert_image_filter = InvertImageFilter()
    array = np.ones(shape=(3, 3, 3, 1), dtype=np.float32)

    nifti_image_container = NumpyImageContainer(image=array)

    invert_image_filter.add_input("image", nifti_image_container)
    invert_image_filter.run()

    assert_array_equal(invert_image_filter.outputs["image"].image, -array)
