""" NiftiLoaderFilter tests """
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import pytest

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.filters.basefilter import FilterInputValidationError
from mrimagetools.v2.filters.nifti_loader import NiftiLoaderFilter


def test_nifti_loader_input_validation_no_input() -> None:
    """Test all of the NiftiLoader input validation -
    No input filename (but some input so the filter will run)"""

    nifti_loader_filter = NiftiLoaderFilter()
    nifti_loader_filter.add_input("dummy", None)
    with pytest.raises(FilterInputValidationError):
        nifti_loader_filter.run()


def test_nifti_loader_input_validation_non_string_input() -> None:
    """Test all of the NiftiLoader input validation -
    Non-string filename"""

    nifti_loader_filter = NiftiLoaderFilter()

    nifti_loader_filter.add_input("filename", 1)
    with pytest.raises(FilterInputValidationError):
        nifti_loader_filter.run()


def test_nifti_loader_input_validation_bad_nifti_filename() -> None:
    """Test all of the NiftiLoader input validation -
    Bad NIFTI filename"""

    nifti_loader_filter = NiftiLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "file.txt")
        Path(temp_file).touch()
        nifti_loader_filter.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            nifti_loader_filter.run()


def test_nifti_loader_input_validation_missing_nifti_file() -> None:
    """Test all of the NiftiLoader input validation -
    Missing NIFTI file"""

    nifti_loader_filter = NiftiLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        # Missing NIFTI file
        temp_file = os.path.join(temp_dir, "file.nii")
        nifti_loader_filter.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            nifti_loader_filter.run()


def test_nifti_loader() -> None:
    """Test the loading functionality"""

    with TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "image.nii")
        nib.nifti2.save(
            img=nib.Nifti1Image(dataobj=np.zeros([3, 3, 3]), affine=np.eye(4)),
            filename=filename,
        )

        nifti_loader_filter = NiftiLoaderFilter()
        nifti_loader_filter.add_input("filename", filename)
        nifti_loader_filter.run()  # This should run OK

        assert isinstance(nifti_loader_filter.outputs["image"], NiftiImageContainer)
        assert (nifti_loader_filter.outputs["image"].image == np.zeros([3, 3, 3])).all()
