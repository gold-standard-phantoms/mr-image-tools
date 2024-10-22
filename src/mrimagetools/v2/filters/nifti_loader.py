""" NIFTI file loader filter """
import os

import nibabel as nib

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.utils.io import nifti_reader


class NiftiLoaderFilter(BaseFilter):
    """A filter for loading a NIFTI image from a file.

    Must have a single string input named 'filename'.

    Creates a single Image container as an output named
    'image'
    """

    def __init__(self) -> None:
        super().__init__("NiftiLoader")

    def _run(self) -> None:
        """Load the input `filename` using nibabel and create
        a Image container from it. Put this in the output named
        `image`."""

        self.outputs["image"] = NiftiImageContainer(
            nifti_img=nifti_reader(self.inputs["filename"])
        )

    def _validate_inputs(self) -> None:
        """There must be an input named `filename`.
        It must end in .nii or .nii.gz. It must
        point to a existing file."""

        if self.inputs.get("filename", None) is None:
            raise FilterInputValidationError(
                "NiftiLoader filter requires a `filename` input"
            )
        if not isinstance(self.inputs["filename"], str):
            raise FilterInputValidationError(
                "NiftiLoader filter `filename` input must be a string"
            )
        if not self.inputs["filename"].endswith((".nii", ".nii.gz")):
            raise FilterInputValidationError(
                "NiftiLoader filter `filename` must be a .nii or .nii.gz file"
            )

        if not (
            os.path.exists(self.inputs["filename"])
            and os.path.isfile(self.inputs["filename"])
        ):
            raise FilterInputValidationError(
                f"{self.inputs['filename']} does not exist"
            )
