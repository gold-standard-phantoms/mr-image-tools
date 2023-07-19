"""General utilities for inputs/output"""

import logging
from os import PathLike
from typing import Union

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
from nibabel.nifti1 import Nifti1Image, Nifti1Pair
from nibabel.nifti2 import Nifti2Image, Nifti2Pair
from nibabel.spatialimages import HeaderDataError

logger = logging.getLogger(__name__)


def nifti_reader(
    filepath: Union[str, PathLike]
) -> Union[Nifti1Pair, Nifti1Image, Nifti2Pair, Nifti2Image]:
    """Reads in a nifti file and returns the image object

    :param filepath: The filepath to the nifti file
    :return: The nifti image object
    :raises nibabel.filebasedimages.ImageFileError: If the file is not a nifti file
    """
    # Disable the logging for the nibabel package as it may throw a lot of warnings
    # when trying to load a nifti1 file as a nifti2 file, as we explictly handle this
    logging.getLogger("nibabel").setLevel(logging.CRITICAL)
    try:
        nii = nib.nifti2.load(filepath)
        logger.info("Loaded %s as a NIfTI2 file", filepath)
        return nii
    except ImageFileError:
        pass
    except HeaderDataError:
        pass

    logging.getLogger("nibabel").setLevel(logging.WARNING)
    nii = nib.nifti1.load(filepath)
    logger.info("Loaded %s as a NIfTI1 file", filepath)
    return nii
