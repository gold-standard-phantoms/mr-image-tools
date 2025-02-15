"""Pipeline to combine fuzzy masks into a single segmentation mask"""

import logging
from typing import Optional

import nibabel as nib

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.filters.combine_fuzzy_masks_filter import CombineFuzzyMasksFilter
from mrimagetools.v2.filters.json_loader import JsonLoaderFilter
from mrimagetools.v2.filters.nifti_loader import NiftiLoaderFilter
from mrimagetools.v2.validators.schemas.index import load_schemas

logger = logging.getLogger(__name__)


def combine_fuzzy_masks(
    params_filename: str, output_filename: Optional[str] = None
) -> BaseImageContainer:
    """Combines fuzzy masks into a single segmentation mask image.

    :param params_filename: Path to the combining masks parameter JSON file.
    :type params_filename: str
    :param output_filename: Path to the output combined mask NIFTI image, defaults to None
    :type output_filename: str
    :return: The combined mask, as an image container.
    :rtype: BaseImageContainer
    """
    # load in the params JSON file and validate against the schema
    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", params_filename)
    json_filter.add_input("schema", load_schemas()["combine_masks"])
    json_filter.run()

    # load in the masks, put them in a list
    mask_files = []
    for nifti_filename in json_filter.outputs["mask_files"]:
        nifti_loader = NiftiLoaderFilter()
        nifti_loader.add_input("filename", nifti_filename)
        nifti_loader.run()
        mask_files.append(nifti_loader.outputs["image"])

    combine_masks_filter = CombineFuzzyMasksFilter()

    combine_masks_filter.add_inputs(json_filter.outputs)
    combine_masks_filter.add_input(CombineFuzzyMasksFilter.KEY_FUZZY_MASK, mask_files)
    combine_masks_filter.run()

    # save the file
    if output_filename is not None:
        nib.nifti2.save(
            combine_masks_filter.outputs[
                CombineFuzzyMasksFilter.KEY_SEG_MASK
            ].nifti_image,
            output_filename,
        )

    return combine_masks_filter.outputs[CombineFuzzyMasksFilter.KEY_SEG_MASK]  # type: ignore
