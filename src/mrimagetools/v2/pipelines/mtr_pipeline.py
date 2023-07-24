"""Calculate Magnetisation Transfer Ratio Pipeline"""
import os
from typing import Optional

import nibabel as nib

from mrimagetools.v2.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.v2.filters.load_bids_filter import LoadBidsFilter
from mrimagetools.v2.filters.mtr_quantification_filter import MtrQuantificationFilter
from mrimagetools.v2.filters.split_image_filter import SplitImageFilter
from mrimagetools.v2.utils.general import splitext


def mtr_pipeline(
    sat_nifti_filename: str,
    nosat_nifti_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Loads in saturated and non saturated images, calculates the magnetisation
    transfer ratio image, optionally saves this image to disk in BIDS format.

    :param sat_nifti_filename: Path to the image with bound pool saturation
    :type sat_nifti_filename: str
    :param nosat_nifti_filename: Path to the image without bound pool saturation.
        If not supplied it is assumed that the image provided via
        sat_nifti_filename is multi-volume and comprises both unsaturated
        and saturated images in this order. Defaults to None.
    :type nosat_nifti_filename: str
    :param output_dir: The directory to save to, defaults to None
    :type output_dir: str, optional
    :return: A dictionary containing:

        :'image': The MTR image
        :'filenames': The saved image filenames

    :rtype: dict

    For implementation details of how the MTR image is calculated see
    :class:`.MtrQuantificationFilter`.
    """
    sat_loader = LoadBidsFilter()
    sat_loader.add_input(LoadBidsFilter.KEY_NIFTI_FILENAME, sat_nifti_filename)

    mtr_quantification_filter = MtrQuantificationFilter()

    if nosat_nifti_filename is not None:
        nosat_loader = LoadBidsFilter()
        nosat_loader.add_input(LoadBidsFilter.KEY_NIFTI_FILENAME, nosat_nifti_filename)
        mtr_quantification_filter.add_parent_filter(
            nosat_loader,
            io_map={LoadBidsFilter.KEY_IMAGE: MtrQuantificationFilter.KEY_IMAGE_NOSAT},
        )
        mtr_quantification_filter.add_parent_filter(
            sat_loader,
            io_map={LoadBidsFilter.KEY_IMAGE: MtrQuantificationFilter.KEY_IMAGE_SAT},
        )
    else:
        sat_loader.run()  # need to do some checking so run the filter to load
        # the data: there should only be two volumes in the 4th dimension of
        # the loaded image
        if not sat_loader.outputs[LoadBidsFilter.KEY_IMAGE].shape[3] == 2:
            raise ValueError(
                "Only the saturation image has been supplied, this must containtwo"
                " volumes in the 4th dimension\n image shape is"
                f" {sat_loader.outputs[LoadBidsFilter.KEY_IMAGE].shape[3]}"
            )
        split_image_filter = SplitImageFilter()
        split_image_filter.add_parent_filter(sat_loader)
        split_image_filter.add_input(SplitImageFilter.KEY_AXIS, 3)
        split_image_filter.add_input(SplitImageFilter.KEY_INDICES, [1])

        mtr_quantification_filter.add_parent_filter(
            split_image_filter,
            io_map={
                "image_0": MtrQuantificationFilter.KEY_IMAGE_NOSAT,
                "image_1": MtrQuantificationFilter.KEY_IMAGE_SAT,
            },
        )

    mtr_quantification_filter.run()
    output_filenames = {}
    if output_dir is not None:
        mtr_base_filename = os.path.join(
            output_dir, os.path.split(splitext(sat_nifti_filename)[0])[1] + "_MTRmap"
        )
        mtr_nifti_filename = mtr_base_filename + ".nii.gz"
        mtr_json_filename = mtr_base_filename + ".json"
        nib.nifti2.save(
            mtr_quantification_filter.outputs[
                MtrQuantificationFilter.KEY_MTR
            ].nifti_image,
            mtr_nifti_filename,
        )

        BidsOutputFilter.save_json(
            mtr_quantification_filter.outputs[
                MtrQuantificationFilter.KEY_MTR
            ].metadata.dict(exclude_none=True),
            mtr_json_filename,
        )
        output_filenames = {"nifti": mtr_nifti_filename, "json": mtr_json_filename}

    return {
        "image": mtr_quantification_filter.outputs[MtrQuantificationFilter.KEY_MTR],
        "filenames": output_filenames,
    }
