"""Calculate Apparent Diffusion Coefficient Pipeline"""

import os
from typing import Optional

import nibabel as nib
import numpy as np

from mrimagetools.filters.adc_quantification_filter import AdcQuantificationFilter
from mrimagetools.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.filters.load_bids_filter import LoadBidsFilter
from mrimagetools.utils.general import splitext


def adc_pipeline(dwi_nifti_filename: str, output_dir: Optional[str] = None) -> dict:
    """Loads in a DWI nifti and corresponding bval and bvec files,
    calculates the ADC map for each non-zero bvalue, optionally saves this image
    to disk in BIDS format.

    :param dwi_nifti_filename: Path to the DWI image.
    :type dwi_nifti_filename: str
    :param output_dir: The directory to save to, defaults to None
    :type output_dir: str, optional
    :return: A dictionary containing

        :'image': The ADC image
        :'filenames': The saved image filenames

    :rtype: dict

    For implementation details of how the ADC image is calculated see
    :class:`.AdcQuantificationFilter`.
    """

    dwi_loader = LoadBidsFilter()
    dwi_loader.add_input(LoadBidsFilter.KEY_NIFTI_FILENAME, dwi_nifti_filename)
    bval_filename = splitext(dwi_nifti_filename)[0] + ".bval"
    bvec_filename = splitext(dwi_nifti_filename)[0] + ".bvec"

    # load in the bval file
    bval = np.loadtxt(bval_filename, delimiter=" ").tolist()
    bvec = np.transpose(np.loadtxt(bvec_filename, delimiter=" ")).tolist()

    adc_quantification_filter = AdcQuantificationFilter()
    adc_quantification_filter.add_parent_filter(
        dwi_loader, io_map={LoadBidsFilter.KEY_IMAGE: AdcQuantificationFilter.KEY_DWI}
    )
    adc_quantification_filter.add_inputs(
        {
            AdcQuantificationFilter.KEY_B_VALUES: bval,
            AdcQuantificationFilter.KEY_B_VECTORS: bvec,
        }
    )
    adc_quantification_filter.run()
    output_filenames = {}
    if output_dir is not None:
        adc_base_filename = os.path.join(
            output_dir, os.path.split(splitext(dwi_nifti_filename)[0])[1] + "_ADCmap"
        )
        adc_nifti_filename = adc_base_filename + ".nii.gz"
        adc_json_filename = adc_base_filename + ".json"

        nib.nifti2.save(
            adc_quantification_filter.outputs[
                AdcQuantificationFilter.KEY_ADC
            ].nifti_image,
            adc_nifti_filename,
        )

        BidsOutputFilter.save_json(
            adc_quantification_filter.outputs[
                AdcQuantificationFilter.KEY_ADC
            ].metadata.dict(exclude_none=True),
            adc_json_filename,
        )
        output_filenames = {"nifti": adc_nifti_filename, "json": adc_json_filename}

    return {
        "image": adc_quantification_filter.outputs[AdcQuantificationFilter.KEY_ADC],
        "filenames": output_filenames,
    }
