"""Pipeline to perform ASL quantification and save CBF map"""
import os
from typing import Optional

import nibabel as nib

from mrimagetools.filters.asl_quantification_filter import AslQuantificationFilter
from mrimagetools.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.filters.json_loader import JsonLoaderFilter
from mrimagetools.filters.load_asl_bids_filter import LoadAslBidsFilter
from mrimagetools.utils.general import splitext

PASL_DEFAULT_PARAMS = {
    "gkm_model": "whitepaper",
    "post_label_delay": 1.8,
    "label_efficiency": 0.98,
    "lambda_blood_brain": 0.9,
    "label_type": "pasl",
    "bolus_cut_off_delay_time": 0.8,
}
CASL_DEFAULT_PARAMS = {
    "gkm_model": "whitepaper",
    "post_label_delay": 1.8,
    "label_efficiency": 0.85,
    "lambda_blood_brain": 0.9,
    "label_type": "pcasl",
    "label_duration": 1.8,
}

DEFAULT_QUANT_PARAMS = {
    "pasl": PASL_DEFAULT_PARAMS,
    "pcasl": CASL_DEFAULT_PARAMS,
    "casl": CASL_DEFAULT_PARAMS,
}
DEFAULT_T1_ARTERIAL_BLOOD = {3: 1.65, 1.5: 1.35}


def asl_quantification(
    asl_nifti_filename: str,
    quant_params_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Performs ASL quantification on an ASL BIDS file, optionally using
    quantification parameters.

    :param asl_nifti_filename: Filename of the ASL NIFTI file. It is assumed
      that there are also corresponding *.json and *context.tsv files within
      the same parent directory
    :type asl_nifti_filename: str
    :param quant_params_filename: Filename of the quantification parameter file
    :type quant_params_filename: str, optional
    :param output_dir: Directory to save the generated NIFTI and JSON sidecar,
      files to, if not supplied then no files will be saved, defaults to None
    :type output_dir: str, optional
    :return: A dictionary with the following entries:

        :image: (NiftiImageContainer) The quantified perfusion rate/CBF map.
        :filenames: (dict) A dictionary containing the filenames of the saved
          NIFTI and JSON sidecar files. Only present if the ``output_dr`` argument
          is not ``None``.
        :quantification_parameters: A dictionay containing entries with the actual
          quantification parameters used. These are based on parameters contained
          in the image,

    :rtype: dict

    The quantification parameters are:

        :QuantificationModel: (string) defaults to "whitepaper"
          (see :class:`.AslQuantificationFilter`)
        :ArterialSpinLabelingType: (string) "PCASL", "CASL" or "PASL"
        :PostLabelingDelay: (float) The post labeling delay in seconds.
        :LabelingDuration: (float) The label duration in seconds (pCASL/CASL only)
        :BolusCutOffDelayTime: (float) The bolus cutoff delay time (PASL only)
        :LabelingEfficiency: (float) The efficiency of the labeling pulse
        :T1ArterialBlood: (float) If not supplied the default value is based on
            the value of the BIDS field "MagneticFieldStrength":

            :1.5 Tesla: 1.35s
            :3.0 Tesla: 1.65s

    Valid ASL BIDS files should contain sufficient information to be able to
    calculate a CBF map. The order of precedence (1 = highest) for parameters
    are:

    1. Supplied quantification parameters
    2. Parameters in the BIDS sidecar
    3. Default parameters

    Default quantification parameters for (p)CASL are:

        :PostLabelingDelay: 1.8
        :LabelingDuration: 1.8
        :LabelingEfficiency: 0.85

    Default quantification parameters for PASL are:

        :PostLabelingDelay: 1.8
        :BolusCutOffDelayTime: 0.8
        :LabelingEfficiency: 0.98

    """
    # load in the asl images and quantification parameters, validate against
    # the schema
    input_quant_params = {}
    if quant_params_filename is not None:
        json_filter = JsonLoaderFilter()
        json_filter.add_input("filename", quant_params_filename)
        #        json_filter.add_input("schema", SCHEMAS["asl_quantification"])
        json_filter.run()
        input_quant_params = json_filter.outputs

    # construct the *_asl.json and *_aslcontext.tsv from the nifti filename
    base_filename = splitext(asl_nifti_filename)
    asl_sidecar_filename = base_filename[0] + ".json"
    aslcontext_filename = base_filename[0] + "context.tsv"

    asl_bids_loader = LoadAslBidsFilter()
    asl_bids_loader.add_input(LoadAslBidsFilter.KEY_IMAGE_FILENAME, asl_nifti_filename)
    asl_bids_loader.add_input(
        LoadAslBidsFilter.KEY_SIDECAR_FILENAME, asl_sidecar_filename
    )
    asl_bids_loader.add_input(
        LoadAslBidsFilter.KEY_ASLCONTEXT_FILENAME, aslcontext_filename
    )
    asl_bids_loader.run()
    label_image = asl_bids_loader.outputs[LoadAslBidsFilter.KEY_LABEL]
    # pull out the required fields from the image's metadata (BIDS sidecar)
    params_from_image = {
        key: getattr(label_image.metadata, key)
        for key in [
            "post_label_delay",
            "label_efficiency",
            "label_duration",
            "label_type",
            "bolus_cut_off_delay_time",
        ]
        if getattr(label_image.metadata, key) is not None
    }

    # the image must have the BIDS field "ArterialSpinLabelingType" otherwise
    # it is not possible to process
    if not params_from_image.get("label_type") in [
        "pcasl",
        "casl",
        "pasl",
    ]:
        raise ValueError(
            "Input ASL image must have BIDS field 'ArterialSpinLabelingType "
            "set to either 'PASL', 'CASL', or 'PCASL'"
            f"\nvalue is {params_from_image.get('ArterialSpinLabelingType')}"
        )

    # merge image derived parameters with the input quantification parameters (if
    # supplied), priority to input parameters.
    quant_params = {
        **params_from_image,
        **input_quant_params,
    }
    # merge with defaults, overriding any that are missing:
    quant_params = {
        **DEFAULT_QUANT_PARAMS[quant_params["label_type"]],
        **quant_params,
    }
    # t1 arterial blood is field strength dependent, determine based on this
    if input_quant_params.get("t1_arterial_blood", None) is None:
        quant_params["t1_arterial_blood"] = DEFAULT_T1_ARTERIAL_BLOOD[
            label_image.metadata.magnetic_field_strength
        ]

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(quant_params)
    asl_quantification_filter.add_inputs(asl_bids_loader.outputs)
    asl_quantification_filter.run()

    # save to pseudo-BIDS format (i.e. NIFTI + JSON sidecar but not with the
    # correct folder structure or filename conventions etc)
    # construct the filename
    output_filenames = {}
    if output_dir is not None:
        output_base_filename = os.path.split(base_filename[0])[1] + "_cbf"
        output_nifti_filename = os.path.join(
            output_dir, output_base_filename + ".nii.gz"
        )
        output_json_filename = os.path.join(output_dir, output_base_filename + ".json")

        nib.save(
            asl_quantification_filter.outputs["perfusion_rate"].nifti_image,
            output_nifti_filename,
        )
        BidsOutputFilter.save_json(
            asl_quantification_filter.outputs["perfusion_rate"].metadata.dict(
                exclude_none=True
            ),
            output_json_filename,
        )
        output_filenames = {
            "nifti": output_nifti_filename,
            "json": output_json_filename,
        }

    return {
        "image": asl_quantification_filter.outputs["perfusion_rate"],
        "filenames": output_filenames,
        "quantification_parameters": quant_params,
    }
