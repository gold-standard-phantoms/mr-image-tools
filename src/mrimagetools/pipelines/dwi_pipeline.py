"""Diffusion Weighted Image (DWI) Pipeline"""

import json
import os
from argparse import ArgumentParser, Namespace, _SubParsersAction
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from mrimagetools.containers.image import BaseImageContainer, NiftiImageContainer
from mrimagetools.filters.add_complex_noise_filter import AddComplexNoiseFilter
from mrimagetools.filters.combine_time_series_filter import CombineTimeSeriesFilter
from mrimagetools.filters.dwi_signal_filter import DwiSignalFilter
from mrimagetools.filters.ground_truth_parser import GroundTruthParser
from mrimagetools.filters.json_loader import JsonLoaderFilter
from mrimagetools.filters.mri_signal_filter import MriSignalFilter
from mrimagetools.filters.nifti_loader import NiftiLoaderFilter
from mrimagetools.filters.transform_resample_image_filter import (
    TransformResampleImageFilter,
)
from mrimagetools.models.file import GroundTruthFiles
from mrimagetools.utils.cli_types import DirType, FileType
from mrimagetools.utils.general import splitext
from mrimagetools.validators.parameter_model import ParameterModel


class DwiInputParameters(ParameterModel):
    """Input parameter class should be removed in the final
    version of this pipeline"""

    b_values: Sequence[float]
    """List of b-values, one for each dwi volume. One of these must be equal to 0,
    and the length of values should be the same as the number of dwi volumes."""

    b_vectors: Sequence[Sequence[float]]
    """List of b-vectors, one for each dwi volume. The number of vectors must be the
    same as the number of dwi volumes."""

    snr: float
    """Signal to Noise Ratio the higher this float is the less noise will be generated.
    Must be a positive float"""

    repetition_time: float
    """The period of time in msec between the beginning of a pulse sequence and the
    beginning of the succeeding (essentially identical) pulse sequence."""

    echo_time: float
    """Time in ms between the middle of the excitation pulse and the peak of the echo
    produced (kx=0). In the case of segmented k-space, the TE(eff) is the time between
    the middle of the excitation pulse to the peak of the echo that is used to cover
    the center of k-space (i.e.-kx=0, ky=0)."""


class Filename(ParameterModel):
    """Filename class made to write a proper DwiPipelineOutput"""

    nifti_name: Optional[str] = None
    """the path to the nifti file created by the pipeline, is None if no output directory
    was specified"""

    json_name: Optional[str] = None
    """the path to the json file created by the pipeline, is None if no output directory
    was specified"""


class DwiPipelineOutput(ParameterModel):
    """Ouput of the DWI Pipeline"""

    image: BaseImageContainer
    """The image resulting from running the full pipeline"""

    filename: Optional[Filename]
    """The filenames resulting from running the full pipeline, is empty if no output
    directory was specified"""


def dwi_pipeline_processing(
    ground_truth_parser: GroundTruthParser, input_parameters: DwiInputParameters
) -> BaseImageContainer:
    mri_signal_filter = MriSignalFilter()
    dwi_signal_filter = DwiSignalFilter()
    # transform resample creation is inside a loop later on
    # add complex noise creation is inside a loop later on
    combine_time_series_filter = CombineTimeSeriesFilter()

    adc_x = ground_truth_parser.outputs["adc_x"]
    adc_y = ground_truth_parser.outputs["adc_y"]
    adc_z = ground_truth_parser.outputs["adc_z"]
    t1 = ground_truth_parser.outputs["t1"]
    t2 = ground_truth_parser.outputs["t2"]
    m0 = ground_truth_parser.outputs["m0"]
    # segmentation = ground_truth_parser.outputs["segmentation"]

    # running MRI Signal Filter
    mri_signal_filter.add_inputs(
        {
            MriSignalFilter.KEY_T1: t1,
            MriSignalFilter.KEY_T2: t2,
            MriSignalFilter.KEY_M0: m0,
            MriSignalFilter.KEY_ACQ_CONTRAST: "se",
            MriSignalFilter.KEY_ECHO_TIME: input_parameters.echo_time,
            MriSignalFilter.KEY_REPETITION_TIME: input_parameters.repetition_time,
        }
    )
    mri_signal_filter.run()

    s0 = mri_signal_filter.outputs[MriSignalFilter.KEY_IMAGE]
    s0.image = np.squeeze(s0.image)

    # Validate the dimensions of adc_x, adc_y and adc_z before stacking
    if not (adc_x.image.shape == adc_y.image.shape == adc_z.image.shape):
        raise ValueError("All input ADC image datasets must have the same shape")
    if len(adc_x.image.shape) != 3:
        raise ValueError("All input ADC image datasets must be 3D")

    adc_image = np.stack(
        (
            np.asarray(adc_x.image),
            np.asarray(adc_y.image),
            np.asarray(adc_z.image),
        ),
        axis=3,
    )

    # running the dwi signal filter
    adc = NiftiImageContainer(
        nib.Nifti1Image(
            adc_image,
            affine=np.eye(4),
        )
    )
    dwi_signal_filter.add_inputs(
        {
            DwiSignalFilter.KEY_ADC: adc,
            DwiSignalFilter.KEY_B_VALUES: input_parameters.b_values,
            DwiSignalFilter.KEY_B_VECTORS: input_parameters.b_vectors,
            DwiSignalFilter.KEY_S0: s0,  # here s0 got squeezed to be 3D again
        }
    )
    dwi_signal_filter.run()

    dwi = dwi_signal_filter.outputs[dwi_signal_filter.KEY_DWI]
    dwi_3d_dict: dict[str, BaseImageContainer] = {}
    for b_value_index, b_value in enumerate(
        dwi_signal_filter.outputs[dwi_signal_filter.KEY_ATTENUATION].metadata.b_values
    ):
        dwi_3d_dict[f"{b_value}"] = NiftiImageContainer(  # the key is the b_value
            nib.Nifti1Image(dwi.image[:, :, :, b_value_index], affine=np.eye(4))
        )

    # running transform resample image filter
    dwi_3d_dict_transformed: dict[str, BaseImageContainer] = {}
    shape = np.shape(dwi.image[:, :, :, 0])
    for key, value in dwi_3d_dict.items():
        transform_resample_image_filter = TransformResampleImageFilter()

        transform_resample_image_filter.add_inputs(
            {
                TransformResampleImageFilter.KEY_IMAGE: value,
                TransformResampleImageFilter.KEY_TARGET_SHAPE: shape,
            }
        )
        transform_resample_image_filter.run()
        dwi_3d_dict_transformed[
            key  # the key is the b_value
        ] = transform_resample_image_filter.outputs[
            transform_resample_image_filter.KEY_IMAGE
        ]

    # running the add complex noise filter
    final_outputs_dict: dict[str, BaseImageContainer] = {}
    for key, value in dwi_3d_dict_transformed.items():
        add_complex_noise_filter = AddComplexNoiseFilter()
        add_complex_noise_filter.add_inputs(
            {
                AddComplexNoiseFilter.KEY_REF_IMAGE: s0,
                AddComplexNoiseFilter.KEY_IMAGE: value,
                AddComplexNoiseFilter.KEY_SNR: input_parameters.snr,
            }
        )

        add_complex_noise_filter.run()

        final_outputs_dict[key] = add_complex_noise_filter.outputs[
            add_complex_noise_filter.KEY_IMAGE
        ]

    # running the combine time series filter
    for index, (key, value) in enumerate(final_outputs_dict.items()):
        if index == 10000:
            raise ValueError("too  many images to combine")
        combine_time_series_filter.add_input(f"image_0{index:04}", value)

    combine_time_series_filter.run()
    resulting_image: BaseImageContainer = combine_time_series_filter.outputs[
        CombineTimeSeriesFilter.KEY_IMAGE
    ]

    return resulting_image


def dwi_pipeline(
    ground_truth_nii_path: str,
    ground_truth_json_path: str,
    input_parameters_path: str,
    output_dir: Optional[str] = None,
) -> DwiPipelineOutput:
    """Run the DWI pipeline.
    Loads in a 5D Nifti ground truth and JSON parameters file, expects the JSON and the
    ground_truth_path to have the same name calculates the DWI signal for each b values,
    then combine them as a 4D Nifti file

    :param ground_truth_nii_path: High Resolution Ground Truth NIfTI file path
    :param ground_truth_json_path: High Resolution Ground Truth JSON file path
    :param input_parameter_path: DWI pipelines parameters file path
    :param output_dir: The directory to save to (must exist), defaults to None

    :return: A DwiPipelineOutput object with the following entries:

        :'image':
        :'filename:
    """
    # loading input parameters
    with open(input_parameters_path, encoding="utf-8") as f_obj:
        input_parameters = DwiInputParameters(**json.load(f_obj))

    ground_truth = GroundTruthFiles(
        nii_file=Path(ground_truth_nii_path), json_file=Path(ground_truth_json_path)
    )
    # loading nifti
    nifti_loader = NiftiLoaderFilter()
    nifti_loader.add_input("filename", ground_truth.nii_file.as_posix())
    nifti_loader.run()
    ground_truth_nifti = nifti_loader.outputs["image"]

    # extracting json
    json_loader = JsonLoaderFilter()
    json_loader.add_input("filename", ground_truth.json_file.as_posix())
    json_loader.add_input("root_object_name", "metadata")
    json_loader.run()
    inputs_json_file = json_loader.outputs["metadata"]

    # creating filters
    ground_truth_parser = GroundTruthParser()
    # running Ground Truth Parser
    ground_truth_parser.add_inputs(
        {
            GroundTruthParser.KEY_CONFIG: inputs_json_file,
            "image": ground_truth_nifti,
        }
    )

    ground_truth_parser.run()
    resulting_image = dwi_pipeline_processing(
        ground_truth_parser=ground_truth_parser, input_parameters=input_parameters
    )
    # saving if a name was given
    if output_dir is not None:
        dwi_base_filename = os.path.join(
            output_dir, os.path.split(splitext(ground_truth_nii_path)[0])[1] + "_DWI"
        )
        dwi_nifti_filename = dwi_base_filename + ".nii.gz"
        dwi_json_filename = dwi_base_filename + ".json"

        nib.save(
            resulting_image.as_nifti().nifti_image,
            dwi_nifti_filename,
        )

        # return a dictionnary with the image
        output_filenames_returned = Filename(
            nifti_name=dwi_nifti_filename,
            json_name=dwi_json_filename,
        )

        output = DwiPipelineOutput(
            image=resulting_image, filename=output_filenames_returned
        )
        return output

    output = DwiPipelineOutput(image=resulting_image)
    return output


def add_cli_arguments_to(parser: ArgumentParser) -> None:
    """Add the CLI arguments for the DWI pipeline to the supplied (sub)parser,
    and assign any execution logic."""
    parser.add_argument(
        "hrgt_nii_path",
        type=FileType(extensions=[".nii", ".nii.gz"], should_exist=True),
        help=(
            "The path to the HRGT image. Must be a NIFTI or gzipped NIFTI"
            " with extension .nii or .nii.gz. The image data can either be integer, or"
            " floatingpoint. For floating point data voxel values will be rounded to"
            " the nearest integer whendefining which region type is in a voxel."
        ),
    )
    parser.add_argument(
        "hrgt_json_path",
        type=FileType(extensions=["json"], should_exist=True),
        help="The path to the HRGT parameter file (JSON)",
    )
    parser.add_argument(
        "dwi_params_path",
        type=FileType(extensions=["json"], should_exist=True),
        help="The path to the DWI parameters",
    )
    parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help=(
            "The directory to output to. Will create 'ORIGINALFILENAME_DWI.nii.gz' "
            " and corresponding JSON files. "
            " The directory must exist. "
            " Will overwrite any existing files with the same names."
        ),
    )

    def parsing_func(args: Namespace) -> None:
        dwi_pipeline(
            ground_truth_nii_path=args.hrgt_nii_path,
            ground_truth_json_path=args.hrgt_json_path,
            input_parameters_path=args.dwi_params_path,
            output_dir=args.output_dir,
        )

    parser.set_defaults(func=parsing_func)
