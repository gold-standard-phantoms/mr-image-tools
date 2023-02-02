"""Diffusion Weighted Image (DWI) Pipeline"""


import json
import logging
import os
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Optional

import nibabel as nib
import numpy as np
from pydantic import root_validator
from typing_extensions import TypeAlias

from mrimagetools.containers.image import BaseImageContainer, NiftiImageContainer
from mrimagetools.filters.add_complex_noise_filter import AddComplexNoiseFilter
from mrimagetools.filters.combine_time_series_filter import CombineTimeSeriesFilter
from mrimagetools.filters.dwi_signal_filter import DwiSignalFilter
from mrimagetools.filters.ground_truth_parser import GroundTruthParser
from mrimagetools.filters.json_loader import JsonLoaderFilter
from mrimagetools.filters.mri_signal_filter import MriSignalFilter
from mrimagetools.filters.nifti_loader import NiftiLoaderFilter
from mrimagetools.filters.phase_magnitude_filter import PhaseMagnitudeFilter
from mrimagetools.models.file import GroundTruthFiles
from mrimagetools.utils.cli_types import DirType, FileType
from mrimagetools.utils.general import splitext
from mrimagetools.validators.parameter_model import ParameterModel

logger = logging.getLogger(__name__)
# A dictionary that maps b_values to corresponding DWI image
DwiDict: TypeAlias = dict[float, BaseImageContainer]


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

    @root_validator(pre=False, skip_on_failure=True)
    def check_b_arrays_same_length(cls, values: dict) -> dict:
        """Check the b_values and b_vectors have the same length"""
        b_values: Sequence = values.get("b_values")  # type: ignore
        b_vectors: Sequence = values.get("b_vectors")  # type: ignore
        if len(b_values) != len(b_vectors):
            raise ValueError(
                f"Length of b_values {b_values} must equal the length of"
                f" b_vectors {b_vectors}"
            )
        return values


class Filename(ParameterModel):
    """Filename class made to write a proper DwiPipelineOutput"""

    nifti_name: Optional[str] = None
    """the path to the nifti file created by the pipeline, is None if no output
    directory was specified"""

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
    logger.info("Creating a DWI DRO using the input parameters: %s", input_parameters)
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

    def equal_shapes(images: Iterable[BaseImageContainer]) -> bool:
        """Returns true if the images have the same shape"""
        iterator = iter(images)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first.shape == x.shape for x in iterator)

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

    if not equal_shapes([adc_x, adc_y, adc_z, t1, t2, m0, s0]):
        raise ValueError("Input images from ground truth should have the same shapes")

    # Use this to check shape consistency
    constant_shape: Final[tuple[int, ...]] = adc_x.shape

    # Validate the dimensions of adc_x, adc_y and adc_z before stacking
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
        nib.Nifti1Image(adc_image, affine=np.eye(4), dtype=adc_image.dtype)
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

    # We should get an output that has the same size 4th dimension as the number
    # b_values/b_vectors
    if dwi_signal_filter.outputs[dwi_signal_filter.KEY_DWI].shape[3] != len(
        input_parameters.b_vectors
    ):
        raise ValueError(
            "The created DWI image has"
            f" {dwi_signal_filter.outputs[dwi_signal_filter.KEY_DWI].shape(3)} DWI"
            " acqusitions, but there are only"
            f" {len(input_parameters.b_values)} b_values"
        )

    logger.info(
        "The DWI signal filter generated an image with shape: %s, and datatype: %s",
        dwi_signal_filter.outputs[dwi_signal_filter.KEY_DWI].shape,
        dwi_signal_filter.outputs[dwi_signal_filter.KEY_DWI].image.dtype,
    )

    # TypeAlias
    dwi = dwi_signal_filter.outputs[dwi_signal_filter.KEY_DWI]
    dwi_3d_dict: DwiDict = {}
    for b_value_index, b_value in enumerate(
        dwi_signal_filter.outputs[dwi_signal_filter.KEY_ATTENUATION].metadata.b_values
    ):
        dwi_3d_dict[b_value] = NiftiImageContainer(  # the key is the b_value
            nib.Nifti1Image(
                dwi.image[:, :, :, b_value_index],
                affine=np.eye(4),
                dtype=dwi.image.dtype,
            )
        )
        assert dwi_3d_dict[b_value].shape == constant_shape

    # We should get an output that has the same size 4th dimension as the number
    # b_values/b_vectors
    if len(dwi_3d_dict) != len(input_parameters.b_values):
        raise ValueError(
            "The created DWI dictionary has"
            f" {len(dwi_3d_dict)} DWI"
            " acqusitions, but there are only"
            f" {len(input_parameters.b_values)} b_values"
        )

    # running the add complex noise filter
    final_outputs_dict: DwiDict = {}
    # for key, value in dwi_3d_dict_transformed.items():
    for key, value in dwi_3d_dict.items():
        add_complex_noise_filter = AddComplexNoiseFilter()
        add_complex_noise_filter.add_inputs(
            {
                AddComplexNoiseFilter.KEY_REF_IMAGE: s0,
                AddComplexNoiseFilter.KEY_IMAGE: value,
                AddComplexNoiseFilter.KEY_SNR: input_parameters.snr,
            }
        )

        add_complex_noise_filter.run()
        phase_magnitude_filter = PhaseMagnitudeFilter()
        phase_magnitude_filter.add_parent_filter(add_complex_noise_filter)
        phase_magnitude_filter.run()
        final_outputs_dict[key] = phase_magnitude_filter.outputs[
            PhaseMagnitudeFilter.KEY_MAGNITUDE
        ]
        assert final_outputs_dict[key].shape == constant_shape

    logger.info(
        (
            "The add complex noise filter + magnitude filter generated %d images with"
            " shape: %s, and datatype: %s"
        ),
        len(final_outputs_dict),
        final_outputs_dict[key].shape,
        final_outputs_dict[key].image.dtype,
    )

    # We should get an output that has the same size 4th dimension as the number
    # b_values/b_vectors
    if len(final_outputs_dict) != len(input_parameters.b_values):
        raise ValueError(
            "The created DWI+noise dictionary has"
            f" {len(dwi_3d_dict)} DWI"
            " acqusitions, but there are only"
            f" {len(input_parameters.b_values)} b_values"
        )

    if len(final_outputs_dict) >= 10000:
        raise ValueError("too  many images to combine")

    # running the combine time series filter
    for index, value in enumerate(final_outputs_dict.values()):
        combine_time_series_filter.add_input(f"image_0{index:04}", value)

    combine_time_series_filter.run()
    resulting_image: BaseImageContainer = combine_time_series_filter.outputs[
        CombineTimeSeriesFilter.KEY_IMAGE
    ]

    # We should get an output that has the same size 4th dimension as the number
    # b_values/b_vectors
    if resulting_image.shape[3] != len(input_parameters.b_values):
        raise ValueError(
            "The created final image has"
            f" {len(dwi_3d_dict)} DWI"
            " acqusitions, but there are only"
            f" {len(input_parameters.b_values)} b_values"
        )

    logger.info(
        (
            "The commbine time series filter generated an image with shape: %s, and"
            " datatype: %s"
        ),
        resulting_image.shape,
        resulting_image.image.dtype,
    )
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
