""" The main ASLDRO pipeline """
import json
import logging
import os
import pprint
import shutil
from collections.abc import Sequence
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Annotated, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import Field, validator
from typing_extensions import TypeAlias

from mrimagetools.containers.image import BaseImageContainer, NiftiImageContainer
from mrimagetools.filters.acquire_mri_image_filter import AcquireMriImageFilter
from mrimagetools.filters.append_metadata_filter import AppendMetadataFilter
from mrimagetools.filters.background_suppression_filter import (
    BackgroundSuppressionFilter,
)
from mrimagetools.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.filters.combine_time_series_filter import CombineTimeSeriesFilter
from mrimagetools.filters.gkm_filter import GkmFilter
from mrimagetools.filters.ground_truth_parser import (
    GroundTruthConfig,
    GroundTruthOutput,
    GroundTruthParser,
)
from mrimagetools.filters.invert_image_filter import InvertImageFilter
from mrimagetools.filters.json_loader import JsonLoaderFilter
from mrimagetools.filters.nifti_loader import NiftiLoaderFilter
from mrimagetools.filters.phase_magnitude_filter import PhaseMagnitudeFilter
from mrimagetools.filters.transform_resample_image_filter import (
    TransformResampleImageFilter,
)
from mrimagetools.models.file import GroundTruthFiles
from mrimagetools.pipelines.dwi_pipeline import (
    DwiInputParameters,
    dwi_pipeline_processing,
)
from mrimagetools.utils.general import map_dict, splitext
from mrimagetools.validators.parameter_model import ParameterModel
from mrimagetools.validators.parameters import reserved_string_list_validator
from mrimagetools.validators.user_parameter_input import (
    BS_INV_PULSE_TIMES,
    BS_NUM_INV_PULSES,
    BS_PULSE_EFFICIENCY,
    BS_SAT_PULSE_TIME,
    BS_T1_OPT,
    get_example_input_params,
    validate_input_params,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".zip", ".tar.gz"]
# Used in shutil.make_archive
EXTENSION_MAPPING = {".zip": "zip", ".tar.gz": "gztar"}


SnakeCaseStr: TypeAlias = Annotated[str, Field(regex="^[A-Za-z_][A-Za-z0-9_]*$")]


class ScaleOffset(ParameterModel):
    """Scale and offset"""

    scale: float
    offset: float


class Distribution(ParameterModel):
    """A model of a rotation distribution"""

    distribution: Literal["gaussian"]
    mean: float
    sd: float


class GeneralImageSeriesParameters(ParameterModel):
    """Image series parameters which are general to all types of image series"""

    random_seed: int = Field(0, ge=0)

    acq_matrix: list[Annotated[int, Field(gt=0)]] = Field(
        [64, 64, 40], min_items=3, max_items=3
    )
    """A matrix dimension size must have 3 element, each greater than 0"""

    output_image_type: Literal["complex", "magnitude"] = "magnitude"

    acq_contrast: Literal["ge", "se", "ir"] = "se"

    @validator("acq_contrast")
    def case_insensitve(cls, value: str) -> str:
        """Turn the acq_contrast to lowercase"""
        return value.lower()


class RigidRotationSeriesParameters(GeneralImageSeriesParameters):
    """Series parameters that apply to scans that can be (rigid) rotated"""

    rot_x: Union[Annotated[float, Field(ge=-180, le=180)], Distribution] = Distribution(
        distribution="gaussian", mean=0.0, sd=0.0
    )
    rot_y: Union[Annotated[float, Field(ge=-180, le=180)], Distribution] = Distribution(
        distribution="gaussian", mean=0.0, sd=0.0
    )
    rot_z: Union[Annotated[float, Field(ge=-180, le=180)], Distribution] = Distribution(
        distribution="gaussian", mean=0.0, sd=0.0
    )

    transl_x: Union[
        Annotated[float, Field(ge=-180, le=180)], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)
    transl_y: Union[
        Annotated[float, Field(ge=-180, le=180)], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)
    transl_z: Union[
        Annotated[float, Field(ge=-180, le=180)], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)


class StructuralSeriesParameters(RigidRotationSeriesParameters):
    """Series parameters specific to structural scans"""

    desired_snr: Annotated[float, Field(ge=0)] = 100.0
    echo_time: Annotated[float, Field(gt=0)] = 0.005
    repetition_time: Annotated[float, Field(gt=0)] = 0.3
    excitation_flip_angle: Annotated[float, Field(ge=-180, le=180)] = 90.0
    inversion_flip_angle: Annotated[float, Field(ge=-180, le=180)] = 180.0
    inversion_time: Annotated[float, Field(ge=0)] = 1.0
    interpolation: Literal["linear", "continuous", "nearest"] = "linear"
    modality: Literal[
        "T1w",
        "T2w",
        "FLAIR",
        "PDw",
        "T2starw",
        "inplaneT1",
        "inplaneT2",
        "PDT2",
        "UNIT1",
    ] = "T1w"


class GroundTruthSeriesParameters(RigidRotationSeriesParameters):
    """Series parameters specific to ground truth data"""

    interpolation: Sequence[Literal["linear", "continuous", "nearest"]] = [
        "linear",
        "continuous",
    ]


class BackgroundSuppressionParameters(ParameterModel):
    """Background suppression parameters"""

    sat_pulse_time: float
    sat_pulse_time_opt: float
    pulse_efficiency: Literal["ideal"]
    num_inv_pulses: int
    apply_to_asl_context: Sequence[Literal["label", "control"]]


class AslContextValues(ParameterModel):
    """ASL-specific echo times"""

    M0SCAN: float
    CONTROL: float
    LABEL: float


class AslSeriesParameters(GeneralImageSeriesParameters):
    """Series parameters specific to ASL"""

    rot_x: Union[
        Sequence[Annotated[float, Field(ge=-180, le=180)]], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)
    rot_y: Union[
        Sequence[Annotated[float, Field(ge=-180, le=180)]], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)
    rot_z: Union[
        Sequence[Annotated[float, Field(ge=-180, le=180)]], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)

    transl_x: Union[
        Sequence[Annotated[float, Field(ge=-180, le=180)]], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)
    transl_y: Union[
        Sequence[Annotated[float, Field(ge=-180, le=180)]], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)
    transl_z: Union[
        Sequence[Annotated[float, Field(ge=-180, le=180)]], Distribution
    ] = Distribution(distribution="gaussian", mean=0.0, sd=0.0)

    background_suppression: Union[bool, BackgroundSuppressionParameters] = False
    signal_time: Union[
        Annotated[float, Field(ge=0, le=100)],
        Sequence[Annotated[float, Field(ge=0, le=100)]],
    ]
    """Signal time. Either a float between 0 and 100, or a list of floats"""

    gkm_model: Literal["full", "whitepaper"] = "full"
    desired_snr: Annotated[float, Field(ge=0)] = 1000.0

    asl_context: str = "m0scan control label"
    """An ASL context. Must be a string, space delimited, using only
    `m0scan` `label` and `control`. e.g. `m0scan m0scan label control`"""

    echo_time: Union[
        Annotated[Sequence[float], Field(min_length=1)], AslContextValues
    ] = AslContextValues(M0SCAN=0.01, CONTROL=0.01, LABEL=0.01)
    repetition_time: Union[
        Annotated[Sequence[float], Field(min_length=1)], AslContextValues
    ] = AslContextValues(M0SCAN=10.0, CONTROL=5.0, LABEL=5.0)

    @validator("background_suppression")
    def set_defaults_if_true(
        cls, value: Union[bool, BackgroundSuppressionParameters]
    ) -> Union[bool, BackgroundSuppressionParameters]:
        """If the background suppression is True, set some sensible defaults"""
        if isinstance(value, bool) and value is True:
            return BackgroundSuppressionParameters(
                sat_pulse_time=4.0,
                sat_pulse_time_opt=3.98,
                pulse_efficiency="ideal",
                num_inv_pulses=4,
                apply_to_asl_context=["label", "control"],
            )
        return value

    @validator("asl_context")
    def check_valid_asl_context(cls, value: str) -> str:
        """Check the ASL context is valid"""
        string_validator = reserved_string_list_validator(
            strings=["m0scan", "control", "label"], case_insensitive=True
        )
        if not string_validator(value):
            raise ValueError(string_validator)
        return value

    interpolation: Literal["linear", "continuous", "nearest"] = "linear"
    """Image interpolation type"""
    excitation_flip_angle: Annotated[float, Field(ge=-180, le=180)] = 90.0
    """Excitation flip angle [-180, 180] degrees"""
    inversion_flip_angle: Annotated[float, Field(ge=-180, le=180)] = 180.0
    """Inversion flip angle [-180, 180] degrees"""
    inversion_time: Annotated[float, Field(ge=0)] = 1.0
    """Inversion time in seconds (>=0)"""
    label_duration: Annotated[float, Field(ge=0, le=100)] = 1.8
    """Label duration in seconds"""
    label_efficiency: Annotated[float, Field(ge=0, le=1)] = 0.85
    """Label efficiency (between 0 and 1)"""
    label_type: Literal["casl", "pcasl", "pasl"] = "pcasl"
    """ASL label type"""

    @validator("label_type")
    def case_insensitve(cls, value: str) -> str:
        """Turn the label_type to lowercase"""
        return value.lower()


class ImageSeries(ParameterModel):
    """An image series (equivalent of a DICOM image series)"""

    series_type: Literal["asl", "structural", "ground_truth", "dwi"]
    series_description: Optional[str] = None
    series_parameters: Union[
        AslSeriesParameters,
        StructuralSeriesParameters,
        GroundTruthSeriesParameters,
        DwiInputParameters,
    ]


class DwiImageSeries(ImageSeries):
    """A diffusion-weighted image series"""

    series_type: Literal["dwi"]
    series_parameters: DwiInputParameters


class AslImageSeries(ImageSeries):
    """An ASL image series"""

    series_type: Literal["asl"]
    series_parameters: AslSeriesParameters


class StructuralImageSeries(ImageSeries):
    """A structural image series"""

    series_type: Literal["structural"]
    series_parameters: StructuralSeriesParameters


class GroundTruthImageSeries(ImageSeries):
    """A ground truth image series"""

    series_type: Literal["ground_truth"]
    series_parameters: GroundTruthSeriesParameters


class GlobalConfiguration(ParameterModel):
    """Global configuration options"""

    ground_truth: GroundTruthFiles
    """Ground truth. Either a pair of nii/json files, or a root file name,
    with assumed .json or .nii extensions."""
    image_override: dict[SnakeCaseStr, float] = Field(default_factory=dict)
    """Image override options. Must be a dict of snakecase keys with the value
    to use to override *every* voxel value for the corresponding image"""
    parameter_override: dict[SnakeCaseStr, float] = Field(default_factory=dict)
    """Parameter override options. Must be a dict of snakecase keys with the value
    used to override a parameter with the same name."""
    ground_truth_modulate: dict[SnakeCaseStr, ScaleOffset] = Field(default_factory=dict)
    """Used to modulate (with a scale and offset) a ground truth image"""
    subject_label: Optional[Annotated[str, Field(regex="^[A-Za-z0-9\\-]+$")]] = None


class GenericImageSeries(ParameterModel):
    """A generic image series (helps to discriminate between the different types based
    on the `series_type` field)"""

    __root__: Union[
        AslImageSeries, StructuralImageSeries, GroundTruthImageSeries
    ] = Field(..., discriminator="series_type")


class InputParameters(ParameterModel):
    """DRO pipeline input parameters"""

    global_configuration: GlobalConfiguration
    image_series: Sequence[GenericImageSeries]


def load_ground_truth(input_params: InputParameters) -> GroundTruthParser:
    """Load and prepare and run the GroundTruthLoaderFilter from the given input parameters
    """
    json_filter = JsonLoaderFilter()
    json_filter.add_input(
        "filename", str(input_params.global_configuration.ground_truth.json_file)
    )
    json_filter.add_input("schema", GroundTruthConfig.schema())
    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input(
        "filename", str(input_params.global_configuration.ground_truth.nii_file)
    )

    json_filter.run()
    ground_truth_filter = GroundTruthParser()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_input("config", json_filter.outputs)
    ground_truth_filter.run()

    logger.info("JsonLoaderFilter outputs:\n%s", pprint.pformat(json_filter.outputs))
    logger.debug("NiftiLoaderFilter outputs:\n%s", pprint.pformat(nifti_filter.outputs))
    logger.debug(
        "GroundTruthLoaderFilter outputs:\n%s",
        pprint.pformat(ground_truth_filter.outputs),
    )
    return ground_truth_filter


def dwi_pipeline(
    image_series: DwiImageSeries,
    ground_truth_filter: GroundTruthParser,
    series_number: int,
) -> BaseImageContainer:
    """DWI pipeline
    :param image_series: The DwiImageSeries parameters
    :param ground_truth_filter: The GroundTruthLoaderParser with loaded ground truth
    :param series_number: The (virtual) series number"""
    dwi_image = dwi_pipeline_processing(
        ground_truth_parser=ground_truth_filter,
        input_parameters=image_series.series_parameters,
    )

    append_metadata_filter = AppendMetadataFilter()
    append_metadata_filter.add_input("image", dwi_image)
    append_metadata_filter.add_input(
        AppendMetadataFilter.KEY_METADATA,
        {
            "series_description": image_series.series_description,
            "series_type": image_series.series_type,
            "series_number": series_number,
        },
    )
    append_metadata_filter.run()
    return append_metadata_filter.outputs["image"]


def asl_pipeline(
    image_series: AslImageSeries,
    ground_truth_filter: GroundTruthParser,
    series_number: int,
) -> NiftiImageContainer:
    """ASL pipeline
    Comprises GKM, then MRI signal model, transform and resampling,
    and noise for each dynamic.
    After the 'acquisition loop' the dynamics are concatenated into a single 4D file
    :param image_series: The AslImageSeries parameters
    :param ground_truth_filter: The GroundTruthParser with loaded ground truth
    :param series_number: The (virtual) series number"""
    asl_params = image_series.series_parameters
    # initialise the random number generator for the image series
    np.random.seed(image_series.series_parameters.random_seed)
    logger.info(
        "Running DRO generation with the following parameters:\n%s",
        pprint.pformat(asl_params),
    )
    # Create one-time data that is required by the Acquisition Loop
    # 1. m0 resampled at the acquisition resolution
    m0_resample_filter = TransformResampleImageFilter()
    m0_resample_filter.add_parent_filter(
        ground_truth_filter,
        io_map={"m0": TransformResampleImageFilter.KEY_IMAGE},
    )
    m0_resample_filter.add_input(
        TransformResampleImageFilter.KEY_TARGET_SHAPE,
        tuple(asl_params.acq_matrix),
    )
    # 2. determine if background suppression is to be performed
    # if so then generate the suppressed static magnetisation
    if asl_params.background_suppression:
        if not isinstance(
            asl_params.background_suppression, BackgroundSuppressionParameters
        ):
            raise ValueError("Background suppression parameters have not been set")
        bs_params = map_dict(
            asl_params.background_suppression.dict(exclude_none=True),
            {
                BS_SAT_PULSE_TIME: BackgroundSuppressionFilter.KEY_SAT_PULSE_TIME,
                BS_INV_PULSE_TIMES: BackgroundSuppressionFilter.KEY_INV_PULSE_TIMES,
                BS_PULSE_EFFICIENCY: BackgroundSuppressionFilter.KEY_PULSE_EFFICIENCY,
                BS_NUM_INV_PULSES: BackgroundSuppressionFilter.KEY_NUM_INV_PULSES,
                BS_T1_OPT: BackgroundSuppressionFilter.KEY_T1_OPT,
            },
            io_map_optional=True,
        )
        # if "sat_pulse_time_opt" is provided then some values need switching
        # round
        if asl_params.background_suppression.sat_pulse_time_opt is not None:
            # "sat_pulse_time" goes to "mag_time", as this is the time we
            # want to generate mangetisation at.
            bs_params[BackgroundSuppressionFilter.KEY_MAG_TIME] = bs_params[
                BackgroundSuppressionFilter.KEY_SAT_PULSE_TIME
            ]
            # "sat_pulse_time_opt" goes to "sat_pulse_time", because this is what
            # we want to generate optimised times for
            bs_params[
                BackgroundSuppressionFilter.KEY_SAT_PULSE_TIME
            ] = asl_params.background_suppression.sat_pulse_time_opt

        bs_filter = BackgroundSuppressionFilter()
        bs_filter.add_inputs(bs_params)
        bs_filter.add_parent_filter(
            parent=ground_truth_filter,
            io_map={
                "m0": BackgroundSuppressionFilter.KEY_MAG_Z,
                "t1": BackgroundSuppressionFilter.KEY_T1,
            },
        )

    # initialise combine time series filter outside of both loops
    combine_time_series_filter = CombineTimeSeriesFilter()
    # if "signal_time" is a singleton, copy and place in a list

    signal_time_list: Sequence[float]
    if isinstance(asl_params.signal_time, (float, int)):
        signal_time_list = [deepcopy(asl_params.signal_time)]
    else:
        # otherwise it is a list so copy the entire list
        signal_time_list = deepcopy(asl_params.signal_time)

    vol_index = 0

    # Multiphase ASL Loop: loop over signal_time_list
    for multiphase_index, t in enumerate(signal_time_list):
        # place the loop value for signal_time into the asl_params dict
        asl_params.signal_time = t
        # Run the GkmFilter on the ground_truth data
        gkm_filter = GkmFilter()
        # Add ground truth parameters from the ground_truth_filter:
        # perfusion_rate, transit_time
        # m0,lambda_blood_brain, t1_arterial_blood all have the same keys; t1 maps
        # to t1_tissue
        gkm_filter.add_parent_filter(
            parent=ground_truth_filter,
            io_map={
                "perfusion_rate": GkmFilter.KEY_PERFUSION_RATE,
                "transit_time": GkmFilter.KEY_TRANSIT_TIME,
                "m0": GkmFilter.KEY_M0,
                "t1": GkmFilter.KEY_T1_TISSUE,
                "lambda_blood_brain": GkmFilter.KEY_LAMBDA_BLOOD_BRAIN,
                "t1_arterial_blood": GkmFilter.KEY_T1_ARTERIAL_BLOOD,
            },
        )
        # Add parameters from the input_params: label_type, signal_time, label_duration and
        # label_efficiency all have the same keys
        gkm_filter.add_inputs(asl_params.dict(exclude_none=True))
        gkm_filter.add_input(GkmFilter.KEY_MODEL, asl_params.gkm_model)
        # reverse the polarity of delta_m.image for encoding it into the label signal
        invert_delta_m_filter = InvertImageFilter()
        invert_delta_m_filter.add_parent_filter(
            parent=gkm_filter, io_map={GkmFilter.KEY_DELTA_M: "image"}
        )

        # ASL Context Loop: loop over ASL context, run the AcquireMriImageFilter and put the
        # output image into the CombineTimeSeriesFilter

        for asl_context_index, asl_context in enumerate(asl_params.asl_context.split()):
            acquire_mri_image_filter = AcquireMriImageFilter()
            # check that background suppression is enabled, and that it should be run
            # for the current ``asl_context```
            if asl_params.background_suppression and (
                asl_context in asl_params.background_suppression.apply_to_asl_context
            ):
                # map all inputs except for m0
                acquire_mri_image_filter.add_parent_filter(
                    parent=ground_truth_filter,
                    io_map={
                        key: key
                        for key in ground_truth_filter.outputs.keys()
                        if key != "m0"
                    },
                )
                # get m0 from bs
                acquire_mri_image_filter.add_parent_filter(
                    parent=bs_filter,
                    io_map={
                        BackgroundSuppressionFilter.KEY_MAG_Z: AcquireMriImageFilter.KEY_M0,
                    },
                )
            else:
                # map all inputs from the ground truth
                acquire_mri_image_filter.add_parent_filter(parent=ground_truth_filter)

            # map inputs from asl_params. acq_contrast, excitation_flip_angle, desired_snr,
            # inversion_time, inversion_flip_angle (last 2 are optional)
            acquire_mri_image_filter.add_inputs(
                asl_params.dict(exclude_none=True),
                io_map={
                    "acq_contrast": AcquireMriImageFilter.KEY_ACQ_CONTRAST,
                    "excitation_flip_angle": AcquireMriImageFilter.KEY_EXCITATION_FLIP_ANGLE,
                    "desired_snr": AcquireMriImageFilter.KEY_SNR,
                    "inversion_time": AcquireMriImageFilter.KEY_INVERSION_TIME,
                    "inversion_flip_angle": AcquireMriImageFilter.KEY_INVERSION_FLIP_ANGLE,
                },
                io_map_optional=True,
            )

            # if asl_context == "label" use the inverted delta_m as
            # the input MriSignalFilter.KEY_MAG_ENC
            if asl_context.lower() == "label":
                acquire_mri_image_filter.add_parent_filter(
                    parent=invert_delta_m_filter,
                    io_map={"image": AcquireMriImageFilter.KEY_MAG_ENC},
                )
            # set the image flavour to "PERFUSION"
            acquire_mri_image_filter.add_input(
                AcquireMriImageFilter.KEY_IMAGE_FLAVOR, "PERFUSION"
            )
            # Type guard some variables to ensure we have lists
            if not isinstance(asl_params.echo_time, Sequence):
                raise TypeError("Echo time is not a list")
            if not isinstance(asl_params.repetition_time, Sequence):
                raise TypeError("repetition_time is not a list")
            if not isinstance(asl_params.rot_x, Sequence):
                raise TypeError("asl_params.rot_x is not a list")
            if not isinstance(asl_params.rot_y, Sequence):
                raise TypeError("asl_params.rot_y is not a list")
            if not isinstance(asl_params.rot_z, Sequence):
                raise TypeError("asl_params.rot_z is not a list")
            if not isinstance(asl_params.transl_x, Sequence):
                raise TypeError("asl_params.transl_x is not a list")
            if not isinstance(asl_params.transl_y, Sequence):
                raise TypeError("asl_params.transl_y is not a list")
            if not isinstance(asl_params.transl_z, Sequence):
                raise TypeError("asl_params.transl_z is not a list")

            # build acquisition loop parameter dictionary
            # parameters that cannot be directly mapped
            acq_loop_params = {
                AcquireMriImageFilter.KEY_ECHO_TIME: asl_params.echo_time[
                    asl_context_index
                ],
                AcquireMriImageFilter.KEY_REPETITION_TIME: asl_params.repetition_time[
                    asl_context_index
                ],
                AcquireMriImageFilter.KEY_ROTATION: (
                    asl_params.rot_x[asl_context_index],
                    asl_params.rot_y[asl_context_index],
                    asl_params.rot_z[asl_context_index],
                ),
                AcquireMriImageFilter.KEY_TRANSLATION: (
                    asl_params.transl_x[asl_context_index],
                    asl_params.transl_y[asl_context_index],
                    asl_params.transl_z[asl_context_index],
                ),
                AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(asl_params.acq_matrix),
                AcquireMriImageFilter.KEY_INTERPOLATION: asl_params.interpolation,
            }
            # add these inputs to the filter
            acquire_mri_image_filter.add_inputs(acq_loop_params)
            # map the reference_image for the noise generation to the m0 ground truth.
            acquire_mri_image_filter.add_parent_filter(
                m0_resample_filter,
                io_map={
                    m0_resample_filter.KEY_IMAGE: AcquireMriImageFilter.KEY_REF_IMAGE
                },
            )
            append_metadata_filter = AppendMetadataFilter()

            if asl_params.output_image_type == "magnitude":
                # if 'output_image_type' is 'magnitude' use the phase_magnitude filter
                phase_magnitude_filter = PhaseMagnitudeFilter()
                phase_magnitude_filter.add_parent_filter(acquire_mri_image_filter)
                append_metadata_filter.add_parent_filter(
                    phase_magnitude_filter, io_map={"magnitude": "image"}
                )
            else:
                # otherwise just pass on the complex data
                append_metadata_filter.add_parent_filter(acquire_mri_image_filter)

            append_metadata_filter.add_input(
                AppendMetadataFilter.KEY_METADATA,
                {
                    "series_description": image_series.series_description,
                    "series_type": image_series.series_type,
                    "series_number": series_number,
                    "asl_context": asl_context,
                    "multiphase_index": multiphase_index,
                },
            )

            # Add the acqusition pipeline to the combine time series filter
            combine_time_series_filter.add_parent_filter(
                parent=append_metadata_filter,
                io_map={"image": f"image_{vol_index}"},
            )
            # increment the volume index
            vol_index += 1

    combine_time_series_filter.run()
    acquired_timeseries_nifti_container: NiftiImageContainer = (
        combine_time_series_filter.outputs["image"].as_nifti()
    )
    acquired_timeseries_nifti_container.header[
        "descrip"
    ] = image_series.series_description

    # logging
    logger.debug("GkmFilter outputs: \n %s", pprint.pformat(gkm_filter.outputs))
    logger.debug(
        "combine_time_series_filter outputs: \n %s",
        pprint.pformat(combine_time_series_filter.outputs),
    )

    return acquired_timeseries_nifti_container


def structural_pipeline(
    image_series: StructuralImageSeries,
    ground_truth_filter: GroundTruthParser,
    series_number: int,
) -> NiftiImageContainer:
    """Structural pipeline
    Comprises MRI signal,transform and resampling and noise models
    :param image_series: The StructuralImageSeries parameters
    :param ground_truth_filter: The GroundTruthParser with loaded ground truth
    :param series_number: The (virtual) series number"""
    struct_params: StructuralSeriesParameters = image_series.series_parameters
    # initialise the random number generator for the image series
    np.random.seed(struct_params.random_seed)

    logger.info(
        "Running DRO generation with the following parameters:\n%s",
        pprint.pformat(struct_params),
    )

    # Simulate acquisition
    acquire_mri_image_filter = AcquireMriImageFilter()
    # map inputs from the ground truth: t1, t2, t2_star, m0 all share the same name
    # so no explicit mapping is necessary.
    acquire_mri_image_filter.add_parent_filter(parent=ground_truth_filter)

    # append struct_params with additional parameters that need to be built/modified
    acquire_mri_image_filter_inputs = {
        **struct_params.dict(exclude_none=True),
        AcquireMriImageFilter.KEY_ROTATION: (
            struct_params.rot_x,
            struct_params.rot_y,
            struct_params.rot_z,
        ),
        AcquireMriImageFilter.KEY_TRANSLATION: (
            struct_params.transl_x,
            struct_params.transl_y,
            struct_params.transl_z,
        ),
        AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(struct_params.acq_matrix),
        AcquireMriImageFilter.KEY_SNR: struct_params.desired_snr,
        AcquireMriImageFilter.KEY_INTERPOLATION: struct_params.interpolation,
    }

    # map inputs from struct_params. acq_contrast, excitation_flip_angle, desired_snr,
    # inversion_time, inversion_flip_angle (last 2 are optional)
    acquire_mri_image_filter.add_inputs(
        acquire_mri_image_filter_inputs,
        io_map_optional=True,
    )

    append_metadata_filter = AppendMetadataFilter()
    append_metadata_filter.add_parent_filter(acquire_mri_image_filter)
    append_metadata_filter.add_input(
        AppendMetadataFilter.KEY_METADATA,
        {
            "series_description": image_series.series_description,
            "modality": struct_params.modality,
            "series_type": image_series.series_type,
            "series_number": series_number,
        },
    )

    struct_image_container: NiftiImageContainer
    if struct_params.output_image_type == "magnitude":
        phase_magnitude_filter = PhaseMagnitudeFilter()
        phase_magnitude_filter.add_parent_filter(append_metadata_filter)
        phase_magnitude_filter.run()
        struct_image_container = phase_magnitude_filter.outputs[
            PhaseMagnitudeFilter.KEY_MAGNITUDE
        ]
    else:
        append_metadata_filter.run()
        struct_image_container = append_metadata_filter.outputs[
            AcquireMriImageFilter.KEY_IMAGE
        ]

    struct_image_container.header["descrip"] = image_series.series_description
    return struct_image_container


def ground_truth_pipeline(
    image_series: GroundTruthImageSeries,
    ground_truth_filter: GroundTruthParser,
    series_number: int,
) -> list[NiftiImageContainer]:
    """Ground truth pipeline
    Comprises resampling all of the ground truth images with the specified resampling parameters
    :param image_series: The GroundTruthImageSeries parameters
    :param ground_truth_filter: The GroundTruthLoaderFilter with loaded ground truth
    :param series_number: The (virtual) series number"""
    ground_truth_params: GroundTruthSeriesParameters = image_series.series_parameters
    logger.info(
        "Running DRO generation with the following parameters:\n%s",
        pprint.pformat(ground_truth_params),
    )
    # Loop over all the ground truth images and resample as specified
    ground_truth_keys = ground_truth_filter.outputs.keys()
    ground_truth_image_keys = [
        key
        for key in ground_truth_keys
        if isinstance(ground_truth_filter.outputs[key], BaseImageContainer)
    ]
    return_image_list: list[NiftiImageContainer] = []
    for quantity in ground_truth_image_keys:
        resample_filter = TransformResampleImageFilter()
        # map the ground_truth_filter to the resample filter
        resample_filter.add_parent_filter(
            ground_truth_filter, io_map={quantity: "image"}
        )
        # there are two interpolation parameters in an array, the first is for all
        # quantities except for "seg_label", the second is for "seg_label". This is
        # because "seg_label" is a mask nearest neighbour interpolation is usually
        # more appropriate.
        interp_idx = 0
        if quantity == "seg_label":
            interp_idx = 1

        resample_filter_input_params = {
            **ground_truth_params.dict(exclude_none=True),
            TransformResampleImageFilter.KEY_ROTATION: (
                ground_truth_params.rot_x,
                ground_truth_params.rot_y,
                ground_truth_params.rot_z,
            ),
            TransformResampleImageFilter.KEY_TRANSLATION: (
                ground_truth_params.transl_x,
                ground_truth_params.transl_y,
                ground_truth_params.transl_z,
            ),
            AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(
                ground_truth_params.acq_matrix
            ),
            AcquireMriImageFilter.KEY_INTERPOLATION: ground_truth_params.interpolation[
                interp_idx
            ],
        }
        resample_filter.add_inputs(resample_filter_input_params)
        resample_filter.run()
        append_metadata_filter = AppendMetadataFilter()
        append_metadata_filter.add_parent_filter(resample_filter)
        append_metadata_filter.add_input(
            AppendMetadataFilter.KEY_METADATA,
            {
                "series_description": image_series.series_description,
                "series_type": image_series.series_type,
                "series_number": series_number,
            },
        )
        # Run the append_metadata_filter to generate an acquired volume
        append_metadata_filter.run()
        # append to output image list
        return_image_list.append(
            append_metadata_filter.outputs[AppendMetadataFilter.KEY_IMAGE]
        )
    return return_image_list


class PipelineReturnVariables(ParameterModel):
    """The parameters return by any of the DRO generation pipelines"""

    hrgt: GroundTruthOutput
    """The ground truth after modifications (outputs from the GroundTruthParser)"""
    dro_output: list[BaseImageContainer]
    """list of the image containers generated by the pipeline which would
    normally be saved in BIDS format"""
    params: InputParameters
    """The input parameters to the pipeline"""


def run_full_asl_dro_pipeline(
    input_params: Optional[Union[dict, InputParameters]] = None,
    output_filename: Optional[str] = None,
) -> PipelineReturnVariables:
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """A function that runs the entire DRO pipeline. This
    can be extended as more functionality is included.
    This function is deliberately verbose to explain the
    operation, inputs and outputs of individual filters.

    :param input_params: The input parameter dictionary. If None, the defaults will be
      used.
    :param output_filename: The output filename. Must be an zip/tar.gz archive. If None,
      no files will be generated.

    :returns: PipelineReturnVariables
    """

    if input_params is None:
        input_params = get_example_input_params()

    if isinstance(input_params, dict):
        # Validate parameter and update defaults
        input_params = validate_input_params(input_params)

    input_parameters = (
        InputParameters(**input_params)
        if isinstance(input_params, dict)
        else input_params
    )

    if output_filename is not None:
        _, output_filename_extension = splitext(output_filename)
        if output_filename_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"File output type {output_filename_extension} not supported"
            )

    subject_label = input_parameters.global_configuration.subject_label

    ground_truth_filter = load_ground_truth(input_parameters)

    # create output lists to be populated in the "image_series" loop
    output_image_list: list[BaseImageContainer] = []

    # Loop over "image_series" in input_params
    # Take the asl image series and pass it to the remainder of the pipeline
    # update the input_params variable so it contains the asl series parameters
    for series_index, image_series in enumerate(
        [i.__root__ for i in input_parameters.image_series]
    ):
        if not isinstance(
            image_series,
            (AslImageSeries, GroundTruthImageSeries, StructuralImageSeries),
        ):
            raise TypeError(
                f"{image_series} is not a known image series type"
                f" (type:{type(image_series)})"
            )
        series_number = series_index + 1

        if isinstance(image_series, AslImageSeries):
            output_image_list.append(
                asl_pipeline(
                    image_series=image_series,
                    ground_truth_filter=ground_truth_filter,
                    series_number=series_number,
                )
            )

        if isinstance(image_series, StructuralImageSeries):
            output_image_list.append(
                structural_pipeline(
                    image_series=image_series,
                    ground_truth_filter=ground_truth_filter,
                    series_number=series_number,
                )
            )

        ############################################################################################
        if isinstance(image_series, GroundTruthImageSeries):
            output_image_list.extend(
                ground_truth_pipeline(
                    image_series=image_series,
                    ground_truth_filter=ground_truth_filter,
                    series_number=series_number,
                )
            )

    # Output everything to a temporary directory
    with TemporaryDirectory() as temp_dir:
        for image_to_output in output_image_list:
            bids_output_filter = BidsOutputFilter()
            bids_output_filter.add_input(
                BidsOutputFilter.KEY_OUTPUT_DIRECTORY, temp_dir
            )
            bids_output_filter.add_input(BidsOutputFilter.KEY_IMAGE, image_to_output)
            bids_output_filter.add_input(
                BidsOutputFilter.KEY_SUBJECT_LABEL, subject_label
            )
            # run the filter to write the BIDS files to disk
            bids_output_filter.run()
            # save the DRO parameters at the root of the directory
            bids_output_filter.save_json(
                json.loads(input_parameters.json(exclude_none=True)),
                os.path.join(temp_dir, "asldro_parameters.json"),
            )

            if output_filename is not None:
                filename, file_extension = splitext(output_filename)
                # output the file archive
                logger.info("Creating output archive: %s", output_filename)
                shutil.make_archive(
                    filename, EXTENSION_MAPPING[file_extension], root_dir=temp_dir
                )

    return PipelineReturnVariables(
        hrgt=ground_truth_filter.parsed_outputs,
        dro_output=output_image_list,
        params=input_parameters,
    )


if __name__ == "__main__":
    run_full_asl_dro_pipeline()
