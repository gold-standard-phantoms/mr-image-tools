"""ASL quantification filter

    A filter that calculates the perfusion rate for arterial spin labelling data.

"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from mrimagetools.filters.asl_quant_functions import (
    asl_quant_lsq_gkm,
    asl_quant_wp_casl,
    asl_quant_wp_pasl,
)
from mrimagetools.filters.gkm_filter import check_and_make_image_from_value
from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.gkm_filter import GkmFilter
from mrimagetools.v2.validators.fields import UnitField

KEY_CONTROL = "control"
KEY_LABEL = "label"
KEY_MODEL = "gkm_model"
KEY_M0 = GkmFilter.KEY_M0
KEY_PERFUSION_RATE = GkmFilter.KEY_PERFUSION_RATE
KEY_LABEL_TYPE = GkmFilter.KEY_LABEL_TYPE
KEY_LABEL_DURATION = GkmFilter.KEY_LABEL_DURATION
KEY_LABEL_EFFICIENCY = GkmFilter.KEY_LABEL_EFFICIENCY
KEY_LAMBDA_BLOOD_BRAIN = GkmFilter.KEY_LAMBDA_BLOOD_BRAIN
KEY_T1_ARTERIAL_BLOOD = GkmFilter.KEY_T1_ARTERIAL_BLOOD
KEY_POST_LABEL_DELAY = GkmFilter.M_POST_LABEL_DELAY
KEY_MULTIPHASE_INDEX = "multiphase_index"
KEY_T1_TISSUE = GkmFilter.KEY_T1_TISSUE
KEY_TRANSIT_TIME = GkmFilter.KEY_TRANSIT_TIME
KEY_PERFUSION_RATE_ERR = "perfusion_rate_err"
KEY_TRANSIT_TIME_ERR = "transit_time_err"
KEY_STD_ERROR = "std_error"

WHITEPAPER = GkmFilter.MODEL_WP
FULL = GkmFilter.MODEL_FULL
M0_TOL = 1e-6

FIT_IMAGE_NAME = {
    KEY_STD_ERROR: "FITErr",
    KEY_PERFUSION_RATE_ERR: "RCBFErr",
    KEY_TRANSIT_TIME: "ATT",
    KEY_TRANSIT_TIME_ERR: "ATTErr",
}
FIT_IMAGE_UNITS = {
    KEY_STD_ERROR: "a.u.",
    KEY_PERFUSION_RATE_ERR: "ml/100g/min",
    KEY_TRANSIT_TIME: "s",
    KEY_TRANSIT_TIME_ERR: "s",
}
ESTIMATION_ALGORITHM = {
    WHITEPAPER: """Calculated using the single subtraction simplified model for
CBF quantification from the ASL White Paper:

Alsop et. al., Recommended implementation of arterial
spin-labeled perfusion MRI for clinical applications:
a consensus of the ISMRM perfusion study group and the
european consortium for ASL in dementia. Magnetic Resonance
in Medicine, 73(1):102–116, apr 2014. doi:10.1002/mrm.25197
""",
    FULL: """Least Squares fit to the General Kinetic Model for
Arterial Spin Labelling:

Buxton et. al., A general
kinetic model for quantitative perfusion imaging with arterial
spin labeling. Magnetic Resonance in Medicine, 40(3):383–396,
sep 1998. doi:10.1002/mrm.1910400308.""",
}


@dataclass
class AslQuantificationFilterImages:
    """ASL Quantification Filter Images"""

    control: BaseImageContainer
    """ The control image (3D or 4D timeseries)"""

    label: BaseImageContainer
    """ The label image (3D or 4D timeseries)"""

    m0: BaseImageContainer
    """ Equilibrium magnetisation image"""

    t1_tissue: Optional[Union[float, BaseImageContainer]]  # = Field(..., ge=0)
    """Longitudinal relaxation time of the tissue, seconds
        (greater than 0). Required if ``'model'=='full'``"""


class AslQuantificationFilterParameters(BaseModel):
    """ASL Quantification Filter Parameters"""

    label_type: Literal["pasl", "pcasl", "casl"]
    """the type of labelling used: "pasl" for pulsed ASL "pcasl" or "casl" for for
    continuous ASL."""

    lambda_blood_brain: float = Field(..., ge=0, le=1)
    """The blood-brain-partition-coefficient (0 to 1 inclusive)"""

    label_duration: float = Field(..., ge=0)
    """The temporal duration of the labelled bolus, seconds
      (0 or greater). For PASL this is equivalent to :math:`\text{TI}_1`"""

    post_label_delay: Union[float, list[float]]
    """The duration between the end of the labelling
        pulse and the imaging excitation pulse, seconds (0 or greater).
        For PASL this is equivalent to :math:`\text{TI}`.
        If ``'model'=='full'`` then this must be a list and the length of this
        must match the number of unique entries in ``'multiphase_index'``."""

    label_efficiency: float = Field(..., ge=0, le=1)
    """The degree of inversion of the labelling (0 to 1 inclusive)"""

    t1_arterial_blood: float = Field(..., ge=0)
    """Longitudinal relaxation time of arterial blood,
        seconds (greater than 0)"""

    key_model: Literal["whitepaper", "full"]
    """defines which model to use

        * 'whitepaper' uses the single-subtraction white paper equation
        * 'full' least square fitting to the full GKM."""

    multiphase_index: Optional[list[int]]
    """A list the same length as the fourth dimension
        of the label image that defines which phase each image belongs to,
        and is also the corresponding index in the ``'post_label_delay'`` list.
        Required if ``'gkm_model'=='full'``"""


def white_paper(
    input_images: AslQuantificationFilterImages,
    parameters: AslQuantificationFilterParameters,
) -> NDArray[np.floating]:
    """White Paper"""
    images: dict[str, NDArray] = {}
    for key in [KEY_M0, KEY_CONTROL, KEY_LABEL]:
        current_image = getattr(input_images, key)
        if len(current_image.shape) == 4:
            # take the average along the 4th (time) dimension
            images[key] = np.average(current_image.image, axis=3)
        else:
            images[key] = current_image.image

    post_label_delay_float = cast(
        float,
        parameters.post_label_delay,
    )

    if parameters.label_type.lower() in [
        GkmFilter.CASL,
        GkmFilter.PCASL,
    ]:
        perfusion_rate = asl_quant_wp_casl(
            control=images[KEY_CONTROL],
            label=images[KEY_LABEL],
            m0=images[KEY_M0],
            lambda_blood_brain=parameters.lambda_blood_brain,
            label_duration=parameters.label_duration,
            post_label_delay=post_label_delay_float,
            label_efficiency=parameters.label_efficiency,
            t1_arterial_blood=parameters.t1_arterial_blood,
        )

    elif parameters.label_type.lower() in [
        GkmFilter.PASL,
    ]:
        perfusion_rate = asl_quant_wp_pasl(
            control=input_images.control.image,
            label=input_images.label.image,
            m0=input_images.m0.image,
            lambda_blood_brain=parameters.lambda_blood_brain,
            bolus_duration=parameters.label_duration,
            inversion_time=post_label_delay_float,
            label_efficiency=parameters.label_efficiency,
            t1_arterial_blood=parameters.t1_arterial_blood,
        )
    # Should never reach this block. It's purpose is to guard against an undefined
    # perfusion_rate being returned

    else:
        raise ValueError("Cannot return an undefined perfusion rate")

    return perfusion_rate


def full_model(
    input_images: AslQuantificationFilterImages,
    parameters: AslQuantificationFilterParameters,
) -> dict[str, NDArray]:
    """Full Model"""
    # fit multi PLD data to the General Kinetic Model
    # AslQuantificationFilter.asl_quant_lsq_gkm requires `t1_tissue` and
    # `lambda_blood_brain` to be np.ndarrays (same dimensions as m0), so
    # first create arrays of these if they are not
    shape = input_images.m0.shape

    if input_images.t1_tissue is None:
        #     # if not isinstance(input_images.t1_tissue, BaseImageContainer):
        raise ValueError("for Full model t1_tissue is required")

    t1_tissue = check_and_make_image_from_value(
        input_images.t1_tissue, shape, ImageMetadata(), None
    )
    lambda_blood_brain = check_and_make_image_from_value(
        parameters.lambda_blood_brain, shape, ImageMetadata(), None
    )
    # The input `post_label_delay` is values of PLD's corresponding to
    # each multiphase index. The actual PLD array needs to be built
    # using this information.
    if not isinstance(parameters.multiphase_index, list):
        raise ValueError("for Full model a multiphase index is required")
    if not isinstance(parameters.post_label_delay, list):
        raise ValueError("for Full model a post label delay is required as a list")
    if not len(parameters.multiphase_index) == len(parameters.post_label_delay):
        raise ValueError(
            "for Full model lists of the same length are required for multiphase index"
            " and post label delay"
        )

    post_label_delays = [
        parameters.post_label_delay[i] for i in parameters.multiphase_index
    ]

    # compute `perfusion_rate` and `transit_time`
    results = asl_quant_lsq_gkm(
        control=input_images.control.image,
        label=input_images.label.image,
        m0_tissue=input_images.m0.image,
        lambda_blood_brain=lambda_blood_brain,
        label_duration=parameters.label_duration,
        post_label_delay=post_label_delays,
        label_efficiency=parameters.label_efficiency,
        t1_arterial_blood=parameters.t1_arterial_blood,
        t1_tissue=t1_tissue,
        label_type=parameters.label_type.lower(),
    )

    return results


def asl_quantification_filter(
    input_images: AslQuantificationFilterImages,
) -> BaseImageContainer:
    """ASL Quantification Filter"""

    output_image: BaseImageContainer = input_images.control.clone()

    # amend the metadata
    output_image.metadata.repetition_time = None
    output_image.metadata.repetition_time_preparation = None
    output_image.metadata.echo_time = None
    output_image.metadata.m0_type = None
    output_image.metadata.excitation_flip_angle = None

    output_image.metadata.asl_context = "cbf"
    output_image.metadata.units = UnitField(root="ml/100g/min")
    output_image.metadata.image_type = (
        "DERIVED",
        "PRIMARY",
        "PERFUSION",
        "RCBF",
    )

    return output_image
