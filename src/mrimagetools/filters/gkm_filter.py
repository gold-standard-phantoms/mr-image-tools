""" General Kinetic Model Filter """

import logging
from typing import Annotated, Final, Literal, Optional, TypedDict, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, validate_call

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata

logger = logging.getLogger(__name__)

# Key constants
KEY_M0 = "m0"
KEY_LAMBDA_BLOOD_BRAIN = "lambda_blood_brain"

# Value constants
CASL = "casl"
PCASL = "pcasl"
PASL = "pasl"

MODEL_FULL: Final[str] = "full"
MODEL_WP: Final[str] = "whitepaper"


class GkmParameters(BaseModel):
    """Parameter for the filter that generates the ASL signal using the General Kinetic
    Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    perfusion_rate: BaseImageContainer
    """The perfusion rate, CBF, in ml/100g/min. All values must be positive"""

    transit_time: BaseImageContainer
    """The transit time, in seconds. All values must be positive"""

    m0: Union[Annotated[float, Field(ge=0)], BaseImageContainer]
    """The equilibrium magnetisation of the tissue, in arbitrary units. All values
    must be positive"""

    label_type: Literal["casl", "pcasl", "pasl"]
    """Determines which GKM equations to use:
    * "casl" OR "pcasl" (case insensitive) for the continuous model
    * "pasl" (case insensitive) for the pulsed model
    """
    label_duration: float = Field(..., ge=0, le=100)
    """The length of the labelling pulse, seconds (0 to 100 inclusive)"""

    signal_time: float = Field(..., ge=0, le=100)
    """The time after labelling commences to generate signal, seconds (0 to 100
    inclusive)"""

    label_efficiency: float = Field(..., ge=0, le=1)
    """The degree of inversion of the labelling (0 to 1 inclusive)"""

    lambda_blood_brain: Union[Annotated[float, Field(ge=0, le=1)], BaseImageContainer]
    """The blood-brain-partition-coefficient (0 to 1 inclusive)"""

    t1_arterial_blood: float = Field(..., ge=0, le=100)
    """Longitudinal relaxation time of arterial blood,
    seconds (0 exclusive to 100 inclusive)"""

    t1_tissue: BaseImageContainer
    """Longitudinal relaxation time of the tissue,
    seconds (0 to 100 inclusive, however voxels with ``t1 = 0`` will have
    ``delta_m = 0``)"""

    @field_validator("t1_tissue", "perfusion_rate", "transit_time")
    def check_non_negative(cls, values: BaseImageContainer) -> BaseImageContainer:
        """Check that we don't have any negative values in our image"""
        if np.any(values.image < 0):
            raise ValueError("image must not contain negative values")
        return values

    @field_validator("m0", "lambda_blood_brain")
    def check_float_or_image_non_negative(
        cls, values: Union[float, BaseImageContainer]
    ) -> Union[float, BaseImageContainer]:
        """Check that we don't have any negative values in our image or value,
        depending on the input type"""
        if isinstance(values, BaseImageContainer):
            if np.any(values.image < 0):
                raise ValueError("image must not contain negative values")
        else:
            if values < 0:
                raise ValueError("value must not be negative")
        return values

    @field_validator("lambda_blood_brain")
    def check_lambda_blood_brain(
        cls, values: Union[float, BaseImageContainer]
    ) -> Union[float, BaseImageContainer]:
        """Check that we don't have any values greater than 1 in our image or value,
        depending on the input type"""
        if isinstance(values, BaseImageContainer):
            if np.any(values.image > 1):
                raise ValueError("image must not contain values greater than 1")
        else:
            if values > 1:
                raise ValueError("value must not be greater than 1")
        return values

    model: Literal["full", "whitepaper"] = "full"
    """The model to use to generate the perfusion signal:
    * 'full' for the full 'Buxton' General Kinetic Model :cite:p:`Buxton1998`
    * 'whitepaper' for the simplified model, derived from the quantification
    equations the ASL Whitepaper consensus paper :cite:p:`Alsop2014`.
    Defaults to 'full'.
    """


def gkm_filter(gkm_parameters: GkmParameters) -> BaseImageContainer:
    r"""
    A filter that generates the ASL signal using the General Kinetic Model.

    :param gkm_parameters: The parameters for the filter. See
        :class:`GkmParameters` for more details.

    :return: An image with synthetic ASL perfusion contrast. This will
      be the same class as the input perfusion_rate

    **Metadata**

    The following parameters are added to the output image's metadata:

    * ``label_type``
    * ``label_duration`` (pcasl/casl only)
    * ``post_label_delay``
    * ``bolus_cut_off_flag`` (pasl only)
    * ``bolus_cut_off_delay_time`` (pasl only)
    * ``label_efficiency``
    * ``lambda_blood_brain`` (only if a single value is supplied)
    * ``t1_arterial_blood``
    * ``m0`` (only if a single value is supplied)
    * ``gkm_model`` = ``model``

    ``post_label_delay`` is calculated as ``signal_time - label_duration``

    ``bolus_cut_off_delay_time`` takes the value of the input
    ``label_duration``, this field is used for pasl in line with
    the BIDS specification.


    **Equations**

    The general kinetic model :cite:p:`Buxton1998` is the standard signal model
    for ASL perfusion measurements. It considers the difference between the
    control and label conditions to be a deliverable tracer, referred to
    as :math:`\Delta M(t)`.

    The amount of :math:`\Delta M(t)` within a voxel at time :math:`t`
    depends on the history of:

    * delivery of magnetisation by arterial flow
    * clearance by venous flow
    * longitudinal relaxation

    These processes can be described by defining three functions of time:

    1. The delivery function :math:`c(t)` - the normalised arterial
       concentration of magnetisation arriving at the voxel
       at time :math:`t`.
    2. The residue function :math:`r(t,t')` - the fraction of tagged water
       molecules that arrive at time :math:`t'` and
       are still in the voxel at time :math:`t`.
    3. The magnetisation relaxation function :math:`m(t,t')` is the fraction
       of the original longitudinal magnetisation tag carried by the water
       molecules that arrived at time :math:`t'` that remains at time :math:`t`.

    Using these definitions :math:`\Delta M(t)` can be constructed as the sum
    over history of delivery of magnetisation to the tissue weighted with the
    fraction of that magnetisation that remains in the voxel:

    .. math::

        &\Delta M(t)=2\cdot M_{0,b}\cdot f\cdot\left\{ c(t)\ast\left[r(t)\cdot m(t)\right]\right\}\\
        &\text{where}\\
        &\ast=\text{convolution operator} \\
        &r(t)=\text{residue function}=e^{-\frac{ft}{\lambda}}\\
        &m(t)=e^{-\frac{t}{T_{1}}}\\
        &c(t)=\text{delivery function, defined as plug flow} = \begin{cases}
        0  &  0<t<\Delta t\\
        \alpha e^{-\frac{t}{T_{1,b}}}\,\text{(PASL)}  &  \Delta t<t<\Delta t+\tau\\
        \alpha e^{-\frac{\Delta t}{T_{1,b}}}\,\text{(CASL/pCASL)}\\
        0  &  t>\Delta t+\tau
        \end{cases}\\
        &\alpha=\text{labelling efficiency} \\
        &\tau=\text{label duration} \\
        &\Delta t=\text{initial transit delay, ATT} \\
        &M_{0,b} = \text{equilibrium magnetisation of arterial blood} =
        \frac{M_{0,\text{tissue}}}{\lambda} \\
        & f = \text{the perfusion rate, CBF}\\
        &\lambda = \text{blood brain partition coefficient}\\
        &T_{1,b} = \text{longitudinal relaxation time of arterial blood}\\
        &T_{1} = \text{longitudinal relaxation time of tissue}\\

    Note that all units are in SI, with :math:`f` having units :math:`s^{-1}`.
    Multiplying by 6000 gives units of :math:`ml/100g/min`.

    *Full Model*

    The full solutions to the GKM :cite:p:`Buxton1998` are used to calculate
    :math:`\Delta M(t)` when ``model=="full"``:

    *   (p)CASL:

        .. math::

            &\Delta M(t)=\begin{cases}
            0 & 0<t\leq\Delta t\\
            2M_{0,b}fT'_{1}\alpha e^{-\frac{\Delta t}{T_{1,b}}}q_{ss}(t) &
            \Delta t<t<\Delta t+\tau\\
            2M_{0,b}fT'_{1}\alpha e^{-\frac{\Delta t}{T_{1,b}}}
            e^{-\frac{t-\tau-\Delta t}{T'_{1}}}q_{ss}(t) & t\geq\Delta t+\tau
            \end{cases}\\
            &\text{where}\\
            &q_{ss}(t)=\begin{cases}
            1-e^{-\frac{t-\Delta t}{T'_{1}}} & \Delta t<t <\Delta t+\tau\\
            1-e^{-\frac{\tau}{T'_{1}}} & t\geq\Delta t+\tau
            \end{cases}\\
            &\frac{1}{T'_{1}}=\frac{1}{T_1} + \frac{f}{\lambda}\\

    *   PASL:

        .. math::

            &\Delta M(t)=\begin{cases}
            0 & 0<t\leq\Delta t\\
            2M_{0,b}f(t-\Delta t) \alpha e^{-\frac{t}{T_{1,b}}}q_{p}(t)
            & \Delta t < t < t\Delta t+\tau\\
            2M_{0,b}f\alpha \tau e^{-\frac{t}{T_{1,b}}}q_{p}(t)
            & t\geq\Delta t+\tau
            \end{cases}\\
            &\text{where}\\
            &q_{p}(t)=\begin{cases}
            \frac{e^{kt}(e^{-k \Delta t}-e^{-kt})}{k(t-\Delta t)}
            & \Delta t<t<\Delta t+\tau\\
            \frac{e^{kt}(e^{-k\Delta t}-e^{k(\tau + \Delta t)}}{k\tau}
            & t\geq\Delta t+\tau
            \end{cases}\\
            &\frac{1}{T'_{1}}=\frac{1}{T_1} + \frac{f}{\lambda}\\
            &k=\frac{1}{T_{1,b}}-\frac{1}{T'_1}

    *Simplified Model"

    The simplified model, derived from the single subtraction quantification
    equations (see :class:`.AslQuantificationFilter`) are used when
    ``model=="whitepaper"``:

    *   (p)CASL:

        .. math::

            &\Delta M(t) = \begin{cases}
            0 & 0<t\leq\Delta t + \tau\\
            {2  M_{0,b}  f  T_{1,b} \alpha
            (1-e^{-\frac{\tau}{T_{1,b}}}) e^{-\frac{t-\tau}{T_{1,b}}}}
            & t > \Delta t + \tau\\
            \end{cases}\\

    *   PASL

        .. math::

            &\Delta M(t) = \begin{cases}
            0 & 0<t\leq\Delta t + \tau\\
            {2  M_{0,b}  f  \tau  \alpha
            e^{-\frac{t}{T_{1,b}}}} & t > \Delta t + \tau\\
            \end{cases}

    """

    perfusion_rate = gkm_parameters.perfusion_rate.image
    transit_time = gkm_parameters.transit_time.image
    t1_tissue = gkm_parameters.t1_tissue.image
    label_duration = gkm_parameters.label_duration

    signal_time = gkm_parameters.signal_time
    label_efficiency = gkm_parameters.label_efficiency
    t1_arterial_blood = gkm_parameters.t1_arterial_blood
    model = gkm_parameters.model
    label_type = gkm_parameters.label_type

    # blank dictionary for metadata to add
    metadata: ImageMetadata = ImageMetadata()
    m0_tissue = check_and_make_image_from_value(
        gkm_parameters.m0, perfusion_rate.shape, metadata, KEY_M0
    )
    lambda_blood_brain = check_and_make_image_from_value(
        gkm_parameters.lambda_blood_brain,
        perfusion_rate.shape,
        metadata,
        KEY_LAMBDA_BLOOD_BRAIN,
    )

    if model == MODEL_FULL:
        # gkm function
        logger.info(
            "Full General Kinetic Model for Continuous/pseudo-Continuous ASL"
            if label_type in [PCASL, CASL]
            else "Full General Kinetic Model for Pulsed ASL"
        )
        delta_m = calculate_delta_m_gkm(
            perfusion_rate,
            transit_time,
            m0_tissue,
            label_duration,
            signal_time,
            label_efficiency,
            lambda_blood_brain,
            t1_arterial_blood,
            t1_tissue,
            label_type,
        )

    elif model == MODEL_WP:
        # whitepaper function
        logger.info(
            "Simplified Kinetic Model for Continuous/pseudo-Continuous ASL"
            if label_type in [PCASL, CASL]
            else "Simplified Kinetic Model for Pulsed ASL"
        )
        delta_m = calculate_delta_m_whitepaper(
            perfusion_rate,
            transit_time,
            m0_tissue,
            label_duration,
            signal_time,
            label_efficiency,
            lambda_blood_brain,
            t1_arterial_blood,
            label_type,
        )
    # add metadata depending on the label type
    if label_type == PASL:
        metadata.bolus_cut_off_flag = True
        metadata.bolus_cut_off_delay_time = label_duration
    elif label_type in [CASL, PCASL]:
        metadata.label_duration = label_duration

    delta_m_container: BaseImageContainer = gkm_parameters.perfusion_rate.clone()
    # copy 'perfusion_rate' image container and set the image to delta_m
    # remove some metadata fields
    delta_m_container.metadata.units = None
    delta_m_container.metadata.quantity = None
    delta_m_container.image = delta_m

    # add common fields to metadata

    # The below is not a correct BIDS field, but is used for convenience
    metadata.label_type = gkm_parameters.label_type.lower()  # type: ignore
    metadata.post_label_delay = signal_time - label_duration
    metadata.label_efficiency = label_efficiency
    metadata.t1_arterial_blood = t1_arterial_blood
    metadata.image_flavor = "PERFUSION"
    metadata.gkm_model = model
    # merge this with the output image's metadata
    delta_m_container.metadata = ImageMetadata(
        **{
            **delta_m_container.metadata.model_dump(exclude_none=True),
            **metadata.model_dump(exclude_none=True),
        }
    )
    return delta_m_container


def check_and_make_image_from_value(
    arg: Union[float, BaseImageContainer],
    shape: tuple,
    metadata: ImageMetadata,
    metadata_key: Optional[str],
) -> NDArray[np.floating]:
    """Checks the type of the input parameter to see if it is a float or a
    BaseImageContainer. If it is an image:

    * return the image ndarray
    * check if it has the same value everywhere (i.e. an image override), if it does
        then place the value into the `metadata` dict under the `metadata_key`

    If it is a float:
    * make a ndarray with the same value
    * place the value into the `metadata` dict under the `metadata_key`

    This makes calculations more straightforward as a ndarray can always be expected.

    **Arguments**

    :param arg: The input parameter to check
    :param shape: The shape of the image to create
    :param metadata: metadata dict, which is updated by this function
    :param metadata_key: key to assign the value of arg (if a float or single value
        image) to if None - do not assign anything

    :return: image of the parameter
    """

    out_array: NDArray[np.floating]
    if isinstance(arg, BaseImageContainer):
        out_array = arg.image
        # Get a flattened view of nD numpy array
        flatten_arr = np.ravel(out_array)
        # Check if all value in array are equal and update metadata if so
        if np.all(out_array == flatten_arr[0]):
            if metadata_key is not None:
                setattr(metadata, metadata_key, flatten_arr[0])

    else:
        out_array = arg * np.ones(shape)
        if metadata_key is not None:
            setattr(metadata, metadata_key, arg)

    return out_array


class ArrivalState(TypedDict):
    not_arrived: Union[bool, NDArray[np.bool_]]
    arriving: Union[bool, NDArray[np.bool_]]
    arrived: Union[bool, NDArray[np.bool_]]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def compute_arrival_state_masks(
    transit_time: Union[NDArray[np.floating], float],
    signal_time: Annotated[float, Field(ge=0)],
    label_duration: Annotated[float, Field(ge=0)],
) -> ArrivalState:
    """Creates boolean masks for each of the states of the delivery curve

    :param transit_time: map of the transit time
    :param signal_time: the time to generate signal at
    :param label_duration: The duration of the labelling pulse
    :return: a dictionary with three entries, each a ndarray with shape
        the same as `transit_time`:

        :"not_arrived": voxels where the bolus has not reached yet
        :"arriving": voxels where the bolus has reached but not been completely
        delivered.
        :"arrived": voxels where the bolus has been completely delivered

    """
    return {
        "not_arrived": signal_time <= transit_time,
        "arriving": (transit_time < signal_time) & (
            signal_time < transit_time + label_duration
        ),
        "arrived": signal_time >= transit_time + label_duration,
    }


ArrayOrFloatT = TypeVar("ArrayOrFloatT", NDArray[np.floating], float)


def calculate_delta_m_gkm(
    perfusion_rate: ArrayOrFloatT,
    transit_time: ArrayOrFloatT,
    m0_tissue: ArrayOrFloatT,
    label_duration: float,
    signal_time: float,
    label_efficiency: float,
    partition_coefficient: ArrayOrFloatT,
    t1_arterial_blood: float,
    t1_tissue: ArrayOrFloatT,
    label_type: str,
) -> ArrayOrFloatT:
    r"""Calculates the difference in magnetisation between the control
    and label condition (:math:`\Delta M`) using the full solutions to the
    General Kinetic Model :cite:p:`Buxton1998`.

    :param perfusion_rate: Map of perfusion rate
    :param transit_time: Map of transit time
    :param m0_tissue: The tissue equilibrium magnetisation
    :param label_duration: The length of the labelling pulse
    :param signal_time: The time after the labelling pulse commences to generate signal.
    :param label_efficiency: The degree of inversion of the labelling pulse.
    :param partition_coefficient: The tissue-blood partition coefficient
    :param t1_arterial_blood: Longitudinal relaxation time of the arterial blood.
    :param t1_tissue: Longitudinal relaxation time of the tissue
    :param label_type: Determines the specific model to use: Pulsed ("pasl") or
        (pseudo)Continuous ("pcasl" or "casl") labelling
    :return: the difference magnetisation, :math:`\Delta M`
    """
    # divide perfusion_rate by 6000 to put into SI units
    perfusion_rate = perfusion_rate / 6000

    # calculate M0b, handling runtime divide-by-zeros
    m0_arterial_blood = np.divide(
        m0_tissue,
        partition_coefficient,
        out=np.zeros_like(partition_coefficient),
        where=partition_coefficient != 0,
    )
    # calculate T1', handling runtime divide-by-zeros
    flow_over_lambda = np.divide(
        perfusion_rate,
        partition_coefficient,
        out=np.zeros_like(partition_coefficient),
        where=partition_coefficient != 0,
    )
    one_over_t1_tissue = np.divide(
        1, t1_tissue, out=np.zeros_like(t1_tissue), where=t1_tissue != 0
    )
    denominator = one_over_t1_tissue + flow_over_lambda
    t1_prime: np.ndarray = np.divide(
        1, denominator, out=np.zeros_like(denominator), where=denominator != 0
    )
    condition_masks = compute_arrival_state_masks(
        transit_time, signal_time, label_duration
    )

    if label_type.lower() == PASL:
        # do GKM for PASL
        k: np.ndarray = (
            1 / t1_arterial_blood if t1_arterial_blood != 0 else 0
        ) - np.divide(1, t1_prime, out=np.zeros_like(t1_prime), where=t1_prime != 0)
        # if transit_time == signal_time then there is a divide-by-zero condition.
        # Calculate numerator and denominator separately for q_pasl_arriving
        numerator = np.exp(k * signal_time) * (
            np.exp(-k * transit_time) - np.exp(-k * signal_time)
        )
        denominator = k * (signal_time - transit_time)
        q_pasl_arriving = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        numerator = np.exp(k * signal_time) * (
            np.exp(-k * transit_time) - np.exp(-k * (transit_time + label_duration))
        )
        denominator = k * label_duration
        q_pasl_arrived = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        delta_m_arriving = (
            2
            * m0_arterial_blood
            * perfusion_rate
            * (signal_time - transit_time)
            * label_efficiency
            * (np.exp(-signal_time / t1_arterial_blood) if t1_arterial_blood > 0 else 0)
            * q_pasl_arriving
        )
        delta_m_arrived = (
            2
            * m0_arterial_blood
            * perfusion_rate
            * label_efficiency
            * label_duration
            * (np.exp(-signal_time / t1_arterial_blood) if t1_arterial_blood > 0 else 0)
            * q_pasl_arrived
        )
    elif label_type.lower() in [CASL, PCASL]:
        # do GKM for CASL/pCASL
        q_ss_arriving = 1 - np.exp(
            -np.divide(
                (signal_time - transit_time),
                t1_prime,
                out=np.zeros_like(t1_prime),
                where=t1_prime != 0,
            )
        )
        q_ss_arrived = 1 - np.exp(
            -np.divide(
                label_duration,
                t1_prime,
                out=np.zeros_like(t1_prime),
                where=t1_prime != 0,
            )
        )
        delta_m_arriving = (
            2
            * m0_arterial_blood
            * perfusion_rate
            * t1_prime
            * label_efficiency
            * (
                np.exp(-transit_time / t1_arterial_blood)
                if t1_arterial_blood != 0
                else np.zeros_like(transit_time)
            )
            * q_ss_arriving
        )
        delta_m_arrived = (
            2
            * m0_arterial_blood
            * perfusion_rate
            * t1_prime
            * label_efficiency
            * (
                np.exp(-transit_time / t1_arterial_blood)
                if t1_arterial_blood != 0
                else np.zeros_like(transit_time)
            )
            * np.exp(
                -np.divide(
                    (signal_time - label_duration - transit_time),
                    t1_prime,
                    out=np.zeros_like(t1_prime),
                    where=t1_prime != 0,
                )
            )
            * q_ss_arrived
        )
    else:
        raise ValueError(f"Invalid label type: {label_type}")

    delta_m: ArrayOrFloatT
    if isinstance(perfusion_rate, np.ndarray):
        if delta_m_arriving is None:
            raise ValueError("delta_m_arriving should be set")
        if delta_m_arrived is None:
            raise ValueError("delta_m_arrived should be set")
        delta_m = np.zeros(perfusion_rate.shape)
        # combine the different arrival states into delta_m
        delta_m[condition_masks["not_arrived"]] = 0.0
        delta_m[condition_masks["arriving"]] = delta_m_arriving[
            condition_masks["arriving"]
        ]
        delta_m[condition_masks["arrived"]] = delta_m_arrived[
            condition_masks["arrived"]
        ]
    else:
        if condition_masks["arrived"]:
            delta_m = delta_m_arrived
        elif condition_masks["arriving"]:
            delta_m = delta_m_arriving
        elif condition_masks["not_arrived"]:
            delta_m = 0
        else:
            raise ValueError(
                "Invalid combination of conditions, "
                "should have arrived, arriving or not arrived"
            )
        if isinstance(delta_m, np.ndarray):
            # We should have a single value in the array - raise an exception if not
            if delta_m.size != 1:
                raise ValueError("delta_m should be a single value")
            # Extract the value
            delta_m = delta_m[0]

    return delta_m


def calculate_delta_m_whitepaper(
    perfusion_rate: NDArray[np.floating],
    transit_time: NDArray[np.floating],
    m0_tissue: NDArray[np.floating],
    label_duration: float,
    signal_time: float,
    label_efficiency: float,
    partition_coefficient: NDArray[np.floating],
    t1_arterial_blood: float,
    label_type: str,
) -> NDArray[np.floating]:
    r"""Calculates the difference in magnetisation between the control
    and label condition (:math:`\Delta M`) using the single
    subtraction simplification from the  ASL Whitepaper consensus paper
    :cite:p:`Alsop2014`.

    :param perfusion_rate: Map of perfusion rate
    :param transit_time: Map of transit time
    :param m0_tissue: The tissue equilibrium magnetisation
    :param label_duration: The length of the labelling pulse
    :param signal_time: The time after the labelling pulse commences to generate signal.
    :param label_efficiency: The degree of inversion of the labelling pulse.
    :param partition_coefficient: The tissue-blood partition coefficient
    :param t1_arterial_blood: Longitudinal relaxation time of the arterial blood.
    :param label_type: Determines the specific model to use: Pulsed ("pasl") or
        (pseudo)Continuous ("pcasl" or "casl") labelling
    :return: the difference magnetisation, :math:`\Delta M`

    """
    # divide perfusion_rate by 6000 to put into SI units
    perfusion_rate = np.asarray(perfusion_rate) / 6000

    # calculate M0b, handling runtime divide-by-zeros
    m0_arterial_blood = np.divide(
        m0_tissue,
        partition_coefficient,
        out=np.zeros_like(partition_coefficient),
        where=partition_coefficient != 0,
    )
    condition_masks = compute_arrival_state_masks(
        transit_time, signal_time, label_duration
    )
    delta_m = np.zeros(perfusion_rate.shape)  # pre-allocate delta_m

    if label_type.lower() == PASL:
        delta_m_arriving = np.zeros_like(delta_m)
        # use simplified model for PASL
        delta_m_arrived = (
            2
            * m0_arterial_blood
            * perfusion_rate
            * label_duration
            * label_efficiency
            * np.exp(-signal_time / t1_arterial_blood)
            if t1_arterial_blood > 0
            else 0
        )

    elif label_type.lower() in [PCASL, CASL]:
        delta_m_arriving = np.zeros_like(delta_m)
        delta_m_arrived = (
            2
            * m0_arterial_blood
            * perfusion_rate
            * t1_arterial_blood
            * label_efficiency
            * np.exp(
                -(signal_time - label_duration) / t1_arterial_blood
                if t1_arterial_blood != 0
                else 0
            )
            * (
                1 - np.exp(-label_duration / t1_arterial_blood)
                if t1_arterial_blood != 0
                else 0
            )
        )

    # combine the different arrival states into delta_m
    delta_m[condition_masks["not_arrived"]] = 0.0
    delta_m[condition_masks["arriving"]] = delta_m_arriving[condition_masks["arriving"]]
    delta_m[condition_masks["arrived"]] = delta_m_arrived[condition_masks["arrived"]]
    return delta_m
