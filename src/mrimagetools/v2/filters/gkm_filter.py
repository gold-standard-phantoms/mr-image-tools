""" General Kinetic Model Filter """

import logging
from typing import Final, Literal, Optional, Union

import numpy as np

from mrimagetools.filters.gkm_filter import GkmParameters, gkm_filter
from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    from_list_validator,
    greater_than_equal_to_validator,
    isinstance_validator,
    range_inclusive_validator,
)

logger = logging.getLogger(__name__)


class GkmFilter(BaseFilter):
    r"""
    A filter that generates the ASL signal using the General Kinetic Model.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`GkmFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`GkmFilter.KEY_PERFUSION_RATE`

    :param 'perfusion_rate': Map of perfusion rate, in ml/100g/min (>=0)
    :type 'perfusion_rate': BaseImageContainer
    :param 'transit_time'  Map of the time taken for the labelled bolus
      to reach the voxel, seconds (>=0).
    :type 'transit_time': BaseImageContainer
    :param 'm0': The tissue equilibrium magnetisation, can be a map or single value (>=0).
    :type 'perfusion_rate': BaseImageContainer or float
    :param 'label_type': Determines which GKM equations to use:

      * "casl" OR "pcasl" (case insensitive) for the continuous model
      * "pasl" (case insensitive) for the pulsed model

    :type 'label_type': str
    :param 'label_duration': The length of the labelling pulse, seconds (0 to 100 inclusive)
    :type 'label_duration': float
    :param 'signal_time': The time after labelling commences to generate signal,
      seconds (0 to 100 inclusive)
    :type 'signal_time': float
    :param 'label_efficiency': The degree of inversion of the labelling (0 to 1 inclusive)
    :type 'label_efficiency': float
    :param 'lambda_blood_brain': The blood-brain-partition-coefficient (0 to 1 inclusive)
    :type 'lambda_blood_brain': float or BaseImageContainer
    :param 't1_arterial_blood': Longitudinal relaxation time of arterial blood,
        seconds (0 exclusive to 100 inclusive)
    :type 't1_arterial_blood': float
    :param 't1_tissue': Longitudinal relaxation time of the tissue,
        seconds (0 to 100 inclusive, however voxels with ``t1 = 0`` will have ``delta_m = 0``)
    :type 't1_tissue': BaseImageContainer
    :param 'model': The model to use to generate the perfusion signal:

      * "full" for the full "Buxton" General Kinetic Model :cite:p:`Buxton1998`
      * "whitepaper" for the simplified model, derived from the quantification
        equations the ASL Whitepaper consensus paper :cite:p:`Alsop2014`.

      Defaults to "full".

    :type 'model': str

    **Outputs**

    Once run, the filter will populate the dictionary :class:`GkmFilter.outputs`
    with the following entries

    :param 'delta_m': An image with synthetic ASL perfusion contrast. This will
      be the same class as the input 'perfusion_rate'
    :type 'delta_m': BaseImageContainer

    **Metadata**

    The following parameters are added to
    :class:`GkmFilter.outputs["delta_m"].metadata`:

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

    # Key constants
    KEY_PERFUSION_RATE = "perfusion_rate"
    KEY_TRANSIT_TIME = "transit_time"
    KEY_M0 = "m0"
    KEY_LABEL_TYPE = "label_type"
    KEY_LABEL_DURATION = "label_duration"
    KEY_SIGNAL_TIME = "signal_time"
    KEY_LABEL_EFFICIENCY = "label_efficiency"
    KEY_LAMBDA_BLOOD_BRAIN = "lambda_blood_brain"
    KEY_T1_ARTERIAL_BLOOD = "t1_arterial_blood"
    KEY_T1_TISSUE = "t1_tissue"
    KEY_DELTA_M = "delta_m"
    KEY_MODEL = "model"
    M_POST_LABEL_DELAY = "post_label_delay"
    M_BOLUS_CUT_OFF_FLAG = "bolus_cut_off_flag"
    M_BOLUS_CUT_OFF_DELAY_TIME = "bolus_cut_off_delay_time"
    M_GKM_MODEL = "gkm_model"

    # Value constants
    CASL = "casl"
    PCASL = "pcasl"
    PASL = "pasl"

    MODEL_FULL: Final[str] = "full"
    MODEL_WP: Final[str] = "whitepaper"

    def __init__(self) -> None:
        super().__init__(name="General Kinetic Model")

    def _run(self) -> None:
        """Generates the delta_m signal based on the inputs"""

        perfusion_rate: BaseImageContainer = self.inputs[self.KEY_PERFUSION_RATE]
        transit_time: BaseImageContainer = self.inputs[self.KEY_TRANSIT_TIME]
        t1_tissue: BaseImageContainer = self.inputs[self.KEY_T1_TISSUE]
        m0: Union[float, BaseImageContainer] = self.inputs[self.KEY_M0]
        lambda_blood_brain: Union[float, BaseImageContainer] = self.inputs[
            self.KEY_LAMBDA_BLOOD_BRAIN
        ]

        label_duration: float = self.inputs[self.KEY_LABEL_DURATION]
        signal_time: float = self.inputs[self.KEY_SIGNAL_TIME]
        label_efficiency: float = self.inputs[self.KEY_LABEL_EFFICIENCY]
        t1_arterial_blood: float = self.inputs[self.KEY_T1_ARTERIAL_BLOOD]
        model: Literal["full", "whitepaper"] = self.inputs[self.KEY_MODEL]
        label_type = self.inputs[self.KEY_LABEL_TYPE].lower()

        gkm_parameters = GkmParameters(
            perfusion_rate=perfusion_rate,
            transit_time=transit_time,
            t1_tissue=t1_tissue,
            label_duration=label_duration,
            signal_time=signal_time,
            label_efficiency=label_efficiency,
            t1_arterial_blood=t1_arterial_blood,
            model=model,
            label_type=label_type,
            m0=m0,
            lambda_blood_brain=lambda_blood_brain,
        )

        self.outputs[self.KEY_DELTA_M] = gkm_filter(gkm_parameters=gkm_parameters)

    def _validate_inputs(self) -> None:
        """Checks that the inputs meet their validation criteria
        'perfusion_rate' must be derived from BaseImageContainer and be >= 0
        'transit_time' must be derived from BaseImageContainer and be >= 0
        'm0' must be either a float or derived from BaseImageContainer and be >= 0
        'label_type' must be a string and equal to "CASL" OR "pCASL" OR "PASL"
        'label_duration" must be a float between 0 and 100
        'signal_time' must be a float between 0 and 100
        'label_efficiency' must be a float between 0 and 1
        'lambda_blood_brain' must be a float between 0 and 1
        't1_arterial_blood' must be a float between 0 and 100

        all BaseImageContainers supplied should be the same dimensions
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_PERFUSION_RATE: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_TRANSIT_TIME: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_M0: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator((BaseImageContainer, float)),
                    ]
                ),
                self.KEY_T1_TISSUE: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_LABEL_TYPE: Parameter(
                    validators=from_list_validator(
                        [self.CASL, self.PCASL, self.PASL], case_insensitive=True
                    )
                ),
                self.KEY_LABEL_DURATION: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_SIGNAL_TIME: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_LABEL_EFFICIENCY: Parameter(
                    validators=[
                        range_inclusive_validator(0, 1),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_LAMBDA_BLOOD_BRAIN: Parameter(
                    validators=[
                        range_inclusive_validator(0, 1),
                        isinstance_validator((BaseImageContainer, float)),
                    ]
                ),
                self.KEY_T1_ARTERIAL_BLOOD: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_MODEL: Parameter(
                    validators=from_list_validator(
                        [self.MODEL_FULL, self.MODEL_WP], case_insensitive=True
                    ),
                    default_value=self.MODEL_FULL,
                ),
            }
        )

        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # Check that all the input images are all the same dimensions
        input_keys = self.inputs.keys()
        keys_of_images = [
            key
            for key in input_keys
            if isinstance(self.inputs[key], BaseImageContainer)
        ]

        list_of_image_shapes = [self.inputs[key].shape for key in keys_of_images]
        if list_of_image_shapes.count(list_of_image_shapes[0]) != len(
            list_of_image_shapes
        ):
            raise FilterInputValidationError(
                [
                    "Input image shapes do not match.",
                    [
                        f"{keys_of_images[i]}: {list_of_image_shapes[i]}, "
                        for i in range(len(list_of_image_shapes))
                    ],
                ]
            )
        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}
