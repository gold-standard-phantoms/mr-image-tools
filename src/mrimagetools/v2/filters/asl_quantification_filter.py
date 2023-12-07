"""ASL quantification filter class"""

from mrimagetools.filters.asl_quantification_filter import (
    KEY_PERFUSION_RATE,
    AslQuantificationFilterImages,
    AslQuantificationFilterParameters,
    asl_quantification_filter,
    full_model,
    white_paper,
)
from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.filters.gkm_filter import GkmFilter
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    and_validator,
    for_each_validator,
    from_list_validator,
    greater_than_equal_to_validator,
    greater_than_validator,
    isinstance_validator,
    range_inclusive_validator,
    shape_validator,
)


class AslQuantificationFilter(BaseFilter):
    r"""
    A filter that calculates the perfusion rate for arterial spin labelling data.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:
    `AslQuantificationFilter.add_input()`
    member function. They are also accessible via class constants, for example
    :class:`AslQuantificationFilter.KEY_CONTROL`

    :param 'control': the control image (3D or 4D timeseries)
    :type 'control': BaseImageContainer
    :param 'label': the label image (3D or 4D timeseries)
    :type 'label': BaseImageContainer
    :param 'm0': equilibrium magnetisation image
    :type 'm0': BaseImageContainer
    :param 'label_type': the type of labelling used: "pasl" for pulsed ASL
      "pcasl" or "casl" for for continuous ASL.
    :type 'label_type': str
    :param 'lambda_blood_brain': The blood-brain-partition-coefficient
    (0 to 1 inclusive)
    :type 'lambda_blood_brain': float
    :param 'label_duration': The temporal duration of the labelled bolus, seconds
      (0 or greater). For PASL this is equivalent to :math:`\text{TI}_1`
    :type 'label_duration': float
    :param 'post_label_delay': The duration between the end of the labelling
        pulse and the imaging excitation pulse, seconds (0 or greater).
        For PASL this is equivalent to :math:`\text{TI}`.
        If ``'model'=='full'`` then this must be a list and the length of this
        must match the number of unique entries in ``'multiphase_index'``.
    :type 'post_label_delay': float or List[float]
    :param 'label_efficiency': The degree of inversion of the labelling
      (0 to 1 inclusive)
    :type 'label_efficiency': float
    :param 't1_arterial_blood': Longitudinal relaxation time of arterial blood,
        seconds (greater than 0)
    :type 't1_arterial_blood': float
    :param 't1_tissue': Longitudinal relaxation time of the tissue, seconds
        (greater than 0). Required if ``'model'=='full'``
    :type 't1_tissue': float or BaseImageContainer
    :param 'gkm_model': defines which model to use

        * 'whitepaper' uses the single-subtraction white paper equation
        * 'full' least square fitting to the full GKM.

    :type 'gkm_model': str
    :param 'multiphase_index': A list the same length as the fourth dimension
        of the label image that defines which phase each image belongs to,
        and is also the corresponding index in the ``'post_label_delay'`` list.
        Required if ``'gkm_model'=='full'``.

    **Outputs**

    :param 'perfusion_rate': map of the calculated perfusion rate
    :type 'perfusion_rate': BaseImageContainer

    If ``'gkm_model'=='full'`` the following are also output:

    :param 'transit_time': The estimated transit time in seconds.
    :type 'transit_time': BaseImageContainer
    :param 'std_error': The standard error of the estimate of the fit.
    :type 'std_error': BaseImageContainer
    :param 'perfusion_rate_err': One standard deviation error in the fitted
    perfusion rate.
    :type 'perfusion_rate_err': BaseImageContainer
    :param 'transit_time_err': One standard deviation error in the fitted
      transit time.
    :type 'transit_time_err': BaseImageContainer

    **Quantification Model**

    The following equations are used to calculate the perfusion rate, depending
    on the input ``gkm_model``:

    :'whitepaper': simplified single subtraction equations :cite:p:`Alsop2014`.

      * for pCASL/CASL see :class:`AslQuantificationFilter.asl_quant_wp_casl`
      * for PASL see :class:`AslQuantificationFilter.asl_quant_wp_pasl`.

    :'full': Lease squares fitting to the full General Kinetic Model
        :cite:p:`Buxton1998`.
      See :class:`AslQuantificationFilter.asl_quant_lsq_gkm`.

    """

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

    def __init__(self) -> None:
        super().__init__(name="ASL Quantification")

    def _run(self) -> None:
        """Calculates the perfusion rate based on the inputs"""

        # # Image inputs
        control: BaseImageContainer = self.inputs[self.KEY_CONTROL]
        label: BaseImageContainer = self.inputs[self.KEY_LABEL]
        m0: BaseImageContainer = self.inputs[self.KEY_M0]

        # # t1_tissue is optional
        t1_tissue = None
        if self.KEY_T1_TISSUE in self.inputs:
            t1_tissue = self.inputs[self.KEY_T1_TISSUE]
        multiphase_index = None
        if self.KEY_MULTIPHASE_INDEX in self.inputs:
            multiphase_index = self.inputs[self.KEY_MULTIPHASE_INDEX]

        images = AslQuantificationFilterImages(
            control=control, label=label, m0=m0, t1_tissue=t1_tissue
        )

        parameters = AslQuantificationFilterParameters(
            label_type=self.inputs[self.KEY_LABEL_TYPE],
            lambda_blood_brain=self.inputs[self.KEY_LAMBDA_BLOOD_BRAIN],
            label_duration=self.inputs[self.KEY_LABEL_DURATION],
            post_label_delay=self.inputs[self.KEY_POST_LABEL_DELAY],
            label_efficiency=self.inputs[self.KEY_LABEL_EFFICIENCY],
            t1_arterial_blood=self.inputs[self.KEY_T1_ARTERIAL_BLOOD],
            key_model=self.inputs[self.KEY_MODEL],
            multiphase_index=multiphase_index,
        )

        output_image = asl_quantification_filter(input_images=images)
        self.outputs[self.KEY_PERFUSION_RATE] = output_image
        if self.inputs[self.KEY_MODEL] == self.WHITEPAPER:
            # single subtraction quantification
            perfusion_rate = white_paper(input_images=images, parameters=parameters)

            self.outputs[self.KEY_PERFUSION_RATE].image = perfusion_rate
            output_image.metadata.estimation_algorithm = self.ESTIMATION_ALGORITHM[
                self.WHITEPAPER
            ]
        elif self.inputs[self.KEY_MODEL] == self.FULL:
            results = full_model(
                input_images=images,
                parameters=parameters,
            )
            self.outputs[self.KEY_PERFUSION_RATE].image = results[KEY_PERFUSION_RATE]

            output_image.metadata.estimation_algorithm = self.ESTIMATION_ALGORITHM[
                self.FULL
            ]
            output_image.metadata.multiphase_index = None
            # when using the full model there are additional outputs
            for key in [
                self.KEY_PERFUSION_RATE_ERR,
                self.KEY_TRANSIT_TIME,
                self.KEY_TRANSIT_TIME_ERR,
                self.KEY_STD_ERROR,
            ]:
                self.outputs[key] = self.outputs[self.KEY_PERFUSION_RATE].clone()
                self.outputs[key].image = results[key]
                self.outputs[key].metadata.asl_context = self.FIT_IMAGE_NAME[
                    key
                ].lower()
                self.outputs[key].metadata.units = self.FIT_IMAGE_UNITS[key]
                self.outputs[key].metadata.image_type = (
                    "DERIVED",
                    "PRIMARY",
                    "PERFUSION",
                    self.FIT_IMAGE_NAME[key].upper(),
                )

    def _validate_inputs(self) -> None:
        """Checks the inputs meet their validation criteria
        'control' must be derived from BaseImageContainer
        'label' must be derived from BaseImageContainer
        'm0' must be derived from BaseImageContainer
        'label_type' must be a str and equal to "pasl", "casl" or "pcasl"
        (case-insensitive)
        'model' must be a str and equal to "whitepaper"
        'label_duration' must be a float and >= 0
        'post_label_delay' must be a float and >= 0
        'label_efficiency' must be a float between 0 and 1 inclusive
        'lambda_blood_brain' must be a float between 0 and 1 inclusive
        't1_arterial_blood' must be a float and >0
        't1_tissue' must be a float or BaseImageContainer and >0
        'multiphase_index' must be a list of integers

        'multiphase_index' should match the length of the 4th dimension of
        the 'label' image.
        'multiphase_index' and 't1_tissue' are required if 'model' is 'full'.
        'control' and 'label' must have the same shape
        The shape of 'm0' must match the first 3 dimensions of 'label'

        """

        input_validator = {
            "common": ParameterValidator(
                parameters={
                    self.KEY_M0: Parameter(
                        validators=[
                            isinstance_validator(BaseImageContainer),
                        ]
                    ),
                    self.KEY_CONTROL: Parameter(
                        validators=[
                            isinstance_validator(BaseImageContainer),
                        ]
                    ),
                    self.KEY_LABEL: Parameter(
                        validators=[
                            isinstance_validator(BaseImageContainer),
                        ]
                    ),
                    self.KEY_LABEL_TYPE: Parameter(
                        validators=from_list_validator(
                            [GkmFilter.CASL, GkmFilter.PCASL, GkmFilter.PASL],
                            case_insensitive=True,
                        )
                    ),
                    self.KEY_LABEL_DURATION: Parameter(
                        validators=[
                            greater_than_equal_to_validator(0),
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
                            isinstance_validator(float),
                        ]
                    ),
                    self.KEY_T1_ARTERIAL_BLOOD: Parameter(
                        validators=[
                            greater_than_validator(0),
                            isinstance_validator(float),
                        ]
                    ),
                    self.KEY_MODEL: Parameter(
                        validators=from_list_validator(
                            [self.WHITEPAPER, self.FULL],
                            case_insensitive=True,
                        )
                    ),
                },
                post_validators=[
                    shape_validator([self.KEY_CONTROL, self.KEY_LABEL, self.KEY_M0], 3)
                ],
            ),
            "full": ParameterValidator(
                parameters={
                    self.KEY_POST_LABEL_DELAY: Parameter(
                        validators=[
                            for_each_validator(
                                and_validator(
                                    [
                                        greater_than_equal_to_validator(0),
                                        isinstance_validator(float),
                                    ]
                                )
                            )
                        ]
                    ),
                    self.KEY_MULTIPHASE_INDEX: Parameter(
                        validators=[for_each_validator(isinstance_validator(int))]
                    ),
                    self.KEY_T1_TISSUE: Parameter(
                        validators=[
                            isinstance_validator((float, BaseImageContainer)),
                            greater_than_validator(0),
                        ]
                    ),
                },
                post_validators=[shape_validator([self.KEY_CONTROL, self.KEY_LABEL])],
            ),
            "whitepaper": ParameterValidator(
                parameters={
                    self.KEY_POST_LABEL_DELAY: Parameter(
                        validators=[
                            greater_than_equal_to_validator(0),
                            isinstance_validator(float),
                        ]
                    ),
                }
            ),
        }
        # validate the common parameters
        input_validator["common"].validate(
            self.inputs, error_type=FilterInputValidationError
        )
        # validate the model specific parameters
        input_validator[self.inputs[self.KEY_MODEL]].validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # extra validation for the full GKM
        if self.inputs[self.KEY_MODEL] == self.FULL:
            label_shape = self.inputs[self.KEY_LABEL].shape
            # length of 'multiphase_index' should match the length of the label image
            # in the 4th dimension
            if not len(self.inputs[self.KEY_MULTIPHASE_INDEX]) == label_shape[3]:
                raise FilterInputValidationError(
                    "The length of 'multiphase_index' must be equal to the length"
                    "of the 'label' image in the 4th dimension "
                )

            if not len(self.inputs[self.KEY_POST_LABEL_DELAY]) == len(
                set(self.inputs[self.KEY_MULTIPHASE_INDEX])
            ):
                raise FilterInputValidationError(
                    "The length of 'post_label_delay' should be the same as the"
                    "number of unique entries in 'multiphase_index'"
                )
