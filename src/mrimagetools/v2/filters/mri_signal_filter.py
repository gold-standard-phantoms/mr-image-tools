""" MRI Signal Filter """

from typing import Any, Optional

from mrimagetools.filters.mri_signal_filter import (
    MriSignalFilterImages,
    MriSignalFilterParameters,
    mri_signal_filter,
)
from mrimagetools.v2.containers.image import COMPLEX_IMAGE_TYPE, BaseImageContainer
from mrimagetools.v2.containers.image_metadata import AcqContrastType
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.utils.typing import T
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    from_list_validator,
    greater_than_equal_to_validator,
    isinstance_validator,
)


class MriSignalFilter(BaseFilter):
    r""" A filter that generates either the Gradient Echo, Spin Echo or
    Inversion Recovery MRI signal.

    * Gradient echo is with arbitrary excitation flip angle.
    * Spin echo assumes perfect 90° excitation and 180° refocusing pulses.
    * Inversion recovery can have arbitrary inversion pulse and excitation pulse flip
        angles.


    **Inputs**

    Input Parameters are all keyword arguments for the
    :class:`MriSignalFilter.add_inputs()` member function. They are also accessible via
    class constants, for example :class:`MriSignalFilter.KEY_T1`

    :param 't1':  Longitudinal relaxation time in seconds (>=0, non-complex data)
    :type 't1': BaseImageContainer
    :param 't2': Transverse relaxation time in seconds (>=0, non-complex data)
    :type 't2': BaseImageContainer
    :param 't2_star': Transverse relaxation time including time-invariant magnetic
        field inhomogeneities, only required for gradient echo (>=0, non-complex data)
    :type 't2_star': BaseImageContainer
    :param 'm0': Equilibrium magnetisation (non-complex data)
    :type 'm0': BaseImageContainer
    :param 'mag_enc': Added to M0 before relaxation is calculated,
        provides a means to encode another signal into the MRI signal (non-complex data)
    :type 'mag_enc': BaseImageContainer, optional.
    :param 'acq_contrast': Determines which signal model to use:
        ``"ge"`` (case insensitive) for Gradient Echo, ``"se"`` (case insensitive) for
        Spin Echo, ``"ir"`` (case insensitive) for Inversion Recovery.
    :type 'acq_contrast': str
    :param 'echo_time': The echo time in seconds (>=0)
    :type 'echo_time': float
    :param 'repetition_time': The repeat time in seconds (>=0)
    :type 'repetition_time': float
    :param 'excitation_flip_angle': Excitation pulse flip angle in degrees. Only used
        when ``'acq_contrast'`` is ``"ge"`` or ``"ir"``.  Defaults to 90.0
    :type 'excitation_flip_angle': float, optional
    :param 'inversion_flip_angle': Inversion pulse flip angle in degrees. Only used when
        ``acq_contrast`` is ``"ir"``. Defaults to 180.0
    :type 'inversion_flip_angle': float, optional
    :param 'inversion_time': The inversion time in seconds. Only used when
        ``'acq_contrast'`` is ``"ir"``. Defaults to 1.0.
    :param 'image_flavor': sets the metadata ``'image_flavor'`` in the output image to
    this.
    :type 'image_flavor': str

    **Outputs**

    Once run, the filter will populate the dictionary :class:`MriSignalFilter.outputs`
    with the following entries

    :param 'image': An image of the generated MRI signal. Will be of the same class
      as the input ``'m0'``
    :type 'image': BaseImageContainer

    **Output Image Metadata**

    The metadata in the output image :class:`MriSignalFilter.outputs["image"]` is
    derived from the input ``'m0'``. If the input ``'mag_enc'`` is present, its
    metadata is merged with precedence. In addition, following parameters are added:

    * ``'acq_contrast'``
    * ``'echo time'``
    * ``'excitation_flip_angle'``
    * ``'image_flavor'``
    * ``'inversion_time'``
    * ``'inversion_flip_angle'``
    * ``'mr_acquisition_type'` = "3D"

    Metadata entries for ``'units'`` and ``'quantity'`` will be removed.

    ``'image_flavor'`` is obtained (in order of precedence):

    #. If present, from the input ``'image_flavor'``
    #. If present, derived from the metadata in the input ``'mag_enc'``
    #. "OTHER"

    **Signal Equations**

    The following equations are used to compute the MRI signal:

    *Gradient Echo*

    .. math::
        S(\text{TE},\text{TR}, \theta_1) = \sin\theta_1\cdot(\frac{M_0
        \cdot(1-e^{-\frac{TR}{T_{1}}})}
        {1-\cos\theta_1 e^{-\frac{TR}{T_{1}}}-e^{-\frac{TR}{T_{2}}}\cdot
        \left(e^{-\frac{TR}{T_{1}}}-\cos\theta_1\right)}  + M_{\text{enc}})
        \cdot e^{-\frac{\text{TE}}{T^{*}_2}}

    *Spin Echo* (assuming 90° and 180° pulses)

    .. math::
       S(\text{TE},\text{TR}) = (M_0 \cdot (1-e^{-\frac{\text{TR}}{T_1}}) + M_{\text{enc}})
       \cdot e^{-\frac{\text{TE}}{T_2}}

    *Inversion Recovery*

    .. math::
        &S(\text{TE},\text{TR}, \text{TI}, \theta_1, \theta_2) =
        \sin\theta_1 \cdot (\frac{M_0(1-\left(1-\cos\theta_{2}\right)
        e^{-\frac{TI}{T_{1}}}-\cos\theta_{2}e^{-\frac{TR}{T_{1}}})}
        {1-\cos\theta_{1}\cos\theta_{2}e^{-\frac{TR}{T_{1}}}}+ M_\text{enc})
        \cdot e^{-\frac{TE}{T_{2}}}\\
        &\theta_1 = \text{excitation pulse flip angle}\\
        &\theta_2 = \text{inversion pulse flip angle}\\
        &\text{TI} = \text{inversion time}\\
        &\text{TR} = \text{repetition time}\\
        &\text{TE} = \text{echo time}\\

    """

    # Key constants
    KEY_T1 = "t1"
    KEY_T2 = "t2"
    KEY_T2_STAR = "t2_star"
    KEY_M0 = "m0"
    KEY_MAG_ENC = "mag_enc"
    KEY_ACQ_CONTRAST = "acq_contrast"
    KEY_ECHO_TIME = "echo_time"
    KEY_REPETITION_TIME = "repetition_time"
    KEY_EXCITATION_FLIP_ANGLE = "excitation_flip_angle"
    KEY_INVERSION_FLIP_ANGLE = "inversion_flip_angle"
    KEY_INVERSION_TIME = "inversion_time"
    KEY_IMAGE = "image"
    KEY_IMAGE_FLAVOR = "image_flavor"
    KEY_ACQ_TYPE = "mr_acquisition_type"
    KEY_BACKGROUND_SUPPRESSION = "background_suppression"

    # Value constants
    CONTRAST_GE = "ge"
    CONTRAST_SE = "se"
    CONTRAST_IR = "ir"

    def __init__(self) -> None:
        super().__init__(name="MRI Signal Model")

    def get_typed_input(self, key: str, input_type: type[T]) -> T:
        """Get a variable from the inputs, checking that the type is correct.
        (helps with mypy type checking"""
        value: Any = self.inputs.get(key)
        if isinstance(value, input_type):
            return value
        raise TypeError(
            f"Value with key {key} is of type {type(value)} and should be {input_type}"
        )

    def _run(self) -> None:
        # Image inputs
        t1: BaseImageContainer = self.inputs[self.KEY_T1]
        t2: BaseImageContainer = self.inputs[self.KEY_T2]
        t2_star: Optional[BaseImageContainer] = self.inputs.get(self.KEY_T2_STAR, None)
        m0: BaseImageContainer = self.inputs[self.KEY_M0]
        mag_enc: Optional[BaseImageContainer] = self.inputs.get(self.KEY_MAG_ENC, None)

        # Required inputs
        acq_contrast: AcqContrastType = self.inputs[self.KEY_ACQ_CONTRAST]
        echo_time: float = self.inputs[self.KEY_ECHO_TIME]
        repetition_time: float = self.inputs[self.KEY_REPETITION_TIME]

        # Optional inputs
        excitation_flip_angle: Optional[float] = self.inputs.get(
            self.KEY_EXCITATION_FLIP_ANGLE, None
        )
        image_flavor: Optional[str] = self.inputs.get(self.KEY_IMAGE_FLAVOR, None)
        inversion_flip_angle: Optional[float] = self.inputs.get(
            self.KEY_INVERSION_FLIP_ANGLE, None
        )
        inversion_time: Optional[float] = self.inputs.get(self.KEY_INVERSION_TIME, None)

        images = MriSignalFilterImages(
            t1=t1,
            t2=t2,
            t2_star=t2_star,
            m0=m0,
            mag_enc=mag_enc,
        )
        parameters = MriSignalFilterParameters(
            acq_contrast=acq_contrast,
            echo_time=echo_time,
            repetition_time=repetition_time,
        )
        if inversion_flip_angle is not None:
            parameters.inversion_flip_angle = inversion_flip_angle

        if inversion_time is not None:
            parameters.inversion_time = inversion_time

        if image_flavor is not None:
            parameters.image_flavor = image_flavor  # type: ignore

        if excitation_flip_angle is not None:
            parameters.excitation_flip_angle = excitation_flip_angle

        self.outputs["image"] = mri_signal_filter(images=images, parameters=parameters)

    def _validate_inputs(self) -> None:
        """Checks that the inputs meet their validation critera
        't1' must be derived from BaseImageContainer, >=0, and non-complex
        't2' must be derived from BaseImageContainer, >=0, and non-complex
        't2_star' must be derived from BaseImageContainer, >=0, and non-complex
            Only required if 'acq_contrast' == 'ge'
        'm0' must be derived from BaseImageContainer, and non-complex
        'mag_enc' (optional) must be derived from BaseImageContainer and non-complex
        'acq_contrast' must be a string and equal to "ge" or "se" (case insensitive)
        'echo_time' must be a float and >= 0
        'repetition_time' must be a float and >= 0
        'excitation_flip_angle' must be a float and >=0
        'inversion_flip_angle' must be a float and >=0
        'inversion_time' must be a float and >=0

        All images must have the same dimensions

        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_M0: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_T1: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_T2: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_T2_STAR: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ],
                    optional=True,
                ),
                self.KEY_MAG_ENC: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)], optional=True
                ),
                self.KEY_ACQ_CONTRAST: Parameter(
                    validators=[
                        isinstance_validator(str),
                        from_list_validator(
                            [self.CONTRAST_GE, self.CONTRAST_SE, self.CONTRAST_IR],
                            case_insensitive=True,
                        ),
                    ]
                ),
                self.KEY_ECHO_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_REPETITION_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_EXCITATION_FLIP_ANGLE: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ],
                    optional=True,
                ),
                self.KEY_INVERSION_FLIP_ANGLE: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ],
                    optional=True,
                ),
                self.KEY_INVERSION_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ],
                    optional=True,
                ),
                self.KEY_IMAGE_FLAVOR: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ],
                    optional=True,
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # Parameters that are conditionally required based on the value of "acq_contrast"
        # if the acquisition contrast is gradient echo ("ge")
        if self.inputs[self.KEY_ACQ_CONTRAST].lower() == self.CONTRAST_GE:
            # 't2_star' must be present in inputs
            if self.inputs.get(self.KEY_T2_STAR) is None:
                raise FilterInputValidationError(
                    "Acquisition contrast is ge, 't2_star' image required"
                )
        # if the acquisition contrast is gradient echo ("ge") or inversion recovery ("ir")
        if self.inputs[self.KEY_ACQ_CONTRAST].lower() in (
            self.CONTRAST_GE,
            self.CONTRAST_IR,
        ):
            # 'excitation_flip_angle' must be present in inputs
            if self.inputs.get(self.KEY_EXCITATION_FLIP_ANGLE) is None:
                raise FilterInputValidationError(
                    f"Acquisition contrast is {self.inputs[self.KEY_ACQ_CONTRAST]},"
                    " 'excitation_flip_angle' required"
                )

        # if the acquisition contrast is inversion recovery ("ir")
        if self.inputs[self.KEY_ACQ_CONTRAST].lower() == self.CONTRAST_IR:
            if self.inputs.get(self.KEY_INVERSION_FLIP_ANGLE) is None:
                raise FilterInputValidationError(
                    f"Acquisition contrast is {self.inputs[self.KEY_ACQ_CONTRAST]},"
                    " 'inversion_flip_angle' required"
                )
            if self.inputs.get(self.KEY_INVERSION_TIME) is None:
                raise FilterInputValidationError(
                    f"Acquisition contrast is {self.inputs[self.KEY_ACQ_CONTRAST]},"
                    " 'inversion_time' required"
                )
            if self.get_typed_input(self.KEY_REPETITION_TIME, float) < (
                self.get_typed_input(self.KEY_ECHO_TIME, float)
                + self.get_typed_input(self.KEY_INVERSION_TIME, float)
            ):
                raise FilterInputValidationError(
                    "repetition_time must be greater than echo_time + inversion_time"
                )

        # Check repetition_time is not < echo_time for ge and se
        if self.get_typed_input(self.KEY_REPETITION_TIME, float) < self.get_typed_input(
            self.KEY_ECHO_TIME, float
        ):
            raise FilterInputValidationError(
                "repetition_time must be greater than echo_time"
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

        # Check that all the input images are not of image_type == "COMPLEX_IMAGE_TYPE"
        for key in keys_of_images:
            if self.inputs[key].image_type == COMPLEX_IMAGE_TYPE:
                raise FilterInputValidationError(
                    f"{key} has image type {COMPLEX_IMAGE_TYPE}, this is not supported"
                )
