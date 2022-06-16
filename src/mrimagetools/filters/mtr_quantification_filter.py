"""Magnetisation Transfer Ratio Quantification Filter"""

import numpy as np

from mrimagetools.containers.image import BaseImageContainer
from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    Validator,
    isinstance_validator,
)

image_shape_validator_generator = lambda images: Validator(
    func=lambda d: all([image in d for image in images])
    and all([isinstance(d[image], BaseImageContainer) for image in images])
    and [d[image].shape for image in images].count(d[images[0]].shape) == len(images),
    criteria_message=f"{images} must all have the same shapes",
)

image_affine_validator_generator = lambda images: Validator(
    func=lambda data: all([image in data for image in images])
    and all([isinstance(data[image], BaseImageContainer) for image in images])
    and [
        (data[image].affine == data[images[0]].affine).all() for image in images
    ].count(True)
    == len(images),
    criteria_message=f"{images} must all have the same shapes",
)


class MtrQuantificationFilter(BaseFilter):
    r"""Calculates the Magnetisation Transfer Ratio (MTR) :cite:p:`Tofts2003_MT`
    based on input images with and without bound pool saturation.

    **Inputs**

    :param 'image_nosat': An image without bound pool saturation applied.
    :type 'image_nosat': BaseImageContainer
    :param 'image_sat': An image with bound pool saturation applied.
    :type 'image_sat': BaseImageContainer

    Both ``'image_nosat'`` and ``'image_sat'`` should have the same shape and
    affine (i.e. are co-located in world-space).

    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`MtrQuantificationFilter.outputs` with the following entries:

    :param 'mtr': An image of the magnetisation transfer ratio (MTR) in
      percentage units (pu).
    :type 'mtr': BaseImageContainer

    **Metadata**

    :class:`MtrQuantificationFilter.outputs["mtr"].metadata` will be derived
    from the input ``'image_sat'``, with the following entries appended/updated:

    * ``modality`` = "MTRmap"
    * ``Quantity`` = "MTR"
    * ``Units`` = "pu"
    * ``ImageType`` = ["DERIVED", "PRIMARY", "MTRmap", "None"]

    **Equation**

    The Magnetisation Transfer Ratio (MTR) is calculated using the following
    equation:

    .. math::

        \begin{align}
        &\text{MTR} =100 \cdot  \frac{S_0 - S_s}{S_0}\\
        &\text{where}\\
        &S_0 = \text{signal without bound pool saturation}\\
        &S_s = \text{signal with bound pool saturation}\\
        \end{align}

    """

    KEY_IMAGE_SAT = "image_sat"
    KEY_IMAGE_NOSAT = "image_nosat"
    KEY_MTR = "mtr"
    M_MODALITY = "modality"
    M_QUANTITY = "Quantity"
    M_UNITS = "Units"
    M_IMAGE_TYPE = "ImageType"

    def __init__(self) -> None:
        super().__init__(name="MTR Quantification")

    def _run(self) -> None:
        """Calculates the magnetisationt transfer ratio based on the inputs"""
        image_nosat: BaseImageContainer = self.inputs[self.KEY_IMAGE_NOSAT]
        image_sat: BaseImageContainer = self.inputs[self.KEY_IMAGE_SAT]

        self.outputs[self.KEY_MTR] = image_sat
        self.outputs[self.KEY_MTR].image = 100.0 * np.divide(
            image_nosat.image - image_sat.image,
            image_nosat.image,
            out=np.zeros_like(image_nosat.image),
            where=image_nosat.image != 0,
        )
        self.outputs[self.KEY_MTR].metadata[self.M_MODALITY] = "MTRmap"
        self.outputs[self.KEY_MTR].metadata[self.M_QUANTITY] = "MTR"
        self.outputs[self.KEY_MTR].metadata[self.M_UNITS] = "pu"
        self.outputs[self.KEY_MTR].metadata[self.M_IMAGE_TYPE] = [
            "DERIVED",
            "PRIMARY",
            "MTRmap",
            "None",
        ]

    def _validate_inputs(self) -> None:
        """Checks the inputs meet their validation criteria
        'image_nosat' must be derived from BaseImageContainer
        'image_sat' must be derived from BaseImageContainer

        The shapes and affines of 'image_nosat' and 'image_sat' must match.

        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE_NOSAT: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
                self.KEY_IMAGE_SAT: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
            },
            post_validators=[
                image_shape_validator_generator(
                    [self.KEY_IMAGE_NOSAT, self.KEY_IMAGE_SAT]
                ),
                image_affine_validator_generator(
                    [self.KEY_IMAGE_NOSAT, self.KEY_IMAGE_SAT]
                ),
            ],
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
