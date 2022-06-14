"""Diffusion weighting signal generation filter"""

from typing import List

import numpy as np

from mrimagetools.containers.image import BaseImageContainer
from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    isinstance_validator,
)


class DwiSignalFilter(BaseFilter):
    r"""Calculates the attenuation coefficient A for DWI

    **Inputs**

    :param 'adc': Apparent coefficient for each voxel, along
        the 4th dimension is the 3 direction x, y, z. So the 4th
        dimension contains ADC_x, ADC_y, ADC_z all in 3D.
    :type 'adc': BaseImageContainer
    :param 'b_values': List of b-values, must be positive float
    :type 'b_values' : List[float]
    :param 'b_vectors' : List of b-vectors, one for each b-values
    :type 'b_vectors': List[List[float, float, float]]

    **Outputs**

    :param 'image': Output image, must have the same affine
        as 'adc'. It containes the attenuation coefficient of each voxel
        the 4th dimension should be of the same length as 'b_values'
    :type 'image': BaseImageContainer

    **Metadata**
    :class:`DwiSignalFilter.outputs["image"].metadata` will be derived
    from the input ``'adc'``, with the following entries appended/updated:

    *``image_flavor`` = "dwi"
    * ``b_values`` = ``b_values``
    * ``b_vectors`` = ``b_vectors``

    """

    KEY_ADC = "adc"
    KEY_B_VALUES = "b_values"
    KEY_B_VECTORS = "b_vectors"
    KEY_IMAGE = "image"

    def __init__(self) -> None:
        super().__init__(name="DWI Signal")

    def _run(self) -> None:
        """Calculates the attenuation coefficient based on the inputs"""
        b_values: list = self.inputs[self.KEY_B_VALUES]
        b_vectors: List[List[float]] = self.inputs[self.KEY_B_VECTORS]
        adc: BaseImageContainer = self.inputs[self.KEY_ADC]
        self.outputs[self.KEY_IMAGE] = adc.clone()

        A_shape = np.shape(adc.image)
        A_image = np.zeros([A_shape[0], A_shape[1], A_shape[2], len(b_values)])
        for i in range(0, len(b_values)):
            A_image[:, :, :, i] = np.exp(
                -(
                    (b_values[i] * b_vectors[i][0] * adc.image[:, :, :, 0])
                    + (b_values[i] * b_vectors[i][1] * adc.image[:, :, :, 1])
                    + (b_values[i] * b_vectors[i][2] * adc.image[:, :, :, 2])
                )
            )

        self.outputs[self.KEY_IMAGE].image = A_image
        # update metadata
        self.outputs[self.KEY_IMAGE].metadata["ImageFlavor"] = "DWI"
        self.outputs[self.KEY_IMAGE].metadata["b_values"] = b_values
        self.outputs[self.KEY_IMAGE].metadata["b_vectors"] = b_vectors

    def _validate_inputs(self) -> None:
        """Checks the inputs meet their validation criteria
        'adc' must be derived from BaseImageContainer and have length of 3
            in the 4th dimension
        'b_values' must be a list of positive float
        'b_vectors' must be a list of List[float, float, float], with the
            same lenght as b_values
        """
        # first validate the input image
        input_image_validator = ParameterValidator(
            parameters={
                self.KEY_ADC: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
            }
        )

        input_image_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        adc: BaseImageContainer = self.inputs[self.KEY_ADC]
        num_vols = adc.shape[3]
        if not num_vols == 3:
            raise FilterInputValidationError(
                "The input image, 'adc' should have size 3 along the fourth dimension"
            )

        input_image_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # validate b_values and b_vectors
        b_validator = ParameterValidator(
            parameters={
                self.KEY_B_VALUES: Parameter(
                    validators=for_each_validator(isinstance_validator((float, int)))
                ),
                self.KEY_B_VECTORS: Parameter(
                    validators=for_each_validator(
                        for_each_validator(isinstance_validator((float, int)))
                    )
                ),
            }
        )

        b_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # check the lengths of b_values and b_vectors
        if not len(self.inputs[self.KEY_B_VALUES]) == len(
            self.inputs[self.KEY_B_VECTORS]
        ):
            raise FilterInputValidationError(
                "Inputs b_values and b_vectors need to have the same length"
            )

        # check that all the values in b_vectors are of length 3
        if not all([len(val) == 3 for val in self.inputs[self.KEY_B_VECTORS]]):
            raise FilterInputValidationError(
                "All entries in list 'b_vectors' should have length 3"
            )
