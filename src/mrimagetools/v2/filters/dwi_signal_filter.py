"""Diffusion weighting signal generation filter"""

import copy
from typing import List, Union

import numpy as np

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    greater_than_equal_to_validator,
    isinstance_validator,
)


class DwiSignalFilter(BaseFilter):
    r"""Calculates the attenuation coefficient A for DWI

    **Inputs**

    :param 'adc': Apparent coefficient for each voxel, along
        the 4th dimension is the 3 direction x, y, z. So the 4th
        dimension contains ADC_x, ADC_y, ADC_z all in 3D.
    :param 'b_values': List of b-values, must be positive float
    :param 'b_vectors': List of b-vectors, one for each b-values if
        it is not normalized then the actual b_values will be processed
    :param 's0': Image with no diffusion weighting applied. must be
        3D with the three dimension of equal shape to the first three
        dimension of adc

    **Outputs**

    :param 'attenuation': Output image, must have the same affine
        as 'adc'. It containes the attenuation coefficient of each voxel
        the 4th dimension should be of the same length as 'b_values'
    :param 'dwi': Output image, must have the same affine
        as 'adc'. It containes fully encoded MRI signal of each voxel
        the 4th dimension should be of the same length as 'b_values'
        if s0 was NOT provided then 'dwi' = 'attenuation'

    **Metadata**
    :class:`DwiSignalFilter.outputs["attenuation"].metadata` will be derived
    from the input ``'adc'``, with the following entries appended/updated:

    *``image_flavor`` = "dwi"
    * ``b_values`` = ``b_values``
    * ``b_vectors`` = ``b_vectors``

    """

    KEY_ADC = "adc"
    KEY_B_VALUES = "b_values"
    KEY_B_VECTORS = "b_vectors"
    KEY_S0 = "s0"
    KEY_ATTENUATION = "attenuation"
    KEY_DWI = "dwi"
    M_IMAGE_FLAVOR = "image_flavor"

    def __init__(self) -> None:
        super().__init__(name="DWI Signal")

    def _run(self) -> None:
        """Calculates the attenuation coefficient based on the inputs"""
        b_values: list = self.inputs[self.KEY_B_VALUES]
        b_vectors: list[list[float]] = self.inputs[self.KEY_B_VECTORS]
        adc: BaseImageContainer = self.inputs[self.KEY_ADC]
        s0: Union[BaseImageContainer, None] = self.inputs.get(self.KEY_S0, None)

        self.outputs[self.KEY_ATTENUATION] = adc.clone()
        self.outputs[self.KEY_DWI] = adc.clone()

        true_b_values = copy.deepcopy(b_values)
        normalized_b_vectors = copy.deepcopy(b_vectors)
        for i, b_value in enumerate(b_values):
            true_b_values[i] = b_value * np.linalg.norm(
                b_vectors[i]
            )  # calculating true b values
            normalized_b_vectors[i] = b_vectors[i] / np.linalg.norm(
                b_vectors[i]
            )  # type: ignore # normalizing b vectors

        attenuation_shape = np.shape(adc.image)
        attenuation_image = np.zeros(
            [
                attenuation_shape[0],
                attenuation_shape[1],
                attenuation_shape[2],
                len(b_values),
            ]
        )
        dwi_image = copy.deepcopy(attenuation_image)

        for i, true_b_value in enumerate(true_b_values):
            sum_for_exp = 0
            for dimension in range(0, 3):
                sum_for_exp += (
                    true_b_value
                    * normalized_b_vectors[i][dimension]
                    * adc.image[:, :, :, dimension]
                )
            attenuation_image[:, :, :, i] = np.exp(-sum_for_exp)
            if s0 is not None:
                dwi_image[:, :, :, i] = np.multiply(
                    s0.image[:, :, :], attenuation_image[:, :, :, i]
                )
            if s0 is None:
                dwi_image[:, :, :, i] = attenuation_image[:, :, :, i]

        self.outputs[self.KEY_ATTENUATION].image = attenuation_image
        self.outputs[self.KEY_DWI].image = dwi_image
        # update metadata
        self.outputs[self.KEY_ATTENUATION].metadata.image_flavor = "DWI"
        self.outputs[self.KEY_ATTENUATION].metadata.b_values = (
            b_values  # for now the input b_val is returned
        )
        self.outputs[self.KEY_ATTENUATION].metadata.b_vectors = (
            b_vectors  # for now the input b_vect is returned
        )
        self.outputs[self.KEY_DWI].metadata.image_flavor = "DWI"
        self.outputs[self.KEY_DWI].metadata.b_values = b_values
        self.outputs[self.KEY_DWI].metadata.b_vectors = b_vectors

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
                    validators=[
                        for_each_validator(isinstance_validator((float, int))),
                        for_each_validator(greater_than_equal_to_validator(0)),
                    ]
                ),
                self.KEY_B_VECTORS: Parameter(
                    validators=[
                        for_each_validator(
                            for_each_validator(isinstance_validator((float, int)))
                        ),
                    ]
                ),
            }
        )

        b_validator.validate(self.inputs, error_type=FilterInputValidationError)

        s0_validator = ParameterValidator(
            parameters={
                self.KEY_S0: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ],
                    optional=True,
                ),
            }
        )

        s0_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # check the lengths of b_values and b_vectors
        if not len(self.inputs[self.KEY_B_VALUES]) == len(
            self.inputs[self.KEY_B_VECTORS]
        ):
            raise FilterInputValidationError(
                "Inputs b_values and b_vectors need to have the same length"
            )

        # check that all the values in b_vectors are of length 3
        if not all(len(val) == 3 for val in self.inputs[self.KEY_B_VECTORS]):
            raise FilterInputValidationError(
                "All entries in list 'b_vectors' should have length 3"
            )
        # chek that s0 is 3D
        if not self.inputs.get(self.KEY_S0) is None:
            if len(np.shape(self.inputs[self.KEY_S0].image)) != 3:
                raise FilterInputValidationError("s0 is not 3D")

        # check the shape of s0 and adc
        if not self.inputs.get(self.KEY_S0) is None:
            if np.shape(self.inputs[self.KEY_ADC].image[:, :, :, 0]) != np.shape(
                self.inputs[self.KEY_S0].image
            ):
                raise FilterInputValidationError(
                    "adc and s0 have different first three dimension"
                )
