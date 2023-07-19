"""Apparent Diffusion Coefficient Quantification filter"""


import numpy as np

from mrimagetools.containers.image import BaseImageContainer
from mrimagetools.containers.image_metadata import ImageMetadata
from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.validators.fields import UnitField
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    isinstance_validator,
)


class AdcQuantificationFilter(BaseFilter):
    r"""Calculates the apparent diffusion coefficienc (ADC) :cite:p:`Tofts2003_DWI`
    based on input diffusion weighted images.

    **Inputs**

    :param 'dwi': Diffusion weighted image with different b-values and b-vectors
      along the 4th dimension. There must be at least two volumes.
    :type 'dwi': BaseImageContainer
    :param 'b_values': List of b-values, one for each dwi volume. One of these must
      be equal to 0, and the length of values should be the same as the number of
      dwi volumes.
    :type 'b_values': List[float]
    :param 'b_vectors': List of b-vectors, one for each dwi volume. The number of
      vectors must be the same as the number of dwi volumes.
    :type 'b_vectors': List[List[float, float, float]]

    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`AdcQuantificationFilter.outputs` with the following entries:

    :param 'adc': An image of the calculated apparent diffusion coefficient in
      units of :math:`mm^2/s`, with a volume for each non-zero b-value supplied.
    :type 'adc': BaseImageContainer

    **Metadata**
    :class:`AdcQuantificationFilter.outputs["adc"].metadata` will be derived
    from the input ``'dwi'``, with the following entries appended/updated:

    * ``modality`` = "ADCmap"
    * ``Quantity`` = "ADC"
    * ``Units`` = "mm^2/s"
    * ``ImageType`` = ["DERIVED", "PRIMARY", "ADCmap", "None"]
    * ``b_values`` = ``b_values``
    * ``b_vectors`` = ``b_vectors``


    **Equation**

    For each The Apparent Diffusion Coefficient (ADC) is calculated using the
    Stejskal-Tanner formula:

    .. math::

        \begin{align}
        &D = - \frac{ln(\frac{S(b)}{S(0)})}{b}\\
        &\text{where}\\
        &S(b)=\text{signal from dwi with b-value b}\\
        &S(0)\text{signal from image with b = 0}\\
        &D = \text{calculated diffusivity}
        \end{align}

    """

    KEY_DWI = "dwi"
    KEY_B_VALUES = "b_values"
    KEY_B_VECTORS = "b_vectors"
    KEY_ADC = "adc"

    def __init__(self) -> None:
        super().__init__(name="ADC Quantification")

    def _run(self) -> None:
        """Calculates the apparent diffusion coefficient based on the inputs"""
        b_values: list = self.inputs[self.KEY_B_VALUES]
        b_vectors: list[list[float]] = self.inputs[self.KEY_B_VECTORS]
        dwi: BaseImageContainer = self.inputs[self.KEY_DWI]
        self.outputs[self.KEY_ADC] = dwi.clone()

        # determine which DWI volume is corresponds with b = 0
        index_b0 = b_values.index(0)
        index_b = list(range(0, len(b_values)))
        index_b.pop(index_b0)  # remove the b = 0 index
        dwi_b0 = dwi.image[:, :, :, index_b0]

        safelog = lambda x: np.log(x, np.zeros_like(x), where=x > 0)

        adc = np.stack(
            [
                (
                    -(
                        safelog(
                            np.divide(
                                dwi.image[:, :, :, idx],
                                dwi_b0,
                                out=np.zeros_like(dwi_b0),
                                where=dwi_b0 > 0,
                            )
                        )
                        / b_values[idx]
                    )
                    if b_values[idx] != 0
                    else 0
                )
                for idx in index_b
            ],
            axis=3,
        )

        self.outputs[self.KEY_ADC].image = adc
        # update metadata
        output_metadata: ImageMetadata = self.outputs[self.KEY_ADC].metadata
        output_metadata.modality = "ADCmap"
        output_metadata.quantity = "ADC"
        output_metadata.units = UnitField.model_validate("mm^2/s")
        output_metadata.image_type = (
            "DERIVED",
            "PRIMARY",
            "ADCmap",
            "NONE",
        )
        output_metadata.b_values = b_values
        output_metadata.b_vectors = b_vectors

    def _validate_inputs(self) -> None:
        """Checks the inputs meet their validation criteria
        'dwi' must be derived from BaseImageContainer and have length of at least 2
          in the 4th dimension.
        'b_values' must be a list, the same length as the length of 'dwi' along the
          4th dimension. One value exclusively should be equal to 0.
        'b_vectors' must be a list of List[float, float, float], the same as the length of 'dwi'
        """
        # first validate the input image
        input_image_validator = ParameterValidator(
            parameters={
                self.KEY_DWI: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
            }
        )

        input_image_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        dwi: BaseImageContainer = self.inputs[self.KEY_DWI]
        num_vols = dwi.shape[3]
        if not num_vols > 1:
            raise FilterInputValidationError(
                "The input image, 'dwi' should have size 2 or greater along"
                "the fourth dimension"
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
        for key in [self.KEY_B_VALUES, self.KEY_B_VECTORS]:
            if not len(self.inputs[key]) == num_vols:
                raise FilterInputValidationError(
                    f"Input {key} should have a length equal to the number ofdwi"
                    f" volumes ({num_vols}). The length of {key} is"
                    " len(self.inputs[key])"
                )

        # check that one of the values of b_values == 0
        if not [val == 0 for val in self.inputs[self.KEY_B_VALUES]].count(1):
            raise FilterInputValidationError(
                "Input 'b_values' should have one value == 0, corresponding"
                "to the b = 0 image"
            )

        # check that all the values in b_vectors are of length 3
        if not all(len(val) == 3 for val in self.inputs[self.KEY_B_VECTORS]):
            raise FilterInputValidationError(
                "All entries in list 'b_vectors' should have length 3"
            )
