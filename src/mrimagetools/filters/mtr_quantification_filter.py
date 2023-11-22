"""Magnetisation Transfer Ratio Quantification Filter"""


import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.validators.fields import UnitField

KEY_IMAGE_SAT = "image_sat"
KEY_IMAGE_NOSAT = "image_nosat"
KEY_MTR = "mtr"
M_MODALITY = "modality"
M_QUANTITY = "Quantity"
M_UNITS = "Units"
M_IMAGE_TYPE = "ImageType"


class MtrQuantificationParameters(BaseModel):
    """Parameters for the MTR Quantification Filter

    The shapes and affines of the input images must match for the filter to run.
    i.e. they must be co-located in world-space."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_nosat: BaseImageContainer
    """An image without bound pool saturation applied"""

    image_sat: BaseImageContainer
    """An image with bound pool saturation applied"""

    @model_validator(mode="after")
    def check_shapes_match(self) -> "MtrQuantificationParameters":
        """Checks that the shapes of the input images match"""
        if self.image_nosat.shape != self.image_sat.shape:
            raise ValueError("Input images must have the same shape")
        return self

    @model_validator(mode="after")
    def check_affines_match(self) -> "MtrQuantificationParameters":
        """Checks that the affines of the input images match"""
        if not (self.image_nosat.affine == self.image_sat.affine).all():
            raise ValueError("Input images must have the same affine")
        return self


def mtr_quantification_filter(
    parameters: MtrQuantificationParameters,
) -> BaseImageContainer:
    r"""Calculates the Magnetisation Transfer Ratio (MTR) :cite:p:`Tofts2003_MT`
    based on input images with and without bound pool saturation.


    :param parameters: The parameters for the filter. See
        :class:`MtrQuantificationParameters` for more details.

    **Outputs**

    :return: An image of the magnetisation transfer ratio (MTR) in
      percentage units (pu).

    **Metadata**

    The returned image will be derived from the input ``'image_sat'``, with the
    following entries appended/updated:

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

    output_image = parameters.image_sat.clone()
    output_image.image = 100.0 * np.divide(
        parameters.image_nosat.image - parameters.image_sat.image,
        parameters.image_nosat.image,
        out=np.zeros_like(parameters.image_nosat.image),
        where=parameters.image_nosat.image != 0,
    )
    output_image.metadata.modality = "MTRmap"
    output_image.metadata.quantity = "MTR"
    output_image.metadata.units = UnitField(root="pu")
    output_image.metadata.image_type = ("DERIVED", "PRIMARY", "MTRmap", "NONE")
    return output_image
