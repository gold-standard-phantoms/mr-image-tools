""" MRI Signal Filter """

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.containers.image_metadata import AcqContrastType, ImageMetadata

# Value constants
CONTRAST_GE = "ge"
CONTRAST_SE = "se"
CONTRAST_IR = "ir"


@dataclass
class MriSignalFilterImages:
    """MRI Signal Filter Images"""

    t1: BaseImageContainer
    """ Longitudinal relaxation time in seconds (>=0, non-complex data) """

    t2: BaseImageContainer
    """ Transverse relaxation time in seconds (>=0, non-complex data) """

    t2_star: Optional[BaseImageContainer]
    """ Transverse relaxation time including time-invariant magnetic field
    inhomogeneities, only required for gradient echo (>=0, non-complex data) """

    m0: BaseImageContainer
    """ Equilibrium magnetisation (non-complex data) """

    mag_enc: Optional[BaseImageContainer]
    """ Added to M0 before relaxation is calculated, provides a means to encode another
    signal into the MRI signal (non-complex data) """


class MriSignalFilterParameters(BaseModel):
    """MRI Signal Filter Parameters"""

    acq_contrast: AcqContrastType
    """ Determines which signal model to use:
    ``"ge"`` (case insensitive) for Gradient Echo, ``"se"`` (case insensitive) for
    Spin Echo, ``"ir"`` (case insensitive) for Inversion Recovery. """

    echo_time: float = Field(..., ge=0)
    """ The echo time in seconds (>=0) """

    repetition_time: float = Field(..., ge=0)
    """ The repeat time in seconds (>=0) """

    excitation_flip_angle: float = Field(90.0, ge=0, le=360)
    """ Excitation pulse flip angle in degrees. Only used when ``acq_contrast`` is
    ``"ge"`` or ``"ir"``.  Defaults to 90.0 """

    inversion_flip_angle: float = Field(180.0, ge=0, le=360)
    """ Inversion pulse flip angle in degrees. Only used when ``acq_contrast`` is
    ``"ir"``. Defaults to 180.0 """

    inversion_time: float = Field(1.0, ge=0)
    """ The inversion time in seconds. Only used when ``acq_contrast`` is ``"ir"``.
    Defaults to 1.0. """

    image_flavor: Optional[Literal["PERFUSION", "DIFFUSION", "OTHER"]] = None
    """ sets the metadata ``image_flavor`` in the output image to this. """


def mri_signal_filter(
    images: MriSignalFilterImages, parameters: MriSignalFilterParameters
) -> BaseImageContainer:
    r""" A filter that generates either the Gradient Echo, Spin Echo or
    Inversion Recovery MRI signal.

    * Gradient echo is with arbitrary excitation flip angle.
    * Spin echo assumes perfect 90° excitation and 180° refocusing pulses.
    * Inversion recovery can have arbitrary inversion pulse and excitation pulse flip
        angles.

    :param images: The image data used to generate the MRI signal. See
        :class:`MriSignalFilterImages` for details.

    :param parameters: The parameters used to generate the MRI signal. See
        :class:`MriSignalFilterParameters` for details.


    :return: An image of the generated MRI signal. Will be of the same class
      as the input ``'m0'``

    **Output Image Metadata**

    The metadata in the output image is derived from the input ``'m0'``. If the input
    ``'mag_enc'`` is present, its metadata is merged with precedence. In addition,
    following parameters are added:

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
    t1 = images.t1.image
    t2 = images.t2.image
    m0 = images.m0.image

    metadata = ImageMetadata()
    mag_enc: np.ndarray
    if images.mag_enc is not None:
        mag_enc = images.mag_enc.image
        metadata = ImageMetadata(
            **{
                **metadata.model_dump(exclude_none=True),
                **images.mag_enc.metadata.model_dump(exclude_none=True),
            }
        )
    else:
        mag_enc = np.zeros(t1.shape)

    # mag_enc might not have "image_flavor" set
    if metadata.image_flavor is None:
        metadata.image_flavor = "OTHER"

    # if present override image_flavor with the input
    if parameters.image_flavor is not None:
        metadata.image_flavor = parameters.image_flavor

    acq_contrast = parameters.acq_contrast
    echo_time = parameters.echo_time
    repetition_time = parameters.repetition_time

    mri_signal = np.zeros(t1.shape)

    # pre-calculate the exponent exp(-echo_time/t2) as it is used multiple times
    exp_te_t2 = np.exp(-np.divide(echo_time, t2, out=np.zeros_like(t2), where=t2 != 0))

    # pre-calculate the exponent exp(-repetition_time/t1) as it is used multiple times
    exp_tr_t1 = np.exp(
        -np.divide(repetition_time, t1, out=np.zeros_like(t1), where=t1 != 0)
    )

    # add common fields to metadata
    metadata.acq_contrast = acq_contrast
    metadata.echo_time = echo_time
    metadata.repetition_time = repetition_time
    metadata.mr_acquisition_type = (  # 2D not currently supported so everything is 3D
        "3D"
    )

    # Gradient Echo Contrast. Equation is from p246 in the book MRI from Picture to
    # Proton, second edition, 2006, McRobbie et. al.
    if acq_contrast.lower() == CONTRAST_GE:
        if images.t2_star is None:
            raise ValueError(
                "Gradient echo requires t2_star to be set in the input images"
            )
        t2_star = images.t2_star.image
        flip_angle = np.radians(parameters.excitation_flip_angle)
        # pre-calculate the exponent exp(-echo_time/t2_star)
        exp_t2_star = np.exp(
            -np.divide(
                echo_time, t2_star, out=np.zeros_like(t2_star), where=t2_star != 0
            )
        )
        # pre-calculate the exponent exp(-repetition_time/t2)
        exp_tr_t2 = np.exp(
            -np.divide(repetition_time, t2, out=np.zeros_like(t2), where=t2 != 0)
        )

        # pre-calculate the numerator and denominator for use in np.divide to avoid
        # runtime divide-by-zero
        numerator = m0 * (1 - exp_tr_t1)
        denominator = (
            1
            - np.cos(flip_angle) * exp_tr_t1
            - exp_tr_t2 * (exp_tr_t1 - np.cos(flip_angle))
        )

        mri_signal = (
            np.sin(flip_angle)
            * (
                np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(denominator),
                    where=denominator != 0,
                )
                + mag_enc
            )
            * exp_t2_star
        )
        metadata.excitation_flip_angle = parameters.excitation_flip_angle

    # Spin Echo Contrast, equation is the standard spin-echo signal equation assuming a
    # 90° excitation pulse and 180° refocusing pulse. Equation is from p69 in the book
    # MRI from Picture to Proton, second edition, 2006, McRobbie et. al.
    elif acq_contrast.lower() == CONTRAST_SE:
        mri_signal = (
            m0
            * (
                1
                - np.exp(
                    -np.divide(
                        repetition_time, t1, out=np.zeros_like(t1), where=t1 != 0
                    )
                )
            )
            + mag_enc
        ) * exp_te_t2
        # for spin echo the flip angle is assumed to be 90°
        metadata.excitation_flip_angle = 90.0

    # Inversion Recovery contrast.  Equation is from equation 7 in
    # http://www.paul-tofts-phd.org.uk/talks/ismrm2009_rt.pdf
    elif acq_contrast.lower() == CONTRAST_IR:
        flip_angle = np.radians(parameters.excitation_flip_angle)
        inversion_time = parameters.inversion_time
        inversion_flip_angle = np.radians(parameters.inversion_flip_angle)
        # pre-calculate the exponent exp(-inversion_time/t1)
        exp_ti_t1 = np.exp(
            -np.divide(inversion_time, t1, out=np.zeros_like(t1), where=t1 != 0)
        )
        numerator = m0 * (
            1
            - (1 - np.cos(inversion_flip_angle)) * exp_ti_t1
            - np.cos(inversion_flip_angle) * exp_tr_t1
        )

        denominator = 1 - np.cos(flip_angle) * np.cos(inversion_flip_angle) * exp_tr_t1

        mri_signal = (
            np.sin(flip_angle)
            * (
                np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(denominator),
                    where=denominator != 0,
                )
                + mag_enc
            )
            * exp_te_t2
        )
        # add ir specific metadata
        metadata.excitation_flip_angle = parameters.excitation_flip_angle
        metadata.inversion_flip_angle = parameters.inversion_flip_angle
        metadata.inversion_time = inversion_time

    output_image = images.m0.clone()
    output_image.image = mri_signal
    # merge the metadata field with the constructed one (we don't want to merge)
    output_image.metadata = ImageMetadata(
        **{
            **output_image.metadata.model_dump(exclude_none=True),
            **metadata.model_dump(exclude_none=True),
        }
    )
    output_image.metadata.units = None
    output_image.metadata.quantity = None
    return output_image
