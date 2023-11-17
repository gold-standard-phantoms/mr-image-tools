""" Add noise filter """
import logging
from typing import Union

import numpy as np

from mrimagetools.v2.containers.image import (
    INVERSE_DOMAIN,
    SPATIAL_DOMAIN,
    BaseImageContainer,
)

logger = logging.getLogger(__name__)


def add_noise_filter_function(
    input_image: BaseImageContainer,
    snr: float,
    reference_image: Union[BaseImageContainer, None] = None,
) -> BaseImageContainer:
    """A filter that adds normally distributed random noise to an input image.

    :param input_image: An input image which noise will be added to. Can be either
      scalar or complex. If it is complex, normally distributed random noise will be
      added to both real and imaginary parts.
    :param snr: the desired signal-to-noise ratio (>= 0). A value of zero means that no
      noise is added to the input image.
    :param reference_image: The reference image that is used to calculate the
    amplitude of the random noise to add to `'image'`. The shape of this must match the
      shape of `'image'`. If this is not supplied then `'image'` will be used for
      calculating the noise amplitude.

    :return: The input image with noise added.

    `'reference_image'` can be in a different data domain to the `'image'`.  For
    example, `'image'` might be in the inverse domain (i.e. fourier transformed)
    whereas `'reference_image'` is in the spatial domain.
    Where data domains differ the following scaling is applied to the noise amplitude:
        * `'image'` is `SPATIAL_DOMAIN` and 'reference_image' is `INVERSE_DOMAIN`: 1/N
        * `'image'` is `INVERSE_DOMAIN` and 'reference_image' is `SPATIAL_DOMAIN`: N
    Where N is `reference_image.image.size`

    The noise is added pseudo-randomly based on the state of numpy.random. This should
    be appropriately controlled prior to running the filter.

    Note that the actual SNR (as calculated using "A comparison of two methods for
    measuring the signal to noise ratio on MR images", PMB, vol 44, no. 12, pp.N261-N264
    (1999)) will not match the desired SNR under the following circumstances:
        * `'image'` is `SPATIAL_DOMAIN` and `'reference_image'` is `INVERSE_DOMAIN`
        * `'image'` is `INVERSE_DOMAIN` and `'reference_image'` is `SPATIAL_DOMAIN`
    In the second case, performing an inverse fourier transform on the output image
    with noise results in a spatial domain image where the calculated SNR matches the
    desired SNR. This is how the :class:`AddNoiseFilter` is used within the
    :class:`AddComplexNoiseFilter`
    """

    noise_amplitude_scaling: float = 1.0  # default if domains match

    reference_image_local: BaseImageContainer
    if isinstance(reference_image, BaseImageContainer):
        reference_image_local = reference_image
    else:
        reference_image_local = input_image

    # Otherwise correct for differences in scaling due to fourier transform
    logger.debug("input image domain is %s", input_image.data_domain)
    logger.debug("reference_image domain is %s", reference_image_local.data_domain)
    if (
        input_image.data_domain == SPATIAL_DOMAIN
        and reference_image_local.data_domain == INVERSE_DOMAIN
    ):
        noise_amplitude_scaling = 1.0 / np.sqrt(reference_image_local.image.size)
    if (
        input_image.data_domain == INVERSE_DOMAIN
        and reference_image_local.data_domain == SPATIAL_DOMAIN
    ):
        noise_amplitude_scaling = np.sqrt(reference_image_local.image.size)

    # Calculate the noise amplitude (i.e. its standard deviation) using the non-zero
    # voxels in the magnitude of the reference image (in case it is complex)
    logger.debug("Noise amplitude scaling: %s", noise_amplitude_scaling)

    noise_amplitude = (
        noise_amplitude_scaling
        * np.mean(
            np.abs(reference_image_local.image[reference_image_local.image.nonzero()])
        )
        / (snr)
    )

    logger.debug("Noise amplitude: %s", noise_amplitude)

    # Make an image container for the image with noise
    image_with_noise: BaseImageContainer = input_image.clone()

    # Create noise arrays with mean=0, sd=noise_amplitude, and same dimensions
    # as the input image.
    if input_image.image.dtype in [np.complex128, np.complex64]:
        # Data are complex, create the real and imaginary components separately
        real_image = np.real(input_image.image)
        imag_image = np.imag(input_image.image)
        if not isinstance(noise_amplitude, float):
            raise TypeError("noise_amplitude must be a float")
        norm1 = np.random.normal(0, noise_amplitude, input_image.shape)
        norm2 = np.random.normal(0, noise_amplitude, input_image.shape)
        image_with_noise.image = (real_image + norm1) + 1j * (imag_image + norm2)
    else:
        # Data are not complex
        noise_matrix = np.random.normal(0, noise_amplitude, input_image.shape)
        image_with_noise.image = input_image.image + noise_matrix
    return image_with_noise
