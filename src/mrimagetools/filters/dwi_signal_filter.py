"""Diffusion weighting signal generation filter"""

import copy
from typing import Union

import numpy as np
from numpy.typing import NDArray

from mrimagetools.v2.containers.image import BaseImageContainer


def dwi_signal_filter_function(
    adc: BaseImageContainer,
    b_values: list[float],
    b_vectors: list[list[float]],
    s0: Union[BaseImageContainer, None],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Calculates the attenuation coefficient A for DWI

    :param adc: Apparent coefficient for each voxel, along
        the 4th dimension is the 3 direction x, y, z. So the 4th
        dimension contains ADC_x, ADC_y, ADC_z all in 3D.
    :param b_values: List of b-values, must be positive
    :param b_vectors: List of b-vectors, one for each b-values if
        it is not normalized then the actual b_values will be processed
    :param s0: Image with no diffusion weighting applied. must be
        3D with the three dimension of equal shape to the first three
        dimension of adc

    :return: a tuple of: the attenuation output image and the dwi output image.
        The attenuation output image must have the same affine as 'adc'.
        It containes the attenuation coefficient of each voxel
        the 4th dimension should be of the same length as 'b_values'
        The dwi output image, must have the same affine as 'adc'.
        It containes fully encoded MRI signal of each voxelthe 4th dimension
        should be of the same length as 'b_values'if s0 was NOT provided then
        'dwi' = 'attenuation'.
    """

    true_b_values = copy.deepcopy(b_values)
    normalized_b_vectors = copy.deepcopy(b_vectors)

    for i, b_value in enumerate(b_values):
        scaling_factor = float(np.linalg.norm(b_vectors[i]))

        # calculating true b values
        true_b_values[i] = b_value * scaling_factor

        array_format_normalized_b_vector: NDArray[np.floating] = np.divide(
            b_vectors[i], scaling_factor
        )

        # normalizing b vectors
        normalized_b_vectors[i] = list(array_format_normalized_b_vector)

    attenuation_shape = np.shape(adc.image)
    attenuation_image = np.zeros(
        [
            attenuation_shape[0],
            attenuation_shape[1],
            attenuation_shape[2],
            len(b_values),
        ],
        dtype=float,
    )
    dwi_image = copy.deepcopy(attenuation_image)

    for i, true_b_value in enumerate(true_b_values):
        sum_for_exp = np.zeros(adc.image[:, :, :, 0].shape, dtype=float)
        for dimension in range(0, 3):
            adc_image = adc.image
            current_adc = adc_image[:, :, :, dimension]
            sum_for_exp += (
                true_b_value * normalized_b_vectors[i][dimension] * current_adc
            )
        attenuation_image[:, :, :, i] = np.exp(-sum_for_exp)
        if s0 is not None:
            dwi_image[:, :, :, i] = np.multiply(
                s0.image[:, :, :], attenuation_image[:, :, :, i]
            )
        if s0 is None:
            dwi_image[:, :, :, i] = attenuation_image[:, :, :, i]

    return (attenuation_image, dwi_image)
