""" Affine Matrix Filter """

import numpy as np
from numpy.typing import NDArray


def affine_matrix_filter_function(
    rotation_angles: tuple[float, float, float] = (0, 0, 0),
    rotation_origin: tuple[float, float, float] = (0, 0, 0),
    translation: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    input_affine: NDArray[np.floating] = np.eye(4, dtype=np.float64),
    affine_last: NDArray[np.floating] = np.eye(4, dtype=np.float64),
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""A filter that creates an affine transformation matrix
    based on input parameters for rotation, translation, and scaling.

    Conventions are for RAS+ coordinate systems only

    :param rotation_angles: [:math:`\theta_x`, :math:`\theta_y`, :math:`\theta_z`]
        angles to rotate about the x, y and z axes in degrees(-180 to 180 degrees
        inclusive), defaults to (0, 0, 0)
    :param rotation_origin: [:math:`x_r`, :math:`y_r`, :math:`z_r`]
        coordinates of the point to perform rotations about, defaults to (0, 0, 0)
    :param translation: [:math:`\Delta x`, :math:`\Delta y`, :math:`\Delta z`]
        amount to translate along the x, y and z axes. defaults to (0, 0, 0)
    :param scale: [:math:`s_x`, :math:`s_y`, :math:`s_z`]
        scaling factors along each axis, defaults to (1, 1, 1)
    :param input_affine:  4x4 affine matrix to apply transformation to, defaults to
        `numpy.eye(4)`
    :param affine_last: 4x4 affine matrix to apply transformation to, defaults to
        `numpy.eye(4)`

    :return: a tuple of: the forward transformation and the inverse transformation
    (both are 4x4 affine matrices).
    """

    scale_matrix = np.array(
        (
            (scale[0], 0, 0, 0),
            (0, scale[1], 0, 0),
            (0, 0, scale[2], 0),
            (0, 0, 0, 1),
        )
    )
    translation_matrix = np.array(
        (
            (1, 0, 0, translation[0]),
            (0, 1, 0, translation[1]),
            (0, 0, 1, translation[2]),
            (0, 0, 0, 1),
        )
    )
    rotation_origin_translation_matrix = np.array(
        (
            (1, 0, 0, rotation_origin[0]),
            (0, 1, 0, rotation_origin[1]),
            (0, 0, 1, rotation_origin[2]),
            (0, 0, 0, 1),
        )
    )

    rotation_x_matrix = np.array(
        (
            (1, 0, 0, 0),
            (0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0]), 0),
            (0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0]), 0),
            (0, 0, 0, 1),
        )
    )
    rotation_y_matrix = np.array(
        (
            (np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1]), 0),
            (0, 1, 0, 0),
            (-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1]), 0),
            (0, 0, 0, 1),
        )
    )
    rotation_z_matrix = np.array(
        (
            (np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0, 0),
            (np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )

    # inverse of rotation_origin_translation_matrix doesn't need to be inverted,
    # it is simply 2*np.eye(4) - rotation_origin_translation_matrix
    inv_rotation_origin_translation_matrix = (
        2 * np.eye(4) - rotation_origin_translation_matrix
    )

    # combine
    output_affine: np.ndarray = (
        affine_last
        @ scale_matrix
        @ translation_matrix
        @ rotation_origin_translation_matrix
        @ rotation_z_matrix
        @ rotation_y_matrix
        @ rotation_x_matrix
        @ inv_rotation_origin_translation_matrix
        @ input_affine
    )

    inverse_output_affine = np.linalg.inv(output_affine)
    return (output_affine, inverse_output_affine)
