"""Tests for the T1 mapping module."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from mrimagetools.filters.mapping.t1 import T1MappingParameters, t1_mapping


def _t1_mapping_forward_model(
    s0: NDArray,
    t1: NDArray,
    repetition_times: Union[float, list[float]],
    inversion_times: list[float],
) -> NDArray:
    """The forward model for T1 mapping.

    :param s0: The S0 values of the voxels.
    :param t1: The T1 values of the voxels.
    :param repetition_times: The repetition times (TR) in seconds.
    :param inversion_times: The inversion times (TI) in seconds.

    :return: The signal intensity of the voxels.
    """
    if isinstance(repetition_times, float):
        repetition_times = [repetition_times] * len(inversion_times)
    assert len(repetition_times) == len(
        inversion_times
    ), "The number of repetition times and inversion times must be the same."
    if s0.shape != t1.shape:
        raise ValueError("The S0 and T1 arrays must have the same shape.")

    result = np.zeros(shape=t1.shape + (len(inversion_times),))

    # Loop over the repetition/inversion times
    for inversion_time_index, (inversion_time, repetition_time) in enumerate(
        zip(inversion_times, repetition_times)
    ):
        # Calculate the signal intensity
        result[..., inversion_time_index] = s0 * (
            1 - 2 * np.exp(-inversion_time / t1) + np.exp(-repetition_time / t1)
        )
    if not isinstance(result, np.ndarray):
        raise TypeError("The result must be a NumPy array.")
    return result


def test_t1_mapping() -> None:
    """Some basic tests to check the T1 mapping is calculating the correct values."""
    t1 = np.array([[1.1, 1.2], [1.3, 1.4]])
    s0 = np.array([[2.1, 2.2], [2.3, 2.4]])
    parameters = T1MappingParameters(
        repetition_times=[0.1, 0.2, 0.3],
        inversion_times=[0.05, 0.1, 0.2],
    )

    # Calculate the signal intensity
    signal = _t1_mapping_forward_model(
        s0=s0,
        t1=t1,
        repetition_times=parameters.repetition_times,
        inversion_times=parameters.inversion_times,
    )
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t1_mapping_results.t1, t1, rtol=1e-4, atol=1e-4)
    assert np.allclose(t1_mapping_results.s0, s0, rtol=1e-4, atol=1e-4)


def test_t1_mapping_constant_repetition_time() -> None:
    """Some basic tests to check the T1 mapping is calculating the correct values.
    Uses a constant repetition time."""
    t1 = np.array([[1.1, 1.2], [1.3, 1.4]])
    s0 = np.array([[2.1, 2.2], [2.3, 2.4]])
    parameters = T1MappingParameters(
        # Use the same repetition time for all inversion times
        repetition_times=0.1,
        inversion_times=[0.05, 0.1, 0.2],
    )

    # Calculate the signal intensity
    signal = _t1_mapping_forward_model(
        s0=s0,
        t1=t1,
        repetition_times=parameters.repetition_times,
        inversion_times=parameters.inversion_times,
    )
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t1_mapping_results.t1, t1, rtol=1e-4, atol=1e-4)
    assert np.allclose(t1_mapping_results.s0, s0, rtol=1e-4, atol=1e-4)
