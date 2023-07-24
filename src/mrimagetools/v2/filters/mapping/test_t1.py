"""Tests for the T1 mapping module."""

from typing import Union

import numpy as np
import pytest
from numpy.typing import NDArray

from mrimagetools.v2.filters.mapping.t1 import (
    InversionRecoveryParameters,
    T1Model,
    VtrParameters,
    t1_mapping,
)


def _inversion_recovery_forward_model(
    s0: NDArray[np.floating],
    t1: NDArray[np.floating],
    inv_eff: NDArray[np.floating],
    repetition_times: Union[float, list[float]],
    inversion_times: list[float],
    model: T1Model = T1Model.GENERAL,
) -> NDArray[np.floating]:
    """The forward model for T1 mapping.

    :param s0: The S0 values of the voxels.
    :param t1: The T1 values of the voxels.
    :param inv_eff: The inversion efficiency of the voxels.
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
            1
            - 2 * inv_eff * np.exp(-inversion_time / t1)
            + (np.exp(-repetition_time / t1) if model == T1Model.GENERAL else 0)
        )
    if not isinstance(result, np.ndarray):
        raise TypeError("The result must be a NumPy array.")
    return result


def _vtr_forward_model(
    m: NDArray[np.floating],
    t1: NDArray[np.floating],
    repetition_times: Union[float, list[float]],
) -> NDArray[np.floating]:
    """The forward model for variable TR T1 mapping.

    :param m: The M values (where M = M_0 * exp^{-TE/T2}) of the voxels.
    :param t1: The T1 values of the voxels.
    :param repetition_times: The repetition times (TR) in seconds.

    :return: The signal intensity of the voxels.
    """

    if isinstance(repetition_times, float):
        repetition_times = [repetition_times]
    if m.shape != t1.shape:
        raise ValueError("The M and T1 arrays must have the same shape.")

    result = np.zeros(shape=t1.shape + (len(repetition_times),))

    # Loop over the repetition times
    for repetition_time_index, repetition_time in enumerate(repetition_times):
        # Calculate the signal intensity
        result[..., repetition_time_index] = m * (1 - np.exp(-repetition_time / t1))
    if not isinstance(result, np.ndarray):
        raise TypeError("The result must be a NumPy array.")
    return result


@pytest.mark.parametrize("model", [T1Model.GENERAL, T1Model.CLASSICAL])
def test_t1_mapping(model: T1Model) -> None:
    """Some basic tests to check the T1 mapping is calculating the correct values."""
    t1 = np.array([[1.1, 1.2], [1.3, 1.4]])
    s0 = np.array([[2.1, 2.2], [2.3, 2.4]])
    inv_eff = np.array([[0.98, 0.99], [1.0, 0.97]])

    # Create the parameters
    parameters = InversionRecoveryParameters(
        repetition_times=[0.1, 0.2, 0.3, 1, 2, 3],
        inversion_times=[0.05, 0.1, 0.2, 1, 2, 3],
        model=model,
    )

    # Calculate the signal intensity
    signal = _inversion_recovery_forward_model(
        s0=s0,
        t1=t1,
        inv_eff=inv_eff,
        repetition_times=parameters.repetition_times,
        inversion_times=parameters.inversion_times,
        model=model,
    )
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters)

    print(t1_mapping_results.t1)
    print(t1)
    # Check the results
    assert np.allclose(t1_mapping_results.t1, t1, rtol=1e-3, atol=1e-3)


def test_vtr_mapping() -> None:
    """Some basic tests to check the VTR T1 mapping is calculating the correct
    values."""
    t1 = np.array([[1.1, 1.2], [1.3, 1.4]])
    m = np.array([[2.1, 2.2], [2.3, 2.4]])

    # Create the parameters
    parameters = VtrParameters(
        repetition_times=[0.1, 0.2, 0.3, 1, 2, 3],
    )

    # Calculate the signal intensity
    signal = _vtr_forward_model(
        m=m,
        t1=t1,
        repetition_times=parameters.repetition_times,
    )
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t1_mapping_results.t1, t1, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("model", [T1Model.GENERAL, T1Model.CLASSICAL])
def test_t1_mapping_constant_repetition_time(model: T1Model) -> None:
    """Some basic tests to check the T1 mapping is calculating the correct values.
    Uses a constant repetition time."""
    t1 = np.array([[1.1, 1.2], [1.3, 1.4]])
    s0 = np.array([[2.1, 2.2], [2.3, 2.4]])
    inv_eff = np.array([[0.98, 0.99], [1.0, 0.97]])
    parameters = InversionRecoveryParameters(
        # Use the same repetition time for all inversion times
        repetition_times=3,
        inversion_times=[0.05, 0.1, 0.2, 1, 2, 3],
        model=model,
    )

    # Calculate the signal intensity
    signal = _inversion_recovery_forward_model(
        s0=s0,
        t1=t1,
        inv_eff=inv_eff,
        repetition_times=parameters.repetition_times,
        inversion_times=parameters.inversion_times,
        model=model,
    )
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t1_mapping_results.t1, t1, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("model", [T1Model.GENERAL, T1Model.CLASSICAL])
def test_t1_mapping_invert_polarity(model: T1Model) -> None:
    """Check that, when the polarity of the signal is inverted (in the early inversion
    times), the T1 mapping still works."""
    t1 = np.array([[1.1, 1.2], [1.3, 1.4]])
    s0 = np.array([[2.1, 2.2], [2.3, 2.4]])
    inv_eff = np.array([[0.98, 0.99], [1.0, 0.97]])
    parameters = InversionRecoveryParameters(
        # Use the same repetition time for all inversion times
        repetition_times=3,
        inversion_times=[0.05, 0.1, 0.2, 1, 2, 3],
        model=model,
    )

    # Calculate the signal intensity
    signal = _inversion_recovery_forward_model(
        s0=s0,
        t1=t1,
        inv_eff=inv_eff,
        repetition_times=parameters.repetition_times,
        inversion_times=parameters.inversion_times,
        model=model,
    )
    # Invert the polarity of the signal
    signal[..., :1] *= -1
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t1_mapping_results.t1, t1, rtol=1e-3, atol=1e-3)
