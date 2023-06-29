"""Tests for the t2 mapping module."""

from typing import Union

import numpy as np
import pytest
from numpy.typing import NDArray

from mrimagetools.filters.mapping.t2 import T2MappingParameters, T2Model, t2_mapping


def _t2_mapping_forward_model(
    t2: NDArray[np.floating],
    a: NDArray[np.floating],
    offset: Union[NDArray[np.floating], None],
    echo_times: list[float],
    model: T2Model = T2Model.FULL,
) -> NDArray[np.floating]:
    """The forward model for t2 mapping.

    :param t2: The t2 values of the voxels.
    :param inv_eff: The inversion efficiency of the voxels.
    :param repetition_times: The repetition times (TR) in seconds.
    :param inversion_times: The inversion times (TI) in seconds.

    :return: The signal intensity of the voxels.
    """
    if a.shape != t2.shape:
        raise ValueError("The S0 and t2 arrays must have the same shape.")

    if offset is not None and a.shape != offset.shape:
        raise ValueError("The S0 and offset arrays must have the same shape.")

    if model == T2Model.FULL and offset is None:
        raise ValueError("The offset must be provided for the full model.")

    if model == T2Model.REDUCED and offset is not None:
        raise ValueError("The offset must not be provided for the reduced model.")

    result = np.zeros(shape=t2.shape + (len(echo_times),))

    # Loop over the echo times
    for echo_time_index, echo_time in enumerate(echo_times):
        # Calculate the signal intensity
        result[..., echo_time_index] = a * np.exp(-echo_time / t2) + (
            offset if offset is not None else 0
        )
    if not isinstance(result, np.ndarray):
        raise TypeError("The result must be a NumPy array.")
    return result


@pytest.mark.parametrize("model", [T2Model.FULL, T2Model.REDUCED])
def test_t2_mapping(model: T2Model) -> None:
    """Some basic tests to check the t2 mapping is calculating the correct values."""
    t2 = np.array([[1.1, 1.2], [1.3, 1.4]])
    a = np.array([[2.1, 2.2], [2.3, 2.4]])
    offset = np.array([[0.98, 0.99], [1.0, 0.97]]) if model == T2Model.FULL else None

    # Create the parameters
    parameters = T2MappingParameters(
        echo_times=[0.1, 0.2, 0.3, 1, 2, 3], model=model, skip_echos=0
    )

    # Calculate the signal intensity
    signal = _t2_mapping_forward_model(
        t2=t2,
        a=a,
        offset=offset,
        echo_times=parameters.echo_times,
        model=model,
    )
    t2_mapping_results = t2_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t2_mapping_results.t2, t2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("model", [T2Model.FULL, T2Model.REDUCED])
def test_echo_skipping(model: T2Model) -> None:
    """Test that the echo skipping is working correctly."""
    t2 = np.array([[1.1, 1.2], [1.3, 1.4]])
    a = np.array([[2.1, 2.2], [2.3, 2.4]])
    offset = np.array([[0.98, 0.99], [1.0, 0.97]]) if model == T2Model.FULL else None

    # Create the parameters
    parameters = T2MappingParameters(
        echo_times=[0.1, 0.2, 0.3, 1, 2, 3], model=model, skip_echos=1
    )

    # Calculate the signal intensity
    signal = _t2_mapping_forward_model(
        t2=t2,
        a=a,
        offset=offset,
        echo_times=parameters.echo_times,
        model=model,
    )

    # pollute the signal from the first echo with lots of noise
    signal[..., 0] = np.random.normal(loc=signal[..., 0], scale=signal[..., 0])

    # check the mapping still works
    t2_mapping_results = t2_mapping(signal=signal, parameters=parameters)

    # Check the results
    assert np.allclose(t2_mapping_results.t2, t2, rtol=1e-3, atol=1e-3)
