"""A module for T1 mapping."""

import logging
from dataclasses import dataclass
from typing import Final, Union
from typing import Final, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.optimize._lsq.least_squares import OptimizeResult
from pydantic import Field, root_validator, validator
from scipy.optimize import OptimizeResult, least_squares

from mrimagetools.validators.parameter_model import ParameterModel

logger = logging.getLogger(__name__)

class T1MappingParameters(ParameterModel):
    """The parameters for T1 mapping."""

    repetition_times: Union[float, list[float]] = Field(..., gt=0.0)
    """The repetition time(s) (TR) in seconds. May be a scalar or a list of floats.
    In the first instance, the same TR is used for all inversion times.
    In the second, the TRs are used in the same order as the inversion times."""

    inversion_times: list[float] = Field(..., gt=0.0)
    """The inversion time(s) (TI) in seconds. Must be a list of floats. The inversion
    times must be in ascending order."""

    @validator("inversion_times")
    def check_inversion_times_order(cls, v: list[float]) -> list[float]:
        """Check that the inversion times are in ascending order."""
        if v != sorted(v):
            raise ValueError("The inversion times must be in ascending order.")
        return v
    @root_validator
    def check_list_lengths(cls, values: dict) -> dict:
        """If the repetition is a list, check that it is the same length as the
        inversion times list."""
        repetition_times: Union[float, list[float]] = values.get(
            "repetition_times"
        )  # type:ignore
        inversion_times: list[float] = values.get("inversion_times")  # type:ignore
        if isinstance(repetition_times, list):
            if len(repetition_times) != len(inversion_times):
                raise ValueError(
                    "If repetition_times is a list, it must be the same length as "
                    "inversion_times"
                )
        return values


@dataclass
class T1MappingResults:
    """The results of the T1 mapping."""

    t1: NDArray[np.floating]
    """An ND NumPy array with the estimated T1 values in seconds."""

    s0: NDArray[np.floating]
    """An ND NumPy array with the estimated S0 values."""

    inv_eff: NDArray[np.floating]
    """An ND NumPy array with the estimated inversion efficiency values."""


def _optimise(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    parameters: T1MappingParameters,
) -> NDArray:
    """The optimisation routine for T1 mapping.

    :param x: The parameters to be optimised. This is a 1D NumPy array with the
        parameters in the following order: T1, S0, inv_eff.

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the inversion times.

    :param parameters: The parameters for the T1 mapping.
        See :class:`T1MappingParameters`.

    :return: The residual between the model and the data.
    """
    residuals = signal - x[1] * (
        1
        - 2 * x[2] * np.exp(-np.array(parameters.inversion_times) / x[0])
        + np.exp(-np.array(parameters.repetition_times) / x[0])
    )
    if not isinstance(residuals, np.ndarray):
        raise ValueError(f"Residuals must be an ndarray, is {type(residuals)}")
    return residuals


def t1_mapping(
    signal: NDArray,
    parameters: T1MappingParameters,
    mask: Optional[NDArray] = None,
) -> T1MappingResults:
    """The method is known as inversion recovery T1 mapping (IR), and it consists of
    inverting the longitudinal magnetization Mz and sampling the MR signal as it
    recovers with an exponential recovery time T1.
    With all models, the fit is performed using a Levenberg-Marquardt (LM) algorithm.

    :param signal: The signal intensity. Must be an ND NumPy array, where the last
        dimension corresponds to the data from the individual inversion time (in
        ascending order).

    :param parameters: The parameters for the T1 mapping.
        See :class:`T1MappingParameters`.
    :param mask: An optional mask. Must be a NumPy array with the same dimensions as
        the input signal. If provided, the T1 mapping is only performed on the voxels
        where the mask is True.

    :return: The results of the T1 mapping. The dimensions of the arrays are the same
        as the input signal. See :class:`T1MappingResults`.
    """

    # Check the parameters
    assert isinstance(signal, np.ndarray), "Signal must be a NumPy array"
    assert isinstance(
        parameters, T1MappingParameters
    ), "Parameters must be a T1MappingParameters instance"

    if signal.shape[-1] != len(parameters.inversion_times):
        raise ValueError(
            f"Signal has {signal.shape[-1]} inversion times, but"
            f" {len(parameters.inversion_times)} are required"
        )

    n_parameters: Final[int] = 3

    # Reshape the signal to a 2D array, where the first dimension corresponds to the
    # voxels and the last dimension corresponds to the data from the individual
    # inversion times (in ascending order).
    flattened_signal = signal.reshape(-1, signal.shape[-1])

    flattened_mask = np.ones(signal.shape[:-1], dtype=bool).reshape(-1)
    if mask is not None:
        flattened_mask = mask.reshape(-1)

    if flattened_signal.shape[:-1] != flattened_mask.shape:
        raise ValueError(
            "The mask must have the same dimensions as the signal, except for the last "
            "dimension"
        )

    # Create the results array. The first dimension is the number of voxels, the second
    # is the number of parameters.
    flattened_result = np.zeros(
        flattened_signal.shape[:-1] + (n_parameters,), dtype=np.float64
    )

    # Loop over the voxels
    logger.info("Fitting %s T1 values", flattened_signal.shape[0])

    # Number of voxels to process
    n_voxels = np.sum(flattened_mask)
    voxel_count = 0

    for voxel_idx in range(flattened_signal.shape[0]):
        if not flattened_mask[voxel_idx]:
            continue
        voxel_count += 1
        print(f"Fitting T1 to voxel {voxel_count} of {n_voxels}", end="\r")
        # Iterate over the inversion times

    # Remove T1 values that are NaN, Inf, negative or greater than 20
    flattened_result[..., 0][flattened_result[..., 0] < 0] = 0
    flattened_result[..., 0][np.isnan(flattened_result[..., 0])] = 0
    flattened_result[..., 0][np.isinf(flattened_result[..., 0])] = 0
    flattened_result[..., 0][flattened_result[..., 0] > 20] = 0

    # Reshape the results to the original shape
    result = flattened_result.reshape(signal.shape[:-1] + (n_parameters,))

    if mask is not None:
        print(f"ROI mean {np.sum(result[...,0]/np.sum(mask))}")

    return T1MappingResults(
        t1=result[..., 0],
        s0=result[..., 1],
        inv_eff=result[..., 2],
    )
