"""A module for T1 mapping."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from pydantic import Field, root_validator, validator
from scipy.optimize import OptimizeResult, least_squares

from mrimagetools.containers.image import NiftiImageContainer
from mrimagetools.utils.io import nifti_reader
from mrimagetools.validators.parameter_model import ParameterModel

logger = logging.getLogger(__name__)


class T1Model(str, Enum):
    """Model to use for T1 mapping"""

    GENERAL = "general"
    """:math:`S = S_0 (1 - 2 inv_eff exp(-TI/T1) + exp(-TR/T1))`"""

    CLASSICAL = "classical"
    """:math:`S = S_0 (1 - 2 inv_eff exp(-TI/T1))`"""

    VTR = "vtr"
    """:math:`S = M (1 - exp(-TR/T1))`"""


class InversionRecoveryParameters(ParameterModel):
    """The parameters for T1 mapping."""

    model: T1Model = Field(
        T1Model.GENERAL,
        description=(
            "The model to use for the T1 mapping. The classical model is :math:`S ="
            " S_0 (1 - 2 inv_eff exp(-TI/T1))`. The general model is :math:`S = S_0"
            " (1 - 2 inv_eff exp(-TI/T1) + exp(-TR/T1))`. The VTR model is :math:`S"
            " = M (1 - exp(-TR/T1))`."
        ),
    )

    @property
    def n_model_parameters(self) -> int:
        """Returns the number of model parameters used in the fitting"""
        return 3

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


class VtrParameters(ParameterModel):
    """The parameters for VTR T1 mapping."""

    repetition_times: Union[float, list[float]] = Field(..., gt=0.0)
    """The repetition time(s) (TR) in seconds. May be a scalar or a list of floats.
    In the first instance, the same TR is used for all inversion times.
    In the second, the TRs are used in the same order as the inversion times."""

    @validator("repetition_times")
    def check_repetition_times_order(cls, v: list[float]) -> list[float]:
        """Check that the TRs are in ascending order."""
        if v != sorted(v):
            raise ValueError("The repetition times must be in ascending order.")
        return v

    @property
    def n_model_parameters(self) -> int:
        """Returns the number of model parameters used in the fitting"""
        return 2


@dataclass
class T1MappingResults:
    """The results of the T1 mapping."""

    t1: NDArray[np.floating]
    """An ND NumPy array with the estimated T1 values in seconds."""


@dataclass
class InversionRecoveryResults(T1MappingResults):
    """The results of the inversion recovery T1 mapping."""

    s0: NDArray[np.floating]
    """An ND NumPy array with the estimated S0 values."""

    inv_eff: NDArray[np.floating]
    """An ND NumPy array with the estimated inversion efficiency values."""


@dataclass
class VtrResults(T1MappingResults):
    """The results of the variable TR T1 mapping."""

    m: Optional[NDArray[np.floating]] = None
    """An ND NumPy array with the estimated M values. """


def _optimise_classical(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    parameters: InversionRecoveryParameters,
) -> NDArray:
    """The optimisation routine for T1 mapping. Uses the classical model:
        :math:`S = S_0 (1 - 2 inv_eff exp(-TI/T1))`

    :param x: The parameters to be optimised. This is a 1D NumPy array with the
        parameters in the following order: T1, S0, inv_eff.

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the inversion times.

    :param parameters: The parameters for the T1 mapping.
        See :class:`T1MappingParameters`.

    :return: The residual between the model and the data.
    """
    residuals = signal - x[1] * (
        1 - 2 * x[2] * np.exp(-np.array(parameters.inversion_times) / x[0])
    )
    if not isinstance(residuals, np.ndarray):
        raise ValueError(f"Residuals must be an ndarray, is {type(residuals)}")
    return residuals


def _optimise_general(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    parameters: InversionRecoveryParameters,
) -> NDArray:
    """The optimisation routine for T1 mapping. Uses the general model:
        :math:`S = S_0 (1 - 2 inv_eff exp(-TI/T1) + exp(-TR/T1))`

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


def _optimise_vtr(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    parameters: VtrParameters,
) -> NDArray:
    """The optimisation routine for VTR T1 mapping. Uses the variable TR model:
        :math:`S = M (1 - exp(-TR/T1))`

    :param x: The parameters to be optimised. This is a 1D NumPy array with the
        parameters in the following order: T1, M.

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the inversion times.

    :param parameters: The parameters for the T1 mapping.
        See :class:`VtrParameters`.

    :return: The residual between the model and the data.
    """
    residuals = signal - x[1] * (
        1 - np.exp(-np.array(parameters.repetition_times) / x[0])
    )
    if not isinstance(residuals, np.ndarray):
        raise ValueError(f"Residuals must be an ndarray, is {type(residuals)}")
    return residuals


def fit_voxel_inversion_recovery(
    signal: NDArray, parameters: InversionRecoveryParameters
) -> Optional[NDArray]:
    """Fit the inversion recovery model to a voxel.

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the inversion times.
    :param parameters: The parameters for the inversion recovery T1 mapping.
        See :class:`InversionRecoveryParameters`.

    :return: An 1D numpy array with the values of t1, s0 and inv_eff.
    """
    # The best fit is the one with the lowest residual
    best_fit = np.inf
    result: Optional[NDArray] = None
    for ti_idx in range(len(parameters.inversion_times) + 1):
        # Perform the polarity restoration. For all inversion times, before the
        # value - invert the polarity
        voxel_values = np.copy(signal)
        voxel_values[:ti_idx] = -voxel_values[:ti_idx]
        # Perform the optimisation
        lsq_result: OptimizeResult = least_squares(
            fun=(
                _optimise_general
                if parameters.model == T1Model.GENERAL
                else _optimise_classical
            ),
            method="lm",  # Levenberg-Marquardt
            # Initial guess for the parameters:
            # the T1 is 1 second, the S0 is the last signal intensity, the inversion
            # efficiency is 1
            max_nfev=100000,
            x0=[1.0, voxel_values[-1], 1.0],
            kwargs={
                "signal": voxel_values,
                "parameters": parameters,
            },
        )
        if not isinstance(lsq_result, OptimizeResult):
            return None
        if not lsq_result["success"]:
            return None
        sum_of_residuals = np.sum(abs(lsq_result["fun"]))

        if sum_of_residuals < best_fit:
            best_fit = sum_of_residuals
            # The solution found
            result = lsq_result["x"]

    return result


def fit_voxel_variable_tr(
    signal: NDArray, parameters: VtrParameters
) -> Optional[NDArray]:
    """Fit the Variable TR model to a voxel.

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the different TRs.
    :param parameters: The parameters for the VTR T1 mapping.
        See :class:`VtrParameters`.

    :return: An 1D numpy array with the values of T1, and M.
    """
    # Perform the optimisation
    lsq_result: OptimizeResult = least_squares(
        fun=_optimise_vtr,
        method="lm",  # Levenberg-Marquardt
        # Initial guess for the parameters:
        # the T1 is 1 second, M is the first signal intensity
        max_nfev=100000,
        x0=[1.0, signal[0]],
        kwargs={
            "signal": signal,
            "parameters": parameters,
        },
    )
    if not isinstance(lsq_result, OptimizeResult):
        return None
    if not lsq_result["success"]:
        return None

    return lsq_result["x"]


def t1_mapping(
    signal: NDArray,
    parameters: Union[InversionRecoveryParameters, VtrParameters],
    mask: Optional[NDArray] = None,
) -> Union[InversionRecoveryResults, VtrResults]:
    """Performs either an inversion recovery or variable TR T1 mapping.
    With all models, the fit is performed using a Levenberg-Marquardt (LM) algorithm.

    :param signal: The signal intensity. Must be an ND NumPy array, where the last
        dimension corresponds to the data from the individual inversion time (in
        ascending order).
    :param parameters: The parameters for the T1 mapping. Either an
        :class:`InversionRecoveryParameters` or a :class:`VtrParameters` instance.
    :param mask: An optional mask. Must be a NumPy array with the same dimensions as
        the input signal. If provided, the T1 mapping is only performed on the voxels
        where the mask is True.

    :return: The results of the T1 mapping. The dimensions of the arrays are the same
        as the input signal. The results are either an :class:`InversionRecoveryResults`
        or a :class:`VtrResults` instance.
    """

    # Check the parameters
    assert isinstance(signal, np.ndarray), "Signal must be a NumPy array"
    assert isinstance(
        parameters, (InversionRecoveryParameters, VtrParameters)
    ), "Parameters must be a InversionRecoveryParameters or VtrParameters instance"

    if isinstance(parameters, InversionRecoveryParameters):
        if signal.shape[-1] != len(parameters.inversion_times):
            raise ValueError(
                f"Signal has {signal.shape[-1]} inversion times, but"
                f" {len(parameters.inversion_times)} are required"
            )

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
        flattened_signal.shape[:-1] + (parameters.n_model_parameters,), dtype=np.float64
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

        lsq_result: Optional[NDArray] = None
        if isinstance(parameters, InversionRecoveryParameters):
            lsq_result = fit_voxel_inversion_recovery(
                flattened_signal[voxel_idx, :], parameters
            )

        elif isinstance(parameters, VtrParameters):
            lsq_result = fit_voxel_variable_tr(
                flattened_signal[voxel_idx, :], parameters
            )
        else:
            raise TypeError(f"Unknown parameters type: {type(parameters)}")

        # This is currently just allowing the optimisation to fail, in which case,
        # all values are set to zero
        if lsq_result is not None:
            flattened_result[voxel_idx, :] = lsq_result

    # Remove T1 values that are NaN, Inf, negative or greater than 20
    flattened_result[..., 0][flattened_result[..., 0] < 0] = 0
    flattened_result[..., 0][np.isnan(flattened_result[..., 0])] = 0
    flattened_result[..., 0][np.isinf(flattened_result[..., 0])] = 0
    flattened_result[..., 0][flattened_result[..., 0] > 20] = 0

    # Reshape the results to the original shape
    result = flattened_result.reshape(
        signal.shape[:-1] + (parameters.n_model_parameters,)
    )

    if mask is not None:
        print(f"ROI mean {np.sum(result[...,0]/np.sum(mask))}")

    if isinstance(parameters, InversionRecoveryParameters):
        return InversionRecoveryResults(
            t1=result[..., 0],
            s0=result[..., 1],
            inv_eff=result[..., 2],
        )
    if isinstance(parameters, VtrParameters):
        return VtrResults(
            t1=result[..., 0],
            m=result[..., 1],
        )
    raise TypeError(f"Unknown parameters type: {type(parameters)}")


@dataclass
class NiftiJsonPair:
    """A NIfTI and JSON pair."""

    nifti: NiftiImageContainer
    json: dict


def t1_mapping_from_files(
    filepaths: list[Path],
    output_t1_file: Path,
    output_s0_file: Optional[Path] = None,
    output_inv_eff_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    model: T1Model = T1Model.GENERAL,
) -> None:
    """T1 mapping from files.

    :param filepaths: The filepaths to the NIfTI files. Assumed that there are
        associated JSON files with the same names and a .json extension
    :param output_t1_file: The file to save the results of the T1 mapping.
        The dimensions of the arrays are the same as the input signal.
    :param output_s0_file: The file to save the results of the S0 mapping. Is only
        relevant for inversion recovery fitting.
        The dimensions of the arrays are the same as the input signal. Optional.
    :param output_inv_eff_file: The file to save the results of the inversion
        efficiency mapping. Is only relevant for inversion recovery fitting.
        The dimensions of the arrays are the same as the input signal. Optional.
    :param mask_file: The mask file. Optional.
    :param model: The model to use. If not provided, the model is inferred from
        the defaults of :class:`T1MappingParameters`.
    """
    # Generate the JSON filenames
    file_pairs: list[tuple[Path, Path]] = []

    for filepath in filepaths:
        s = str(filepath)
        if not s.endswith(".nii.gz") and not s.endswith(".nii"):
            continue
        jsonpath = s.replace(".nii.gz", ".json").replace(".nii", ".json")
        file_pairs.append((filepath, Path(jsonpath)))

    # Read the NIfTI files
    image_containers = [
        NiftiJsonPair(
            NiftiImageContainer(nifti_img=nifti_reader(filepath)),
            # Read the JSON file
            json.loads(metapath.read_text()),
        )
        for filepath, metapath in file_pairs
    ]

    # Load the mask (if provided)
    mask: Optional[np.ndarray] = None
    if mask_file is not None:
        mask = nifti_reader(mask_file).get_fdata() > 0

    # Check the images have a repetition time (IR and VTR should)
    if any(
        image_container.json["RepetitionTime"] is None
        for image_container in image_containers
    ):
        raise ValueError("All images must have a repetition time")

    # Checks to perform if the model is IR
    if model == T1Model.GENERAL:
        # Checks the images are all IR
        if any(
            image_container.json["ScanningSequence"] != "IR"
            for image_container in image_containers
        ):
            raise ValueError("All images must be inversion recovery")

        # Check the images have an inversion time
        if any(
            image_container.json["InversionTime"] is None
            for image_container in image_containers
        ):
            raise ValueError("All images must have an inversion time")

        # Sort by inversion time (relevant for inversion recovery fitting)
        image_containers.sort(key=lambda x: x.json["InversionTime"])  # type: ignore

        logger.info(
            (
                "Performing T1 mapping on %s images, with inversion times: %s, and"
                " repetition times: %s"
            ),
            len(image_containers),
            [
                image_container.json["InversionTime"]
                for image_container in image_containers
            ],
            [
                image_container.json["RepetitionTime"]
                for image_container in image_containers
            ],
        )

    # Checks to perform if the model is VTR for bruker data
    if model == T1Model.VTR:
        if any(
            image_container.json["ScanningSequence"] != "Bruker:RAREVTR"
            for image_container in image_containers
        ):
            raise ValueError("All images must be variable TR.")

        # Sort by repetition time (relevant for VTR fitting)
        image_containers.sort(key=lambda x: x.json["RepetitionTime"])  # type: ignore

    # Check that images are the same shape
    if any(
        image_container.nifti.nifti_image.shape
        != image_containers[0].nifti.nifti_image.shape
        for image_container in image_containers
    ):
        raise ValueError("All images must be the same shape")

    # Check that images are 4D or smaller in dimension
    if any(
        image_container.nifti.nifti_image.ndim > 4
        for image_container in image_containers
    ):
        raise ValueError("Input data must be 4D or less")

    # Concatenate the images into a N+1D array
    signal = np.stack(
        [
            image_container.nifti.nifti_image.get_fdata()
            for image_container in image_containers
        ],
        axis=-1,
    )

    parameters: Union[InversionRecoveryParameters, VtrParameters]
    if model in (T1Model.GENERAL, T1Model.CLASSICAL):
        # Create the parameters
        parameters = InversionRecoveryParameters(
            model=model,
            inversion_times=[
                image_container.json["InversionTime"]
                for image_container in image_containers
            ],
            repetition_times=[
                image_container.json["RepetitionTime"]
                for image_container in image_containers
            ],
        )
    elif model == T1Model.VTR:
        # Create the parameters
        parameters = VtrParameters(
            repetition_times=[
                image_container.json["RepetitionTime"]
                for image_container in image_containers
            ],
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Perform the T1 mapping
    t1_mapping_results = t1_mapping(signal=signal, parameters=parameters, mask=mask)

    # Save the results
    logger.info("Saving T1 map to %s", output_t1_file)
    nib.nifti1.save(
        Nifti1Image(
            t1_mapping_results.t1,
            image_containers[0].nifti.nifti_image.affine,
        ),
        output_t1_file,
    )

    if isinstance(t1_mapping_results, InversionRecoveryResults):
        # If the S0 output is requested, save it
        if output_s0_file is not None:
            logger.info("Saving S0 map to %s", output_s0_file)
            nib.nifti1.save(
                Nifti1Image(
                    t1_mapping_results.s0,
                    image_containers[0].nifti.nifti_image.affine,
                ),
                output_s0_file,
            )

        # If the inversion efficiency output is requested, save it
        if output_inv_eff_file is not None:
            logger.info("Saving inversion efficiency map to %s", output_inv_eff_file)
            nib.nifti1.save(
                Nifti1Image(
                    t1_mapping_results.inv_eff,
                    image_containers[0].nifti.nifti_image.affine,
                ),
                output_inv_eff_file,
            )
