"""A module for Mono-Exponential Fitting in T2-Relaxometry.
Based on the method in:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145255
"""

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from pydantic import Field, PositiveFloat, field_validator
from scipy.optimize import OptimizeResult, least_squares

from mrimagetools.v2.containers.image import NiftiImageContainer
from mrimagetools.v2.utils.io import nifti_reader
from mrimagetools.v2.validators.parameter_model import ParameterModel

logger = logging.getLogger(__name__)


class T2Model(str, Enum):
    """Model to use for T2 mapping"""

    # Uses the full model (includes offset)
    FULL = "full"

    # Uses the reduced model (no offset)
    REDUCED = "reduced"


class T2MappingParameters(ParameterModel):
    """The parameters for T2 mapping."""

    model: T2Model = Field(
        T2Model.FULL,
        description=(
            "The model to use for the Mono-Exponential Fitting in T2-Relaxometry."
            " The full model is"
            " :math:`S(TE)=k.S_o.exp(-TE/T_2)+offset`. The reduced model is"
            " :math:`S(TE)=k.S_o.exp(-TE/T_2)`. In both cases, :math:`k.S_0`"
            " is combined into a single parameter A."
        ),
    )

    @property
    def n_model_parameters(self) -> int:
        """The number of parameters in the model."""
        if self.model == T2Model.FULL:
            return 3
        return 2

    echo_times: list[PositiveFloat]
    """The echo time (TE) in seconds. May be a scalar or a list of floats.
    and must in ascending order, corresponding with the order of the loaded spin
    echo images."""

    skip_echos: int = Field(
        default=1,
        ge=0,
        description=(
            "The number of echoes to skip at the start of the data. Discarding the"
            " first echo is a fast and easy method to minimize the error in T2 fitting"
            " and is the default option"
        ),
    )

    @field_validator("echo_times")
    @classmethod
    def check_echo_times_order(cls, v: list[float]) -> list[float]:
        """Check that the echo times are in ascending order."""
        if v != sorted(v):
            raise ValueError("The echo times must be in ascending order.")
        return v


@dataclass
class T2MappingResults:
    """The results of the T2 mapping."""

    t2: NDArray[np.floating]
    """An ND NumPy array with the estimated T2 values in seconds."""

    a: NDArray[np.floating]
    """An ND NumPy array with the estimated values for A (:math`k.S_0`)."""

    offset: Optional[NDArray[np.floating]]
    """An ND NumPy array with the estimated offset values."""


def _optimise_full(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    parameters: T2MappingParameters,
) -> NDArray:
    """The optimisation routine for T2 mapping. Uses the full model:
        :math:`S(TE)=k.S_o.exp(-TE/T_2)+offset`"

    :param x: The parameters to be optimised. This is a 1D NumPy array with the
        parameters in the following order: T2, A, offset (where A=k.S_0).

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the echo times.

    :param parameters: The parameters for the T2 mapping.
        See :class:`T2MappingParameters`.

    :return: The residual between the model and the data.
    """
    residuals = signal - (x[1] * np.exp(-np.array(parameters.echo_times) / x[0]) + x[2])
    if not isinstance(residuals, np.ndarray):
        raise ValueError(f"Residuals must be an ndarray, is {type(residuals)}")
    return residuals


def _optimise_reduced(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    parameters: T2MappingParameters,
) -> NDArray:
    """The optimisation routine for T2 mapping. Uses the reduced model:
        :math:`S(TE)=k.S_o.exp(-TE/T_2)+offset`"

    :param x: The parameters to be optimised. This is a 1D NumPy array with the
        parameters in the following order: T2, A, offset (where A=k.S_0).

    :param signal: The signal intensity. Must be a 1D NumPy array. The values correspond
        to the signal intensity at the echo times.

    :param parameters: The parameters for the T2 mapping.
        See :class:`T2MappingParameters`.

    :return: The residual between the model and the data.
    """
    residuals = signal - x[1] * np.exp(-np.array(parameters.echo_times) / x[0])
    if not isinstance(residuals, np.ndarray):
        raise ValueError(f"Residuals must be an ndarray, is {type(residuals)}")
    return residuals


def t2_mapping(
    signal: NDArray,
    parameters: T2MappingParameters,
    mask: Optional[NDArray] = None,
) -> T2MappingResults:
    """The method is a Mono-Exponential Fitting in T2-Relaxometry.
    Based on the method in:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145255
    With all models, the fit is performed using a Levenberg-Marquardt (LM) algorithm.

    :param signal: The signal intensity. Must be an ND NumPy array, where the last
        dimension corresponds to the data from the individual echo time (in
        ascending order).
    :param parameters: The parameters for the T2 mapping.
        See :class:`T2MappingParameters`.
    :param mask: An optional mask. Must be a NumPy array with the same dimensions as
        the input signal. If provided, the T2 mapping is only performed on the voxels
        where the mask is True.

    :return: The results of the T2 mapping. The dimensions of the arrays are the same
        as the input signal. See :class:`T2MappingResults`.
    """

    # Check the parameters
    assert isinstance(signal, np.ndarray), "Signal must be a NumPy array"
    assert isinstance(
        parameters, T2MappingParameters
    ), "Parameters must be a T2MappingParameters instance"

    if signal.shape[-1] != len(parameters.echo_times):
        raise ValueError(
            f"Signal has {signal.shape[-1]} echo times, but"
            f" {len(parameters.echo_times)} are required"
        )

    # Reshape the signal to a 2D array, where the first dimension corresponds to the
    # voxels and the last dimension corresponds to the data from the individual
    # echo times (in ascending order).
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

    # Create a copy of the parameters, where the echo times are reduced by the number
    # of skipped echoes
    parameters_to_use = deepcopy(parameters)
    parameters_to_use.echo_times = parameters_to_use.echo_times[
        parameters_to_use.skip_echos :
    ]

    # Number of voxels to process
    n_voxels = np.sum(flattened_mask)
    voxel_count = 0

    # Loop over the voxels
    logger.info("Fitting %s T2 values", flattened_signal.shape[0])
    for voxel_idx in range(flattened_signal.shape[0]):
        if not flattened_mask[voxel_idx]:
            continue

        voxel_count += 1
        print(f"Fitting T2 to voxel {voxel_count} of {n_voxels}", end="\r")
        # Iterate over the echo times

        logger.debug("Fitting voxel %s of %s", voxel_idx, flattened_signal.shape[0])
        # Perform the optimisation
        result: OptimizeResult = least_squares(
            fun=(
                _optimise_full
                if parameters.model == T2Model.FULL
                else _optimise_reduced
            ),
            method="lm",  # Levenberg-Marquardt
            # Initial guess for the parameters :
            # parameters in the following order: T2, A, offset (where A=k.S_0).
            # the T2 is 0.1 second, the first echo times signal for S0 and the offset
            # is 0
            max_nfev=1000000,
            x0=(
                [0.1, flattened_signal[voxel_idx, parameters.skip_echos], 0.0]
                if parameters.model == T2Model.FULL
                else [0.1, flattened_signal[voxel_idx, parameters.skip_echos]]
            ),
            kwargs={
                # Skip the first `skip_echos` echo times
                "signal": flattened_signal[voxel_idx, parameters.skip_echos :],
                "parameters": parameters_to_use,
            },
        )
        if not isinstance(result, OptimizeResult):
            continue
        if not result["success"]:
            continue
        # The solution found
        flattened_result[voxel_idx, :] = result["x"]

    # Remove T2 values that are NaN or Inf
    flattened_result[..., 0][np.isnan(flattened_result[..., 0])] = 0
    flattened_result[..., 0][np.isinf(flattened_result[..., 0])] = 0

    # Reshape the results to the original shape
    fitting_result = flattened_result.reshape(
        signal.shape[:-1] + (parameters.n_model_parameters,)
    )

    if mask is not None:
        logger.info("ROI mean: %f", np.sum(fitting_result[..., 0] / np.sum(mask)))

    return T2MappingResults(
        t2=fitting_result[..., 0],
        a=fitting_result[..., 1],
        offset=fitting_result[..., 2] if parameters.model == T2Model.FULL else None,
    )


@dataclass
class NiftiJsonPair:
    """A NIfTI and JSON pair."""

    nifti: NiftiImageContainer
    json: dict


def t2_mapping_from_files(
    filepaths: list[Path],
    output_t2_file: Path,
    mask_file: Optional[Path] = None,
    model: T2Model = T2Model.FULL,
    skip_echos: int = 1,
) -> None:
    """T2 mapping from files.

    :param filepaths: The filepaths to the NIfTI files. Assumed that there are
        associated JSON files with the same names and a .json extension
    :param output_t2_file: The file to save the results of the T2 mapping.
        The dimensions of the arrays are the same as the input signal.
    :param mask_file: The mask file. Optional.
    :param model: The model to use. If not provided, the model is inferred from
        the defaults of :class:`T2MappingParameters` (is the full model).
    :param skip_echos: The number of echos to skip. Defaults to 1 which helps to
        reduce T2 fitting errors.
    """
    # Generate the JSON filenames
    json_filenames: list[Path] = []

    for filepath in filepaths:
        json_filenames.append(
            Path(str(filepath).replace(".nii.gz", ".json").replace(".nii", ".json"))
        )

    # Read the NIfTI files
    image_containers = [
        NiftiJsonPair(
            NiftiImageContainer(nifti_img=nifti_reader(filepath)),
            # Read the JSON file
            json.loads(metapath.read_text()),
        )
        for filepath, metapath in zip(filepaths, json_filenames)
    ]

    # Load the mask (if provided)
    mask: Optional[np.ndarray] = None
    if mask_file is not None:
        mask = nifti_reader(mask_file).get_fdata() > 0

    # Check the images have an echo time
    if any(
        image_container.json["EchoTime"] is None for image_container in image_containers
    ):
        raise ValueError("All images must have an echo time")

    # Sort by echo time
    image_containers.sort(key=lambda x: x.json["EchoTime"])

    logger.info(
        "Performing T2 mapping on %s images, with echo times: %s",
        len(image_containers),
        [image_container.json["EchoTime"] for image_container in image_containers],
    )

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
    # Create the parameters
    parameters = T2MappingParameters(
        model=model,
        echo_times=[
            image_container.json["EchoTime"] for image_container in image_containers
        ],
        skip_echos=skip_echos,
    )

    # Perform the T2 mapping
    t2_mapping_results = t2_mapping(signal=signal, parameters=parameters, mask=mask)

    # Save the results
    logger.info("Saving T2 map to %s", output_t2_file)
    nib.nifti1.save(
        Nifti1Image(
            t2_mapping_results.t2,
            image_containers[0].nifti.nifti_image.affine,
        ),
        output_t2_file,
    )
