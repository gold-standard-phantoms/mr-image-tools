"""Apparent Diffusion Coefficient Quantification filter

Functional implementation of ADC fitting using various methods including
Linear and Weighted Linear Least Squares.

Based on Veraart et al. (2013) NeuroImage 81:335-346
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from mrimagetools.v2.containers.image import BaseImageContainer


@dataclass(frozen=True)
class ADCResult:
    """Result container for ADC fitting.

    :param adc: Apparent Diffusion Coefficient in mm²/s
    :param s0: Baseline signal intensity (b=0)
    :param r_squared: Coefficient of determination (0 to 1)
    """

    adc: float
    s0: float
    r_squared: float


@dataclass(frozen=True)
class ADCVolume:
    """Container for processed DWI volume results.

    :param adc_map: ADC values in mm²/s × 10⁻³
    :param s0_map: Baseline signal intensities
    :param r2_map: R-squared quality metrics
    """

    adc_map: NDArray[np.float64]
    s0_map: NDArray[np.float64]
    r2_map: NDArray[np.float64]


@dataclass(frozen=True)
class IWLLSResult:
    """Result container for iterative WLLS fitting.

    :param result: Fitted ADC parameters
    :param iterations: Number of iterations performed
    """

    result: ADCResult
    iterations: int


FittingMethod = Literal["lls", "wlls2", "iwlls"]


def adc_quantification_simple(
    dwi: BaseImageContainer, b_values: list[float]
) -> NDArray[np.floating]:
    """Calculates the apparent diffusion coefficient based on the inputs

    :param dwi: Diffusion weighted image with different b-values and b-vectors along
     the 4th dimension. There must be at least two volumes.
    :param b_values: List of b-values, one for each dwi volume. One of these must be
     equal to 0, and the length of values should be the same as the number of dwi
     volumes.


    :return adc: An image of the calculated apparent diffusion coefficient in units of
      :math:`mm^2/s`, with a volume for each non-zero b-value supplied.
    """

    # determine which DWI volume is corresponds with b = 0
    index_b0 = b_values.index(0)

    index_b = list(range(0, len(b_values)))
    index_b.pop(index_b0)  # remove the b = 0 index
    dwi_b0 = dwi.image[:, :, :, index_b0]
    if not isinstance(dwi_b0, np.ndarray):
        raise TypeError("dwi_b0 must be a numpy array")
    if dwi_b0.dtype not in [np.float32, np.float64]:
        raise TypeError("dwi_b0 must be a numpy array of floats")

    def safelog(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """A log function with a safety net- where elements of the original matrix are
        less than or equal to zero and the log of these inputs is not defined the
        output matrix entries will be zero.
        Performs type narrowing to ensure that the output us an array of floats
        """
        temp_log = np.log(x, np.zeros_like(x), where=x > 0)
        if temp_log.dtype not in [np.float32, np.float64]:
            raise TypeError("temp_log must be a numpy array of floats")
        if not isinstance(temp_log, np.ndarray):
            raise TypeError("temp_log must be a numpy array")
        return temp_log

    adc = np.stack(
        [
            (
                -(  # There is a negative sign in the equation by design
                    safelog(
                        np.divide(
                            dwi.image[:, :, :, idx],
                            dwi_b0,
                            out=np.zeros_like(dwi_b0),
                            where=dwi_b0 > 0,
                        )
                    )
                    / b_values[idx]
                )
                if b_values[idx] != 0
                else 0
            )
            for idx in index_b
        ],
        axis=3,
    )

    return adc


def calculate_r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Calculate coefficient of determination.

    :param observed: Measured signal values
    :param predicted: Model-predicted signal values
    :return: R² value between 0 and 1

    .. math::
        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res} = \\sum_i (y_i - \\hat{y}_i)^2` and
    :math:`SS_{tot} = \\sum_i (y_i - \\bar{y})^2`
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_lls(signals: NDArray[np.float64], b_values: NDArray[np.float64]) -> ADCResult:
    """Fit ADC using unweighted Linear Least Squares.

    :param signals: Signal intensities at each b-value. Must be positive.
    :param b_values: Diffusion weighting values in s/mm²
    :return: Fitted ADC, S0, and quality metric

    Implements the standard LLS estimator [1]_:

    .. math::
        \\hat{\\boldsymbol{\\beta}}_{LLS} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}

    where :math:`\\mathbf{y} = \\ln(\\mathbf{S})` and design matrix
    :math:`\\mathbf{X} = [\\mathbf{1}, -\\mathbf{b}]`.

    .. note::
        The method assumes SNR > 2 for unbiased estimation [2]_.

    .. [1] Basser, P.J., Mattiello, J., Le Bihan, D. (1994).
           "Estimation of the effective self-diffusion tensor from the NMR spin echo."
           J Magn Reson B 103(3):247-254.
    .. [2] Salvador, R., et al. (2005).
           "Formal characterization and extension of the linearized diffusion tensor model."
           Hum Brain Mapp 24(2):144-155.
    """
    # Filter valid signals
    valid_mask = signals > 0
    valid_signals = signals[valid_mask]
    valid_b = b_values[valid_mask]

    if len(valid_signals) < 2:
        return ADCResult(adc=0.0, s0=0.0, r_squared=0.0)

    # Linearize: ln(S) = ln(S0) - b*ADC
    y = np.log(valid_signals)

    # Construct design matrix: [1, -b]
    X = np.column_stack([np.ones_like(valid_b), -valid_b])

    # Solve normal equations: beta = (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    # Extract parameters
    ln_s0 = beta[0]
    adc = beta[1]
    s0 = np.exp(ln_s0)

    # Calculate R²
    predicted = s0 * np.exp(-valid_b * adc)
    r_squared = calculate_r_squared(valid_signals, predicted)

    return ADCResult(adc=adc, s0=s0, r_squared=r_squared)


def fit_wlls2(signals: NDArray[np.float64], b_values: NDArray[np.float64]) -> ADCResult:
    """Fit ADC using Weighted Linear Least Squares (WLLS2 approach).

    :param signals: Signal intensities at each b-value. Must be positive.
    :param b_values: Diffusion weighting values in s/mm²
    :return: Fitted ADC, S0, and quality metric

    Implements the WLLS2 estimator from Veraart et al. (2013) [1]_:

    .. math::
        \\hat{\\boldsymbol{\\beta}}_{WLLS} = (\\mathbf{X}^T\\mathbf{W}\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{W}\\mathbf{y}

    where weight matrix :math:`\\mathbf{W} = \\text{diag}(\\exp(2\\mathbf{X}\\hat{\\boldsymbol{\\beta}}_{LLS}))`.

    This approach uses predicted signals from an initial LLS fit for weights,
    which provides better accuracy than WLLS1 (using noisy signals) [1]_.

    .. note::
        The method assumes SNR > 2 for validity [2]_.

    .. [1] Veraart, J., et al. (2013).
           "Weighted linear least squares estimation of diffusion MRI parameters:
           Strengths, limitations, and pitfalls."
           NeuroImage 81:335-346.
    .. [2] Salvador, R., et al. (2005).
           "Formal characterization and extension of the linearized diffusion tensor model."
           Hum Brain Mapp 24(2):144-155.
    """
    # Get initial estimate from LLS
    lls_result = fit_lls(signals, b_values)

    # Filter valid signals
    valid_mask = signals > 0
    valid_signals = signals[valid_mask]
    valid_b = b_values[valid_mask]

    if len(valid_signals) < 2:
        return ADCResult(adc=0.0, s0=0.0, r_squared=0.0)

    # Predict signals from LLS fit
    predicted_signals = lls_result.s0 * np.exp(-valid_b * lls_result.adc)

    # Construct weight matrix from predicted signals
    weights = predicted_signals**2
    W = np.diag(weights)

    # Weighted least squares solution
    y = np.log(valid_signals)
    X = np.column_stack([np.ones_like(valid_b), -valid_b])

    # Solve: beta = (X'WX)^-1 X'Wy
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y
    beta = np.linalg.solve(XtWX, XtWy)

    # Extract parameters
    ln_s0 = beta[0]
    adc = beta[1]
    s0 = np.exp(ln_s0)

    # Calculate R²
    final_predicted = s0 * np.exp(-valid_b * adc)
    r_squared = calculate_r_squared(valid_signals, final_predicted)

    return ADCResult(adc=adc, s0=s0, r_squared=r_squared)


def fit_iwlls(
    signals: NDArray[np.float64],
    b_values: NDArray[np.float64],
    max_iterations: int = 5,
    tolerance: float = 1e-6,
) -> IWLLSResult:
    """Fit ADC using Iterative Weighted Linear Least Squares.

    :param signals: Signal intensities at each b-value. Must be positive.
    :param b_values: Diffusion weighting values in s/mm²
    :param max_iterations: Maximum number of iterations, defaults to 5
    :param tolerance: Convergence tolerance for ADC change, defaults to 1e-6
    :return: Fitted parameters and number of iterations performed

    Implements the iterative WLLS from Veraart et al. (2013) [1]_:

    .. math::
        \\tilde{\\mathbf{W}}_n = \\text{diag}(\\exp(2\\mathbf{X}\\hat{\\boldsymbol{\\beta}}_{n-1}))

    The algorithm iteratively refines weights until convergence.
    Typically converges in 2-3 iterations [1]_.

    .. note::
        The method assumes SNR > 2 for validity [2]_.

    .. [1] Veraart, J., et al. (2013).
           "Weighted linear least squares estimation of diffusion MRI parameters:
           Strengths, limitations, and pitfalls."
           NeuroImage 81:335-346.
    .. [2] Salvador, R., et al. (2005).
           "Formal characterization and extension of the linearized diffusion tensor model."
           Hum Brain Mapp 24(2):144-155.
    """
    # Initial estimate from LLS
    current_result = fit_lls(signals, b_values)

    # Filter valid signals
    valid_mask = signals > 0
    valid_signals = signals[valid_mask]
    valid_b = b_values[valid_mask]

    if len(valid_signals) < 2:
        return IWLLSResult(
            result=ADCResult(adc=0.0, s0=0.0, r_squared=0.0), iterations=0
        )

    # Prepare matrices
    y = np.log(valid_signals)
    X = np.column_stack([np.ones_like(valid_b), -valid_b])

    for iteration in range(max_iterations):
        # Predict signals from current estimate
        predicted = current_result.s0 * np.exp(-valid_b * current_result.adc)

        # Update weights
        weights = predicted**2
        W = np.diag(weights)

        # Weighted least squares step
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ y
        beta = np.linalg.solve(XtWX, XtWy)

        # Extract new parameters
        new_s0 = np.exp(beta[0])
        new_adc = beta[1]

        # Check convergence
        if abs(new_adc - current_result.adc) < tolerance:
            break

        # Update current result
        current_result = ADCResult(
            adc=new_adc,
            s0=new_s0,
            r_squared=current_result.r_squared,  # Will be updated after loop
        )

    # Final R² calculation
    final_predicted = current_result.s0 * np.exp(-valid_b * current_result.adc)
    r_squared = calculate_r_squared(valid_signals, final_predicted)

    return IWLLSResult(
        result=ADCResult(
            adc=current_result.adc, s0=current_result.s0, r_squared=r_squared
        ),
        iterations=iteration + 1,
    )


def process_dwi_volume(
    dwi_4d: NDArray[np.float64],
    b_values: NDArray[np.float64],
    mask: Optional[NDArray[np.bool_]] = None,
    method: FittingMethod = "wlls2",
    max_iterations: int = 5,
    tolerance: float = 1e-6,
) -> ADCVolume:
    """Process entire DWI volume to generate ADC maps.

    :param dwi_4d: 4D DWI volume with shape (x, y, z, n_bvalues)
    :param b_values: B-values corresponding to the last dimension
    :param mask: Binary mask to limit processing, shape (x, y, z), defaults to None
    :param method: Fitting method ('lls', 'wlls2', or 'iwlls'), defaults to 'wlls2'
    :param max_iterations: Maximum iterations for IWLLS method, defaults to 5
    :param tolerance: Convergence tolerance for IWLLS method, defaults to 1e-6
    :raises ValueError: If unknown fitting method is specified
    :return: ADCVolume containing ADC map, S0 map, and R² map

    This function applies the selected fitting method voxel-wise across
    the entire DWI volume. ADC values are converted to clinical units
    (× 10⁻³ mm²/s) in the output.

    .. [1] Veraart, J., et al. (2013).
           "Weighted linear least squares estimation of diffusion MRI parameters:
           Strengths, limitations, and pitfalls."
           NeuroImage 81:335-346.
    """
    nx, ny, nz, _ = dwi_4d.shape

    # Initialize output maps
    adc_map = np.zeros((nx, ny, nz), dtype=np.float64)
    s0_map = np.zeros((nx, ny, nz), dtype=np.float64)
    r2_map = np.zeros((nx, ny, nz), dtype=np.float64)

    # Default mask if not provided
    if mask is None:
        mask = np.ones((nx, ny, nz), dtype=bool)

    # Process each voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    voxel_signals = dwi_4d[i, j, k, :]

                    # Apply selected fitting method
                    if method == "lls":
                        result = fit_lls(voxel_signals, b_values)
                    elif method == "wlls2":
                        result = fit_wlls2(voxel_signals, b_values)
                    elif method == "iwlls":
                        iwlls_result = fit_iwlls(
                            voxel_signals, b_values, max_iterations, tolerance
                        )
                        result = iwlls_result.result
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # Store results (convert ADC to × 10⁻³ mm²/s)
                    adc_map[i, j, k] = result.adc * 1000
                    s0_map[i, j, k] = result.s0
                    r2_map[i, j, k] = result.r_squared

    return ADCVolume(adc_map=adc_map, s0_map=s0_map, r2_map=r2_map)


# Backward compatibility wrapper
def adc_quantification_filter_function(
    dwi: BaseImageContainer, b_values: list[float]
) -> NDArray[np.floating]:
    """Backward compatibility wrapper for the simple ADC calculation.

    This function maintains backward compatibility with existing code
    that uses the original function name.

    :param dwi: Diffusion weighted image with different b-values and b-vectors along
     the 4th dimension. There must be at least two volumes.
    :param b_values: List of b-values, one for each dwi volume. One of these must be
     equal to 0, and the length of values should be the same as the number of dwi
     volumes.
    :return adc: An image of the calculated apparent diffusion coefficient in units of
      :math:`mm^2/s`, with a volume for each non-zero b-value supplied.
    """
    return adc_quantification_simple(dwi, b_values)
