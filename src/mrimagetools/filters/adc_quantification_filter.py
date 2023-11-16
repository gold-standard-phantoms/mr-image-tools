"""Apparent Diffusion Coefficient Quantification filter"""

import numpy as np
from numpy.typing import NDArray

from mrimagetools.v2.containers.image import BaseImageContainer


def adc_quantification_filter_function(
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
    if not dwi_b0.dtype in [np.float32, np.float64]:
        raise TypeError("dwi_b0 must be a numpy array of floats")

    def safelog(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """A log function with a safety net- where elements of the original matrix are
        less than or equal to zero and the log of these inputs is not defined the
        output matrix entries will be zero.
        Performs type narrowing to ensure that the output us an array of floats
        """
        temp_log = np.log(x, np.zeros_like(x), where=x > 0)
        if not temp_log.dtype in [np.float32, np.float64]:
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
