"""Split image filter"""

from mrimagetools.containers.image import BaseImageContainer
from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    isinstance_validator,
    non_empty_list_or_tuple_validator,
    range_exclusive_validator,
    range_inclusive_validator,
)


class SplitImageFilter(BaseFilter):
    r"""Splits an input image at specified indices along a specified axis into
    multiple images.

    N + 1 images are produced, where N is the number of indices supplied, i.e.

    .. math::

        \begin{align}
        &\text{indexes} = [i_n, i_{n+1}, ..., i_{N-1}, i_{N}]\\
        &\text{slice}_0 = 0, 1, 2, ..., i_0-1\\
        &\text{slice}_1 = i_0, i_0 + 1, i_0 + 2, ..., i_1-1\\
        &...\\
        &\text{slice}_n = i_{n-1}, i_{n-1} + 1, ..., i_{n}-1\\
        &...\\
        &\text{slice}_N = i_{N-1}, i_{N-1} + 1, ..., i_{N}-1\\
        &\text{slice}_{N+1} = i_N, i_N+1, ..., M-2, M-1\\
        &\text{where}\\
        &M=\text{length of the image along the specified axis}
        \end{align}

    **Inputs**

    Input parameters are all keyword arguments for the
    :class:`SplitImageFilter.add_inputs()` method. They are also accessible via
    class constants, for example :class:`SplitImageFilter.KEY_IMAGE`

    :param 'image': The image to split.
    :type 'image': BaseImageContainer
    :param 'axis': The axis to split along. Must not be greater than the
      number of axes in ``'image'``.
    :type 'axis': int
    :param 'indices': The indices to split the image at. If not in ascending order
      they will be sorted. For each supplied index
      a slice will be taken between the preceeding index up to, but not including
      the current index (as per standard numpy slicing). For the first index the
      slice will start at 0, and for the final index the slice will continue to
      the end of the image along the chosen ``'axis'``. Each index must be
      exclusively between 0 and the number of elements along the chosen ``'axis'``.
    :type 'indices': List[int]

    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`SplitImageFilter.outputs` with N + 1 images, where N is the number
    of indices provided. These images will be given the name:

    :param 'image_n': Image constructed from a slice between the n-1 and nth
      indices, along the specified axis.
    :type 'image_n': BaseImageContainer


    """

    KEY_IMAGE = "image"
    KEY_AXIS = "axis"
    KEY_INDICES = "indices"

    def __init__(self) -> None:
        super().__init__(name="Split Image Filter")

    def _run(self) -> None:
        """Splits the input image based on the supplied indices and dimension"""
        image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        axis: int = self.inputs[self.KEY_AXIS]

        image_shape = image.shape

        # add the size of the axis as the last index
        indices = self.inputs[self.KEY_INDICES] + [image_shape[axis]]
        indices.sort()

        for n, _ in enumerate(indices):
            new_image = image.clone()

            slc = [slice(None)] * new_image.image.ndim
            if n == 0:
                slc[axis] = slice(0, indices[n])
            else:
                slc[axis] = slice(indices[n - 1], indices[n])

            new_image.image = new_image.image[tuple(slc)]
            self.outputs[f"image_{n}"] = new_image

    def _validate_inputs(self) -> None:
        """Checks the inputs meet their validation criteria
        'image' must be derived from BaseImageContainer
        'axis' must be an integer and <= the number of dimensions in the image
        'indices'  must be an integer or sequence of integers, each element
          must be exclusively between 0 and the size of the image in along
          the dimension specified by 'dim'
        """
        # first validate the input image
        input_image_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
            }
        )

        input_image_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )
        image: BaseImageContainer = self.inputs[self.KEY_IMAGE]

        # then validate the input 'axis', which requires knowing how many dimensions
        # the image has
        input_dim_validator = ParameterValidator(
            parameters={
                self.KEY_AXIS: Parameter(
                    validators=[
                        isinstance_validator(int),
                        range_inclusive_validator(0, image.image.ndim),
                    ]
                )
            }
        )

        input_dim_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # finally validate the input 'indices', the values of each cannot exceed
        # the shape in the dimension specified by 'dim'
        axis = self.inputs[self.KEY_AXIS]

        input_indices_validator = ParameterValidator(
            parameters={
                self.KEY_INDICES: Parameter(
                    validators=[
                        non_empty_list_or_tuple_validator(),
                        for_each_validator(
                            isinstance_validator(int),
                        ),
                        for_each_validator(
                            range_exclusive_validator(0, image.shape[axis])
                        ),
                    ]
                )
            }
        )
        input_indices_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )
