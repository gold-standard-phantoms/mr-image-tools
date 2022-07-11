""" AppendMetadataFilter """

from typing import Dict, Union

from mrimagetools.containers.image import BaseImageContainer
from mrimagetools.containers.image_metadata import ImageMetadata
from mrimagetools.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
)


class AppendMetadataFilter(BaseFilter):
    """A filter that can add key-value pairs to the metadata dictionary property of an
    image container.  If the supplied key already exists the old value will be overwritten
    with the new value.  The input image container is modified and a reference passed
    to the output, i.e. no copy is made.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`AppendMetadataFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`AppendMetadataFilter.KEY_METADATA`

    :param 'image': The input image to append the metadata to
    :type 'image': BaseImageContainer
    :param 'metadata': dictionary of key-value pars to append to the metadata property of the
        input image.
    :type 'metadata': dict

    **Outputs**

    Once run, the filter will populate the dictionary :class:`AppendMetadataFilter.outputs` with the
    following entries

    :param 'image': The input image, with the input metadata merged.
    :type 'image: BaseImageContainer

    """

    # Key constants
    KEY_IMAGE = "image"
    KEY_METADATA = "metadata"

    def __init__(self) -> None:
        super().__init__(name="Append Meta Data")

    def _run(self) -> None:
        """appends the input image with the supplied metadata"""
        # copy the reference to the input image to outputs
        self.outputs[self.KEY_IMAGE] = self.inputs[self.KEY_IMAGE]
        input_metadata: Union[Dict, ImageMetadata] = self.inputs[self.KEY_METADATA]
        # merge the input metadata with the existing metadata
        self.outputs[self.KEY_IMAGE].metadata = ImageMetadata(
            **{
                **self.outputs[self.KEY_IMAGE].metadata.dict(exclude_none=True),
                **(
                    input_metadata
                    if isinstance(input_metadata, Dict)
                    else input_metadata.dict(exclude_none=True)
                ),
            }
        )

    def _validate_inputs(self) -> None:
        """Checks that the inputs meet their validation criteria
        'image' must be derived from BaseImageContainer
        'metadata' must be an ImageMetadata or a dictionary
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_METADATA: Parameter(
                    validators=isinstance_validator((ImageMetadata, dict))
                ),
            }
        )

        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
