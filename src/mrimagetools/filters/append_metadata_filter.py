""" Append Metadata Filter """
from typing import Union

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata


def append_metadata_filter_function(
    image: BaseImageContainer, metadata: Union[ImageMetadata, dict]
) -> BaseImageContainer:
    """appends the input image with the supplied metadata

    :param image: The input image to append the metadata to
    :param metadata: dictionary of key-value pars to append to the metadata property
        of the input image.

    :return: the input image, with the input metadata merged.
    """

    appended_metadata = ImageMetadata(
        **{
            **image.metadata.model_dump(exclude_none=True),
            **(
                metadata
                if isinstance(metadata, dict)
                else metadata.model_dump(exclude_none=True)
            ),
        }
    )

    image.metadata = appended_metadata
    return image
