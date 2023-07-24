""" Invert image filter """

from mrimagetools.v2.containers.image import BaseImageContainer
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError


class InvertImageFilter(BaseFilter):
    """A filter which simply inverts the input image.

    Must have one input named 'image'.
    These correspond with a derivative of BaseImageContainer.

    Creates a single output named 'image'.
    """

    def __init__(self) -> None:
        super().__init__("InvertImageFilter")

    def _run(self) -> None:
        """Invert the input image"""
        cloned_image_container = self.inputs["image"].clone()
        cloned_image_container.image = -cloned_image_container.image
        self.outputs["image"] = cloned_image_container

    def _validate_inputs(self) -> None:
        """There must be a input called 'image' with a BaseImageContainer."""

        if self.inputs.get("image", None) is None or not isinstance(
            self.inputs["image"], BaseImageContainer
        ):
            raise FilterInputValidationError(
                "BaseImageContainer filter requires a `image` input derived from"
                " BaseImageContainer"
            )
