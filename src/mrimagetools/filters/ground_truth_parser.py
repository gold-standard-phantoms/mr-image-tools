"""Ground truth parser"""
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from mrimagetools.containers.image import BaseImageContainer, NiftiImageContainer
from mrimagetools.containers.image_metadata import ImageMetadata
from mrimagetools.filters.basefilter import BaseFilter
from mrimagetools.validators.fields import NiftiDataTypeField, UnitField
from mrimagetools.validators.parameter_model import (
    Field,
    ParameterModel,
    root_validator,
    validator,
)


class Quantity(ParameterModel):
    """A representation of a ground truth quantity"""

    name: str  # The quantity name e.g. 'm0'
    # The units e.g. 'mm^2*s**-1' or 'kilometers/N'.
    # Uses pythonic maths expressions
    units: Optional[UnitField]
    cast_to: Optional[NiftiDataTypeField]


class CalculatedQuantity(Quantity):
    """A representation of a calculated ground truth quantity"""

    expression: str  # The expression used to calculate the quantity.


class GroundTruthConfig(ParameterModel):
    """The main configuration of the ground truth image data containing.
    Information about the ground truth quantities and any quantities
    that need to be calculated"""

    # Quantities that must correspond with the size of the 5th input image dimension
    quantities: List[Quantity]
    # Any calculated quantities - defaults to an empty list
    calculated_quantities: List[CalculatedQuantity] = Field(default_factory=list)
    # Any parameters - defaults to an empty dict
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator("quantities", "calculated_quantities")
    def quantity_names_are_unique(cls, value: List[Quantity]) -> List[Quantity]:
        """To check the quantity names are unique"""

        names = [quantity.name for quantity in value]
        # to check all unique list elements
        if not len(set(names)) == len(names):
            raise ValueError(f"Quantity names must be unique but {names} was supplied")
        return value


class GroundTruthInput(ParameterModel):
    """All of the inputs for the GroundTruthParser"""

    image: NiftiImageContainer
    config: GroundTruthConfig

    @validator("image")
    def image_must_be_5d(cls, value: NiftiImageContainer) -> NiftiImageContainer:
        """The input image must be 5D"""
        if len(value.image.shape) != 5:
            raise ValueError(f"Image must be 5D but has shape {value.image.shape}")
        return value

    @root_validator(skip_on_failure=True)
    def check_num_image_match_quantities(cls, values: dict) -> dict:
        """The size of the fifth dimension of the images should
        match the number of defined quantities"""
        quantity_names = [quantity.name for quantity in values["config"].quantities]
        image_shape = values["image"].image.shape
        if image_shape[4] != len(quantity_names):
            raise ValueError(
                f"The number of images in the ground truth ({image_shape[4]}) "
                f"does not match the number of defined quantities: {quantity_names}"
            )

        return values

    @root_validator(skip_on_failure=True)
    def check_duplicate_quantity_names(cls, values: dict) -> dict:
        """Checks for duplicates between quantities and calculated_quantities"""
        quantity_names = [quantity.name for quantity in values["config"].quantities] + [
            quantity.name for quantity in values["config"].calculated_quantities
        ]
        # to check all unique list elements
        if not len(set(quantity_names)) == len(quantity_names):
            raise ValueError(
                "Quantity and calculatd quantity names must be unique "
                f"but {quantity_names} was supplied"
            )
        return values


class GroundTruthOutput(ParameterModel):
    """All of the outputs for the GroundTruthParser"""

    images: Dict[str, BaseImageContainer]
    config: GroundTruthConfig


class GroundTruthParser(BaseFilter):
    r"""A filter that parses a 5D ImageContainer and an associated JSON dictionary
    with information about the ground truth, and produces a set of output images.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`GroundTruthParser.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`GroundTruthParser.KEY_IMAGE`

    :param image: Ground truth image, must be 5D and the 5th dimension have the same length
    as the number of quantities in the ground truth.
    :type image: BaseImageContainer
    :param config: A configuration, specifying the information contained within the ground truth,
    for example, the names of the quantities, the units, and any addition information (for example,
    does one/more of the ground truth correspond with a segmentation). Additionally, specified
    are any general parameters, for example, the magnetic field strength.
    :type config: dict. Should match the specification of :class:`GroundTruthConfig`
    """

    KEY_CONFIG = "config"
    KEY_UNITS = "units"
    KEY_QUANTITY = "quantity"
    KEY_DATA_TYPE = "data_type"

    def __init__(self) -> None:
        super().__init__(name="Ground Truth Parser")
        self.parsed_inputs: GroundTruthInput
        self.parsed_outputs: GroundTruthOutput

    def _run(self) -> None:
        self.parsed_inputs = GroundTruthInput(**self.inputs)

        # Create a new header for each of the new nifti image containers
        # and initialise the values that are shared between all of them
        shared_header = deepcopy(self.parsed_inputs.image.header)
        shared_header["dim"][0] = 4  #  Remove the 5th dimension
        if shared_header["dim"][4] == 1:
            # If we only have 1 time-step, reduce to 3D
            shared_header["dim"][0] = 3
        shared_header["dim"][5] = 1  # tidy the 5th dimensions size

        # For each new image, based on the "quantities"
        for i, quantity in enumerate(self.parsed_inputs.config.quantities):
            # HEADER
            # Create a copy of the shared header - we might want to modify this further
            header = deepcopy(shared_header)
            # Grab the relevant image data
            image_data: np.ndarray = self.parsed_inputs.image.image[:, :, :, :, i]
            if header["dim"][0] == 3:
                # squeeze the 4th dimension if there is only one time-step
                image_data = np.squeeze(image_data, axis=3)

            # METADATA
            # If we have a segmentation label, round and squash the
            # data to uint16 and update the NIFTI header
            metadata: ImageMetadata = ImageMetadata()
            if quantity.cast_to is not None:
                header["datatype"] = quantity.cast_to.type_code

                image_data = np.around(image_data).astype(quantity.cast_to)
                metadata.data_type = quantity.cast_to

            metadata.quantity = quantity.name

            if quantity.units is not None:
                metadata.units = quantity.units

            # IMAGE
            nifti_image_type = self.parsed_inputs.image.nifti_type
            new_image_container = NiftiImageContainer(
                nifti_img=nifti_image_type(
                    dataobj=image_data,
                    affine=self.parsed_inputs.image.affine,
                    header=header,
                ),
                metadata=metadata,
            )

            self.outputs[quantity.name] = new_image_container
        self.outputs["parameters"] = self.parsed_inputs.config.parameters

    def _validate_inputs(self) -> None:
        GroundTruthInput(**self.inputs)
