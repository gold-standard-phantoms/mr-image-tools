"""Ground truth parser"""
from __future__ import annotations

from copy import deepcopy
from typing import Final, Generic, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from mrimagetools.v2.containers.image import BaseImageContainer, NiftiImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.basefilter import BaseFilter
from mrimagetools.v2.validators.fields import NiftiDataTypeField, UnitField
from mrimagetools.v2.validators.parameter_model import (
    GenericParameterModel,
    ParameterModel,
)


class Quantity(ParameterModel):
    """A representation of a ground truth quantity"""

    name: str  # The quantity name e.g. 'm0'
    # The units e.g. 'mm^2*s**-1' or 'kilometers/N'.
    # Uses pythonic maths expressions
    units: Optional[UnitField] = None
    cast_to: Optional[NiftiDataTypeField] = None


class CalculatedQuantity(Quantity):
    """A representation of a calculated ground truth quantity"""

    expression: str  # The expression used to calculate the quantity.


class GroundTruthConfig(GenericParameterModel):
    """The main configuration of the ground truth image data containing.
    Information about the ground truth quantities and any quantities
    that need to be calculated"""

    # Quantities that must correspond with the size of the 5th input image dimension
    quantities: list[Quantity]
    # Any calculated quantities - defaults to an empty list
    calculated_quantities: list[CalculatedQuantity] = Field(default_factory=list)
    # Any parameters
    parameters: dict = Field(default_factory=dict)
    # Any segmentation_labels - defaults to an empty dict
    segmentation_labels: dict[str, dict[str, int]] = Field(default_factory=dict)

    @field_validator("quantities", "calculated_quantities")
    @classmethod
    def quantity_names_are_unique(cls, value: list[Quantity]) -> list[Quantity]:
        """To check the quantity names are unique"""

        names = [quantity.name for quantity in value]
        # to check all unique list elements
        if not len(set(names)) == len(names):
            raise ValueError(f"Quantity names must be unique but {names} was supplied")
        return value

    @field_validator("segmentation_labels")
    @classmethod
    def segmentation_label_checks(
        cls, value: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, int]]:
        """checks that segmentaion_labels has no negative values and no duplicate"""
        for _, dictionary in value.items():
            # checks no negative values
            for _, item in dictionary.items():
                if item < 0:
                    raise ValueError("Segmentation label must be >= 0")
            # checks no duplicate values
            check_duplicate = list(dictionary.values())
            any(check_duplicate.count(x) > 1 for x in check_duplicate)
            if any(check_duplicate.count(x) > 1 for x in check_duplicate):
                raise ValueError("Duplicate label values detected")
        return value


_BaseModelT = TypeVar("_BaseModelT", bound=BaseModel)


class GroundTruthInput(ParameterModel, Generic[_BaseModelT]):
    """All of the inputs for the GroundTruthParser"""

    image: NiftiImageContainer
    config: GroundTruthConfig
    # parameter_validator
    parameter_validator_type: Optional[type[_BaseModelT]] = None

    # Any required quantities. These should be supplied in the `config.quantities` or
    # `self.calculated_quantities` field
    required_quantities: Optional[tuple[str, ...]] = None

    @model_validator(mode="after")
    def check_parameters_match_model(self, info: ValidationInfo) -> GroundTruthInput:
        """Check that the parameters are validated by the parameter_validator"""
        parameter_validator_type = self.parameter_validator_type
        if parameter_validator_type is not None:
            parameters: dict = self.config.parameters
            parameter_validator_type.model_validate(parameters)
        return self

    @field_validator("image")
    def image_must_be_5d(cls, value: NiftiImageContainer) -> NiftiImageContainer:
        """The input image must be 5D"""
        if len(value.image.shape) != 5:
            raise ValueError(f"Image must be 5D but has shape {value.image.shape}")
        return value

    @model_validator(mode="after")
    def check_num_image_match_quantities(
        self, info: ValidationInfo
    ) -> GroundTruthInput:
        """The size of the fifth dimension of the images should
        match the number of defined quantities"""
        quantity_names = [quantity.name for quantity in self.config.quantities]
        image_shape = self.image.image.shape
        if image_shape[4] != len(quantity_names):
            raise ValueError(
                f"The number of images in the ground truth ({image_shape[4]}) "
                f"does not match the number of defined quantities: {quantity_names}"
            )

        return self

    @model_validator(mode="after")
    def check_duplicate_quantity_names(self, info: ValidationInfo) -> GroundTruthInput:
        """Checks for duplicates between quantities and calculated_quantities"""
        quantity_names = [quantity.name for quantity in self.config.quantities] + [
            quantity.name for quantity in self.config.calculated_quantities
        ]
        # to check all unique list elements
        if not len(set(quantity_names)) == len(quantity_names):
            raise ValueError(
                "Quantity and calculatd quantity names must be unique "
                f"but {quantity_names} was supplied"
            )
        return self

    @model_validator(mode="after")
    def corresponding_entries_segmentation_labels(
        self, info: ValidationInfo
    ) -> GroundTruthInput:
        """check that that each entry corresponds to a quantity in quantities or
        calculated_quantities"""
        quantity_names = [quantity.name for quantity in self.config.quantities] + [
            quantity.name for quantity in self.config.calculated_quantities
        ]
        for key, _ in self.config.segmentation_labels.items():
            if key not in quantity_names:
                raise ValueError(
                    f"no corresponding entry for {key} in quantities or"
                    f" calculated_quantities ({quantity_names})"
                )
        return self

    @model_validator(mode="after")
    def check_all_required_quatities_exist(
        self, info: ValidationInfo
    ) -> GroundTruthInput:
        """Check that all `required_quantities` exist in either `config.quantities`
        or `config.calculated_quantities`"""
        quantity_names = [quantity.name for quantity in self.config.quantities] + [
            quantity.name for quantity in self.config.calculated_quantities
        ]
        required_quantities: Optional[tuple[str, ...]] = self.required_quantities
        if required_quantities is None:
            return self
        for quantity in required_quantities:
            if quantity not in quantity_names:
                raise ValueError(
                    f"Required quantity '{quantity}' has not been supplied in "
                    f"{required_quantities}"
                )
        return self


class GroundTruthOutput(ParameterModel):
    """All of the outputs for the GroundTruthParser"""

    images: dict[str, BaseImageContainer]
    config: GroundTruthConfig


class GroundTruthParser(BaseFilter):
    r"""A filter that parses a 5D ImageContainer and an associated JSON dictionary
    with information about the ground truth, and produces a set of output images.

    It is possible to enforce that certain quantities are loaded by this filter (a
    FilterInputValidationError will be raised if not), by using the following
    instantiation syntax:
    `GroundTruthParser(required_quantities=['t1', 't2', 'adc'])`

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

    def __init__(
        self,
        required_quantities: Optional[tuple[str, ...]] = None,
        parameter_model: Optional[type[BaseModel]] = None,
    ) -> None:
        super().__init__(name="Ground Truth Parser")
        self.required_quantities: Final[Optional[tuple[str, ...]]] = required_quantities
        self.parsed_inputs: GroundTruthInput
        self.parsed_outputs: GroundTruthOutput
        self.parameter_model: Optional[type[BaseModel]] = parameter_model

    def _get_parsed_inputs(self) -> GroundTruthInput:
        return GroundTruthInput(
            **self.inputs,
            required_quantities=self.required_quantities,
            parameter_validator_type=self.parameter_model,
        )

    def _run(self) -> None:
        self.parsed_inputs = self._get_parsed_inputs()
        # Create a new header for each of the new nifti image containers
        # and initialise the values that are shared between all of them
        shared_header = deepcopy(self.parsed_inputs.image.header)
        shared_header["dim"][0] = 4  #  Remove the 5th dimension
        if shared_header["dim"][4] == 1:
            # If we only have 1 time-step, reduce to 3D
            shared_header["dim"][0] = 3
        shared_header["dim"][5] = 1  # tidy the 5th dimensions size

        # For each new image, based on the "quantities"
        output_images: dict[str, BaseImageContainer] = {}
        for quantity_index, quantity in enumerate(self.parsed_inputs.config.quantities):
            # HEADER
            # Create a copy of the shared header - we might want to modify this further
            header = deepcopy(shared_header)
            # Grab the relevant image data
            image_data: np.ndarray = self.parsed_inputs.image.image[
                :, :, :, :, quantity_index
            ]
            if header["dim"][0] == 3:
                # squeeze the 4th dimension if there is only one time-step
                image_data = np.squeeze(image_data, axis=3)

            # METADATA
            # If we have a segmentation label, round and squash the
            # data to uint16 and update the NIFTI header
            metadata: ImageMetadata = ImageMetadata()
            if quantity.cast_to is not None:
                header["datatype"] = quantity.cast_to.type_code

                image_data = np.around(image_data).astype(quantity.cast_to.value)
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

            output_images[quantity.name] = new_image_container
            self.outputs[quantity.name] = new_image_container
        self.outputs["parameters"] = self.parsed_inputs.config.parameters
        # For legacy reasons, these parameters need to be in the base of the outputs
        self.outputs = {**self.outputs, **self.parsed_inputs.config.parameters}
        self.outputs["segmentation_labels"] = (
            self.parsed_inputs.config.segmentation_labels
        )
        self.parsed_outputs = GroundTruthOutput(
            images=output_images, config=self.parsed_inputs.config
        )

    def _validate_inputs(self) -> None:
        self._get_parsed_inputs()
