""" Ground truth loader filter """
import copy
import numpy as np

from asldro.containers.image import NiftiImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
    for_each_validator,
)


class GroundTruthLoaderFilter(BaseFilter):
    """A filter for loading ground truth NIFTI/JSON
    file pairs.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`GroundTruthLoaderFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`GroundTruthLoaderFilter.KEY_IMAGE`

    :param 'image': ground truth image, must be 5D and the 5th dimension have the same length as
        the number of quantities.
    :type 'image': NiftiImageContainer
    :param 'quantities': list of quantity names
    :type 'quantities': list[str]
    :param 'units': list of units corresponding to the quantities, must be the same length as
        quantities
    :type 'units': list[str]
    :param 'parameters': dictionary containing keys
        "t1_arterial_blood", "lambda_blood_brain"
        and "magnetic_field_strength".
    :type 'parameters': dict
    :param 'segmentation': dictionary containing key-value pairs corresponding
        to tissue type and label value in the "seg_label" volume.

    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`GroundTruthLoaderFilter.outputs`
    with output fields based on the input 'quantities'.
    Each key in 'quantities' will result in a NiftiImageContainer
    corresponding to a 3D/4D subset of the nifti input (split along the 5th
    dimension). The data types of images will be the
    same as those input EXCEPT for a quantity labelled "seg_label"
    which will be converted to a uint16 data type.  The keys-value pars in
    the input 'parameters' will also be destructured and piped through
    to the output' For example:

    :param 't1': volume of T1 relaxation times
    :type 't1': NiftiImageContainer
    :param 'seg_label': segmentation label mask corresponding to different tissue types.
    :type 'seg_label': NiftiImageContainer (uint16 data type)
    :param 'magnetic_field_strength': the magnetic field strenght in Tesla.
    :type 'magnetic_field_strength': float
    :param 't1_arterial_blood': the T1 relaxation time of arterial blood
    :type 't1_arterial_blood': float
    :param 'lambda_blood_brain': the blood-brain-partition-coefficient
    :type 'lambda_blood_brain': float


    A field metadata will be created in each image container, with the
    following fields:

    * ``magnetic_field_strength``: corresponds to the value in the
      "parameters" object.
    * ``quantity``: corresponds to the entry in the "quantities" array.
    * ``units``: corresponds with the entry in the "units" array.


    The "segmentation" object from the JSON file will also be
    piped through to the metadata entry of the "seg_label" image container.
    """

    KEY_IMAGE = "image"
    KEY_UNITS = "units"
    KEY_SEGMENTATION = "segmentation"
    KEY_PARAMETERS = "parameters"
    KEY_QUANTITIES = "quantities"
    KEY_QUANTITY = "quantity"
    KEY_MAG_STRENGTH = "magnetic_field_strength"

    def __init__(self):
        super().__init__("GroundTruthLoader")

    def _run(self):
        """Load the inputs using a NiftiLoaderFilter and JsonLoaderFilter.
        Create the image outputs and the segmentation key outputs"""
        image_container: NiftiImageContainer = self.inputs[self.KEY_IMAGE]
        for i, quantity in enumerate(self.inputs[self.KEY_QUANTITIES]):
            # Create a new NiftiContainer - easier as we can just augment
            # the header to remove the 5th dimension

            header = copy.deepcopy(image_container.header)
            header["dim"][0] = 4  #  Remove the 5th dimension
            if header["dim"][4] == 1:
                # If we only have 1 time-step, reduce to 3D
                header["dim"][0] = 3
            header["dim"][5] = 1  # tidy the 5th dimensions size

            # Grab the relevant image data
            image_data: np.ndarray = image_container.image[:, :, :, :, i]
            if header["dim"][0] == 3:
                # squeeze the 4th dimension if there is only one time-step
                image_data = np.squeeze(image_data, axis=3)

            # If we have a segmentation label, round and squash the
            # data to uint16 and update the NIFTI header
            metadata = {}
            if quantity == "seg_label":
                header["datatype"] = 512
                image_data = np.around(image_data).astype(dtype=np.uint16)
                metadata[self.KEY_SEGMENTATION] = self.inputs[self.KEY_SEGMENTATION]

            nifti_image_type = image_container.nifti_type
            metadata[self.KEY_MAG_STRENGTH] = self.inputs[self.KEY_PARAMETERS][
                self.KEY_MAG_STRENGTH
            ]
            metadata[self.KEY_QUANTITY] = quantity
            metadata[self.KEY_UNITS] = self.inputs[self.KEY_UNITS][i]

            new_image_container = NiftiImageContainer(
                nifti_img=nifti_image_type(
                    dataobj=image_data, affine=image_container.affine, header=header
                ),
                metadata=metadata,
            )
            self.outputs[quantity] = new_image_container
        # Pipe through all parameters
        self.outputs = {**self.outputs, **self.inputs[self.KEY_PARAMETERS]}

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria.
        'image': NiftiImageContainer, must be 5D, 5th dimension same length as 'quantities'
        'quantities': list[str]
        'units': list[str]
        'segmentation': dict
        'parameters': dict
        The number of 'units' and 'quantities' should be equal
        The size of the 5th dimension of the image must equal the number of 'quantities'
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(NiftiImageContainer)
                ),
                self.KEY_QUANTITIES: Parameter(
                    validators=for_each_validator(isinstance_validator(str))
                ),
                self.KEY_UNITS: Parameter(
                    validators=for_each_validator(isinstance_validator(str))
                ),
                self.KEY_SEGMENTATION: Parameter(validators=isinstance_validator(dict)),
                self.KEY_PARAMETERS: Parameter(validators=isinstance_validator(dict)),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        image_container: NiftiImageContainer = self.inputs["image"]
        if len(image_container.shape) != 5:
            raise FilterInputValidationError(
                f"{self} filter requires an input nifti which is 5D"
            )

        if image_container.shape[4] != len(self.inputs["quantities"]):
            raise FilterInputValidationError(
                f"{self} filter requires an input nifti which has the "
                "same number of images (across the 5th dimension) as the JSON filter "
                "supplies in 'quantities'"
            )

        if len(self.inputs["units"]) != len(self.inputs["quantities"]):
            raise FilterInputValidationError(
                f"{self} filter requires an input 'units' which is the same length as the input"
                "quantities"
            )
