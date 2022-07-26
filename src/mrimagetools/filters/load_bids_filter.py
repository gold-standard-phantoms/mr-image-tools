"""Load Bids Filter"""
from mrimagetools.containers.image_metadata import ImageMetadata
from mrimagetools.filters.append_metadata_filter import AppendMetadataFilter
from mrimagetools.filters.basefilter import FilterInputValidationError
from mrimagetools.filters.filter_block import FilterBlock
from mrimagetools.filters.json_loader import JsonLoaderFilter
from mrimagetools.filters.nifti_loader import NiftiLoaderFilter
from mrimagetools.utils.general import splitext
from mrimagetools.validators.parameters import (
    Parameter,
    ParameterValidator,
    isfile_validator,
)


class LoadBidsFilter(FilterBlock):
    """Loads in a single BIDS image :cite:`Gorgolewski2016` comprising of:

        * A NIFTI image
        * A JSON sidecar

    **Inputs**

    :param 'nifti_filename': Path to the nifti file, must exist
    :type 'nifti_filename': str

    There should also be a json file with the same filename as the nifti file,
    however with the extension '.json'. This is checked as part of the input
    validation, and if this file doesn't exist an error will be raised.

    **Outputs**

    :param 'image': Image comprising the loaded nifti and the loaded json
      sidecar in the :class:`NiftiImageContainer.metadata` property.
    :type 'image': NiftiImageContainer


    """

    KEY_NIFTI_FILENAME = "nifti_filename"
    KEY_IMAGE = AppendMetadataFilter.KEY_IMAGE

    def __init__(self) -> None:
        super().__init__(name="Load BIDS Filter")

    def _create_filter_block(self) -> AppendMetadataFilter:
        """Runs:
        1. NiftLoaderFilter
        2. JsonLoaderFilter
        3. AppendMetadataFilter

        Returns AppendMetadataFilter
        """
        if self.inputs == {}:
            raise FilterInputValidationError("The input 'nifti_filename' is required")

        nifti_loader_filter = NiftiLoaderFilter()
        nifti_loader_filter.add_input("filename", self.inputs[self.KEY_NIFTI_FILENAME])

        json_loader_filter = JsonLoaderFilter()
        json_loader_filter.add_input("filename", self.inputs["json_filename"])
        json_loader_filter.add_input(
            JsonLoaderFilter.KEY_ROOT_OBJECT_NAME, AppendMetadataFilter.KEY_METADATA
        )

        json_loader_filter.run()
        metadata = ImageMetadata.from_bids(json_loader_filter.outputs["metadata"])

        append_metadata_filter = AppendMetadataFilter()
        append_metadata_filter.add_parent_filter(nifti_loader_filter)
        append_metadata_filter.add_input("metadata", metadata)

        return append_metadata_filter

    def _validate_inputs(self) -> None:
        """Checks that inputs meet their validation criteria
        "nifti_filename" must be a str, ending with .nii or .nii.gz and exist
        on the file system.
        There must also exist a json sidecar with the same path and filename, but
        extension ".json".
        """

        FilterInputValidationError("No inputs supplied")

        input_validator = ParameterValidator(
            parameters={
                self.KEY_NIFTI_FILENAME: Parameter(
                    validators=[isfile_validator([".nii", ".nii.gz"], must_exist=True)]
                )
            }
        )

        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # check the JSON sidecar
        base_filename, _ = splitext(self.inputs[self.KEY_NIFTI_FILENAME])
        d = {
            "json_filename": base_filename + ".json",
        }

        json_file_validator = ParameterValidator(
            parameters={
                "json_filename": Parameter(
                    validators=[isfile_validator(".json", must_exist=True)]
                )
            }
        )

        json_file_validator.validate(d, error_type=FilterInputValidationError)

        # merge the json filename into the inputs dictionary so it available in run
        self.inputs = {**self._i, **d}
