"""Load ASL BIDS filter class"""

import os

from mrimagetools.filters.load_asl_bids_filter import load_asl_bids
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.utils.io import nifti_reader
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
)
from mrimagetools.v2.validators.user_parameter_input import SUPPORTED_ASL_CONTEXTS


class LoadAslBidsFilter(BaseFilter):
    """
    A filter that loads in ASL data in BIDS format, comprising of a NIFTI image file, json
    sidear and tsv aslcontext file.  After loading in the data, image containers are created using
    the volumes described in aslcontext.  For each of these containers, the data in sidecar is added
    to the metadata object.  In addition a metadata 'asl_context' is created which is a list of the
    corresponding volumes contained in each container.  Any metadata entries that are an array and
    specific to each volume have only the corresponding values copied.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`LoadAslBidsFilter.add_inputs()`
    member function. They are also accessible via class constants, for example
    :class:`LoadAslBidsFilter.KEY_SIDECAR`

    :param 'image_filename': path and filename to the ASL NIFTI image (must end in .nii or.nii.gz)
    :type 'image_filename': str
    :param 'sidecar_filename': path and filename to the json sidecar (must end in .json)
    :type 'image_filename': str
    :param 'aslcontext_filename': path and filename to the aslcontext file (must end in .tsv).  This
        must be a tab separated values file, with heading 'volume_type' and then entries which are
        either 'control', 'label', or 'm0scan'.
    :type 'aslcontext_filename': str

    **Outputs**

    Once run, the filter will populate the dictionary :class:`LoadAslBidsFilter.outputs` with the
    following entries

    :param 'source': the full ASL NIFTI image
    :type 'source': BaseImageContainer
    :param 'control': control volumes (as defined by aslcontext)
    :type 'control': BaseImageContainer
    :param 'label': label volumes (as defined by aslcontext)
    :type 'label': BaseImageContainer
    :param 'm0': m0 volumes (as defined by aslcontext)
    :type 'm0': BaseImageContainer
    :param 'sidecar': the sidecar metadata
    :type 'sidecar': dict

    """

    KEY_IMAGE_FILENAME = "image_filename"
    KEY_SIDECAR_FILENAME = "sidecar_filename"
    KEY_ASLCONTEXT_FILENAME = "aslcontext_filename"
    KEY_SOURCE = "source"
    KEY_CONTROL = "control"
    KEY_LABEL = "label"
    KEY_M0 = "m0"
    KEY_SIDECAR = "sidecar"
    ASL_CONTEXT_MAPPING = {
        KEY_CONTROL: "control",
        KEY_LABEL: "label",
        KEY_M0: "m0scan",
    }
    LIST_FIELDS_TO_EXCLUDE = [
        "ScanningSequence",
        "ComplexImageComponent",
        "ImageType",
        "AcquisitionVoxelSize",
    ]

    def __init__(self) -> None:
        super().__init__(name="Load ASL BIDS")

    def _run(self) -> None:
        """
        Loads in the NIFTI image, json sidecar and tsv aslcontext, then creates image containers
        according to the content of the aslcontext.
        """

        asl_data = load_asl_bids(
            image_filename=self.inputs[self.KEY_IMAGE_FILENAME],
            sidecar_filename=self.inputs[self.KEY_SIDECAR_FILENAME],
            aslcontext_filename=self.inputs[self.KEY_ASLCONTEXT_FILENAME],
        )

        # create the output source image
        self.outputs[self.KEY_SOURCE] = asl_data.source
        self.outputs[self.KEY_SIDECAR] = asl_data.sidecar
        self.outputs[self.KEY_CONTROL] = asl_data.control
        self.outputs[self.KEY_LABEL] = asl_data.label
        self.outputs[self.KEY_M0] = asl_data.m0

    def _validate_inputs(self) -> None:
        """Checks that inputs meet their validation criteria
        'image_filename' must be a str, .nii or .nii.gz and exist on the file system
        'sidecar_filename' must be a str, end with .json and exist on the file system
        'aslcontext_filename' must be a str, end with .tsv, exist on the file system
        and container 'volume_type' followed by a list comprised of 'control', 'label'
        or 'm0scan'
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE_FILENAME: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ]
                ),
                self.KEY_SIDECAR_FILENAME: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ]
                ),
                self.KEY_ASLCONTEXT_FILENAME: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ]
                ),
            }
        )

        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # Additional validation
        # 'image_filename' should end with .nii or .nii.gz
        if not self.inputs[self.KEY_IMAGE_FILENAME].endswith((".nii", ".nii.gz")):
            raise FilterInputValidationError(
                "LoadAslBidsFilter input 'image_filename' must be a .nii or .nii.gz"
                " file"
            )

        # 'sidecar_filename' should be a .json
        if not self.inputs[self.KEY_SIDECAR_FILENAME].endswith(".json"):
            raise FilterInputValidationError(
                "LoadAslBidsFilter input 'sidecar_filename' must be a .json"
            )

        # 'aslcontex_filename' should be a .tsv
        if not self.inputs[self.KEY_ASLCONTEXT_FILENAME].endswith(".tsv"):
            raise FilterInputValidationError(
                "LoadAslBidsFilter input 'aslcontex_filename' must be a .tsv"
            )

        # check the files actually exist
        for key in self.inputs.keys():
            # the file should exist
            if not os.path.exists(self.inputs[key]):
                raise FilterInputValidationError(f"Input {key} does not exist")

        # check that the contents of the aslcontext file are valid
        with open(
            self.inputs[self.KEY_ASLCONTEXT_FILENAME], encoding="utf-8"
        ) as tsv_file:
            loaded_tsv = tsv_file.readlines()
            tsv_file.close()

        asl_context = [s.strip() for s in loaded_tsv]
        # the first entry should be 'volume_type'
        if not asl_context[0] == "volume_type":
            raise FilterInputValidationError(
                f"{self.inputs[self.KEY_ASLCONTEXT_FILENAME]} does not"
                "start with the string 'volume_type'"
            )

        if not all(volume in SUPPORTED_ASL_CONTEXTS for volume in asl_context[1:]):
            raise FilterInputValidationError(
                f"{self.inputs[self.KEY_ASLCONTEXT_FILENAME]} does not"
                "contain valid asl context strings"
            )

        # length of asl_context should be the same as total number of volumes in image
        image = nifti_reader(self.inputs[self.KEY_IMAGE_FILENAME])
        if not len(asl_context[1:]) == image.dataobj.shape[3]:
            raise FilterInputValidationError(
                "The number of aslcontext entries must be equal to the number of"
                "volumes in the input image"
            )
