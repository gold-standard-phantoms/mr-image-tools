"""Load ASL BIDS filter class"""

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mrimagetools.v2.containers.image import BaseImageContainer, NiftiImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.utils.io import nifti_reader


@dataclass
class AslBidsData:
    """Dataclass for ASL BIDS data"""

    source: BaseImageContainer
    """The full ASL NIFTI image"""

    control: BaseImageContainer
    """Control volumes (as defined by aslcontext)"""

    label: BaseImageContainer
    """Label volumes (as defined by aslcontext)"""

    m0: BaseImageContainer
    """M0 volumes (as defined by aslcontext)"""

    sidecar: dict
    """The sidecar metadata"""


AslTypes = Literal["control", "label", "m0"]
AslContext = Literal["control", "label", "m0scan"]

ASL_CONTEXT_MAPPING: dict[AslTypes, AslContext] = {
    "control": "control",
    "label": "label",
    "m0": "m0scan",
}
LIST_FIELDS_TO_EXCLUDE = [
    "ScanningSequence",
    "ComplexImageComponent",
    "ImageType",
    "AcquisitionVoxelSize",
]


def load_asl_bids(
    image_filename: str, sidecar_filename: str, aslcontext_filename: str
) -> AslBidsData:
    """
    Loads in ASL data in BIDS format, comprising of a NIFTI image file, json
    sidear and tsv aslcontext file.  After loading in the data, image containers are
    created using the volumes described in aslcontext.  For each of these containers,
    the data in sidecar is added to the metadata object.  In addition a metadata
    'asl_context' is created which is a list of the corresponding volumes contained in
    each container.  Any metadata entries that are an array and specific to each volume
    have only the corresponding values copied.

    :param image_filename: path and filename to the ASL NIFTI image
        (must end in .nii or.nii.gz)
    :param sidecar_filename: path and filename to the json sidecar (must end in .json)
    :param aslcontext_filename: path and filename to the aslcontext file
        (must end in .tsv).  This must be a tab separated values file, with heading
        'volume_type' and then entries which are either 'control', 'label',
        or 'm0scan'.

    :return: AslBidsData object containing the source image and the control, label and
        m0scan containers (see :class:`AslBidsData`)
    """
    # load in the NIFTI image
    image = nifti_reader(image_filename)
    # load in the sidecar
    with open(sidecar_filename, encoding="utf-8") as json_file:
        output_sidecar = json.load(json_file)
        json_file.close()
    # load in the aslcontext tsv
    with open(aslcontext_filename, encoding="utf-8") as tsv_file:
        loaded_tsv = tsv_file.readlines()
        tsv_file.close()

    # get the ASL context array
    asl_context: list[AslContext] = [s.strip() for s in loaded_tsv][1:]  # type: ignore
    for i in asl_context:
        if i not in ASL_CONTEXT_MAPPING.values():
            raise ValueError(
                f"aslcontext file {aslcontext_filename}"
                f" contains an invalid volume type {i}"
            )

    # create the output source image
    output_source = NiftiImageContainer(
        image, metadata=ImageMetadata.from_bids(output_sidecar)
    )
    output_source.metadata.asl_context = asl_context

    # iterate over 'control', 'label' and 'm0'. Determine which volumes correspond using
    # asl_context, then place the volumes into new cloned image containers and update
    # the metadata entry 'asl_context'
    outputs: dict[AslTypes, BaseImageContainer] = {}

    keys: list[AslTypes] = ["control", "label", "m0"]
    for key in keys:
        volume_indices = [
            i for (i, val) in enumerate(asl_context) if val == ASL_CONTEXT_MAPPING[key]
        ]
        if volume_indices is not None:
            outputs[key] = output_source.clone()
            outputs[key].image = np.squeeze(
                output_source.image[:, :, :, volume_indices]
            )
            outputs[key].metadata.asl_context = [asl_context[i] for i in volume_indices]
            # adjust any lists in the metdata that correspond to a value per volume
            for metadata_key in (
                outputs[key].metadata.model_dump(exclude_none=True).keys()
            ):
                if metadata_key not in LIST_FIELDS_TO_EXCLUDE:
                    if isinstance(getattr(outputs[key].metadata, metadata_key), list):
                        if len(getattr(outputs[key].metadata, metadata_key)) == len(
                            asl_context
                        ):
                            setattr(
                                outputs[key].metadata,
                                metadata_key,
                                [
                                    getattr(outputs[key].metadata, metadata_key)[i]
                                    for i in volume_indices
                                ],
                            )
    return AslBidsData(
        source=output_source,
        control=outputs["control"],
        label=outputs["label"],
        m0=outputs["m0"],
        sidecar=output_sidecar,
    )
