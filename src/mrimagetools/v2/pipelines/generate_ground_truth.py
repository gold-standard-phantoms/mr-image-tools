"""Pipeline to generate a ground truth image and save"""
import logging
import os
from typing import Any, Dict, List, Union

import nibabel as nib

from mrimagetools.v2.filters.create_volumes_from_seg_mask import (
    CreateVolumesFromSegMask,
)
from mrimagetools.v2.filters.ground_truth_parser import (
    GroundTruthConfig,
    GroundTruthInput,
    GroundTruthOutput,
    GroundTruthParser,
    Quantity,
)
from mrimagetools.v2.filters.image_tools import FloatToIntImageFilter
from mrimagetools.v2.filters.json_loader import JsonLoaderFilter
from mrimagetools.v2.filters.nifti_loader import NiftiLoaderFilter
from mrimagetools.v2.validators.fields import UnitField
from mrimagetools.v2.validators.schemas.index import SchemaNames, load_schemas

logger = logging.getLogger(__name__)


def generate_hrgt(
    hrgt_params_filename: str,
    seg_mask_filename: str,
    schema_name: SchemaNames,
    output_dir: Union[str, None] = None,
) -> GroundTruthOutput:
    # pylint: disable=too-many-locals, too-many-statements
    """Generates a high-resolution ground truth (hrgt) based on:

        * A segmentation mask image
        * A file describing what values to assign to each region.

    The hrgt is saved in the folder ``output_dir``

    :param hrgt_params_filename: Path to the hrgt parameter JSON file
    :type hrgt_params_filename: str
    :param seg_mask_filename: Path to the segmentation mask NIFTI image
    :type seg_mask_filename: str
    :param output_dir: Directory to save files to, defaults to None
    :type output_dir: str, optional
    :return: dictionary containing the ground truth image, and the ground truth
      parameter file
    :rtype: dict
    """

    # load hrgt_params_filename and validate hrgt_params against the schema
    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", hrgt_params_filename)
    json_filter.add_input("schema", load_schemas()[schema_name])

    # load seg_mask_filename
    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", seg_mask_filename)

    # convert to an integer image, if the image is a float
    round_seg_mask_filter = FloatToIntImageFilter()
    round_seg_mask_filter.add_parent_filter(nifti_filter)  # use default method
    round_seg_mask_filter.add_input(
        FloatToIntImageFilter.KEY_METHOD, FloatToIntImageFilter.CEIL
    )

    create_volume_filter = CreateVolumesFromSegMask()
    create_volume_filter.add_parent_filter(
        round_seg_mask_filter, io_map={"image": "seg_mask"}
    )
    create_volume_filter.add_parent_filter(json_filter)

    create_volume_filter.run()

    create_volume_filter.outputs["image_info"]["parameters"] = json_filter.outputs[
        "parameters"
    ]

    ground_truth_parser = GroundTruthParser()
    ground_truth_parser_input: GroundTruthInput = GroundTruthInput(
        image=create_volume_filter.outputs["image"],
        config=GroundTruthConfig(
            quantities=[
                Quantity(
                    name=name,
                    units=(
                        UnitField.model_validate(unit) if unit is not None else None
                    ),
                )
                for name, unit in zip(
                    create_volume_filter.outputs["image_info"]["quantities"],
                    create_volume_filter.outputs["image_info"]["units"],
                )
            ],
            parameters=json_filter.outputs["parameters"],
            segmentation_labels={
                "seg_label": dict(
                    zip(
                        json_filter.outputs["label_names"],
                        json_filter.outputs["label_values"],
                    )
                )
            },
        ),
    )
    ground_truth_parser.add_inputs(ground_truth_parser_input.dict(exclude_none=True))
    # ground_truth_parser.add_inputs(create_volume_filter.outputs["image_info"])
    # ground_truth_parser.add_input("image", create_volume_filter.outputs["image"])

    ground_truth_parser.run()  # check that no errors occur
    # save the files
    if output_dir is not None:
        json_filename = os.path.join(output_dir, "hrgt.json")
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json_file.write(ground_truth_parser.parsed_outputs.config.json())

        nifti_filename = os.path.join(output_dir, "hrgt.nii.gz")
        nib.nifti2.save(
            create_volume_filter.outputs["image"].nifti_image, nifti_filename
        )

    return ground_truth_parser.parsed_outputs
