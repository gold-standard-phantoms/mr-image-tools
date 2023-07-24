"""Index of the JSON schema files"""
import json
import os
from typing import Any, Dict, Literal, get_args

from jsonschema import Draft7Validator

from mrimagetools.v2.filters.ground_truth_parser import GroundTruthConfig

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

# Typing information for the schema names - should mirror the .json files in this directory
SchemaNames = Literal[
    "asl_ground_truth",
    "dwi_ground_truth",
    "asl_quantification",
    "combine_masks",
    "generate_asl_hrgt_params",
    "generate_dwi_hrgt_params",
    "generate_hrgt_params",
    "input_params",
]


def load_schemas() -> dict[SchemaNames, Any]:
    """Return all of the schemas in this directory in a dictionary where
    the keys are the filename (without the .json extension) and the values
    are the JSON schemas (in dictionary format)
    :raises jsonschema.exceptions.SchemaError if any of the JSON files in this
    directory are not valid (Draft 7) JSON schemas"""
    schemas: dict[SchemaNames, Any] = {}

    for schema in get_args(SchemaNames):
        # Override some of the JSON schemas
        if schema == "asl_ground_truth":
            schemas[schema] = GroundTruthConfig.schema()
            continue
        filename = os.path.join(THIS_DIR, schema + ".json")
        if not os.path.isfile(filename):
            raise ValueError(f"{filename} for schema {schema} does not exist")
        with open(os.path.join(THIS_DIR, filename), encoding="utf-8") as file_obj:
            value = json.load(file_obj)
            Draft7Validator.check_schema(value)
        schemas[schema] = value

    return schemas
