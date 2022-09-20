""" A validator for the JSON file used in the ground truth input """
from jsonschema import validate

from mrimagetools.validators.schemas.index import load_schemas


def validate_input(input_dict: dict) -> None:
    """Validates the provided dictionary against the ground truth schema.
    Raises a jsonschema.exceptions.ValidationError on error"""
    schema = load_schemas()["asl_ground_truth"]
    validate(instance=input_dict, schema=schema)
