""" Test the data here is valid """
import json

from mrimagetools.data.filepaths import GROUND_TRUTH_DATA
from mrimagetools.validators.ground_truth_json import validate_input


def test_hrgt_icbm_2009a_nls_3t() -> None:
    """The high resolution ground truth json
    must be valid as per the schema"""

    with open(GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["json"]) as file:
        validate_input(json.load(file))


def test_hrgt_icbm_2009a_nls_1_5t() -> None:
    """The high resolution ground truth json
    must be valid as per the schema"""

    with open(GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_1.5t"]["json"]) as file:
        validate_input(json.load(file))
