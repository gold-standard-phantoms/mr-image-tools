"""A pipeline that uses the data massager filter"""

import os
from pathlib import Path

from mrimagetools.filters.base import DataMassagerFilter

from ..filters.add_multiply_filter import AddMultiplyParams
from ..filters.json_loader import JsonLoaderFilter

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
JSON_FILE = Path(THIS_DIR) / ".." / "data" / "add_multiply.json"


def data_massager_pipeline() -> DataMassagerFilter:
    """A demo of the data massager filter"""

    # This contains a signal which have the fields of the JSON file
    # {
    # "addend": 1,
    # "multiplier": 2
    # }
    add_mul_data_filter = JsonLoaderFilter(JSON_FILE, AddMultiplyParams)

    # We just want to extract a part of the data from the add_mul_data_filter
    # and pass it to the add_multiply_filter
    def get_addend(input_value: AddMultiplyParams) -> float:
        return input_value.addend

    data_massager = DataMassagerFilter(get_addend)

    # Convert the data types from AddMultiplyParams to a float
    add_mul_data_filter.data_output |= data_massager

    # Run the pipeline - outputs the value of 1.0 (the addend only)

    return data_massager
