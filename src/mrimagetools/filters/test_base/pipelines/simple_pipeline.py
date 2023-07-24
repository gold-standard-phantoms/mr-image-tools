"""A simple pipeline"""

import os
from pathlib import Path

from mrimagetools.filters.base import ContainerInputFilter, ObjRegister

from ..filters.add_multiply_filter import AddMultiplyFilter, AddMultiplyParams
from ..filters.json_loader import JsonLoaderFilter

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
JSON_FILE = Path(THIS_DIR) / ".." / "data" / "add_multiply.json"


def simple_pipeline(use_optional_input: bool = False) -> AddMultiplyFilter:
    """A very simple pipeline that adds and multiplies numbers loaded
    from a JSON file."""

    # This contains a signal which have the fields of the JSON file
    # {
    # "addend": 1,
    # "multiplier": 2
    # }
    json_loader_filter = JsonLoaderFilter(JSON_FILE, AddMultiplyParams)
    print(f"json_loader_filter: {hex(id(json_loader_filter))}")
    print(ObjRegister.gc_status())

    # This contains a slot and a signal (which is not connected)
    add_multiply_filter = AddMultiplyFilter()
    print(f"add_multiply_filter: {hex(id(add_multiply_filter))}")

    float_input_filter = ContainerInputFilter(5.0)
    print(f"float_input_filter: {hex(id(float_input_filter))}")

    # Populate the "input_params" slot of the add_multiply_filter
    json_loader_filter.data_output |= add_multiply_filter.append_multiplier_input
    float_input_filter.output |= add_multiply_filter.float_input

    if use_optional_input:
        # Populate the "second_float_input" slot of the add_multiply_filter
        second_float_input_filter = ContainerInputFilter(10.0)
        second_float_input_filter.output |= add_multiply_filter.second_float_input

    # (5.0 + 1.0 + (10.0?)) * 2.0 = 12.0 or 32.0
    # Solve and output the result of 12.0 or 32.0

    return add_multiply_filter
