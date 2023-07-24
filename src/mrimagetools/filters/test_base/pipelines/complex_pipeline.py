"""A pipeline for adding and multiplying numbers."""

import os
from pathlib import Path
from typing import Any

from mrimagetools.filters.base import (
    ContainerInputFilter,
    DataCombinerFilter,
    DataMassagerFilter,
)

from ..filters.add_multiply_filter import AddMultiplyParams
from ..filters.json_loader import JsonLoaderFilter

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
JSON_FILE = Path(THIS_DIR) / ".." / "data" / "add_multiply.json"


def complex_pipeline() -> DataCombinerFilter[Any, Any, AddMultiplyParams]:
    """For showing the graphing"""

    add_mul_data_filter = JsonLoaderFilter(JSON_FILE, AddMultiplyParams)

    float_input_filter = ContainerInputFilter(5.0)

    # A function that combines data from any two signals
    def addend_multiplier_combiner(
        input_a: AddMultiplyParams, input_b: float
    ) -> AddMultiplyParams:
        """A silly example that arbitrarily combines two inputs by adding the second
        input to the addend of the first"""
        return AddMultiplyParams(
            addend=input_a.addend + input_b, multiplier=input_a.multiplier
        )

    data_combiner = DataCombinerFilter(addend_multiplier_combiner)

    def massage(
        input_value: float,
    ) -> float:
        """A simple massager that adds 1 to the input"""
        return input_value + 1

    # Connect the signals/slots

    # {added:1.0, multiplier:2.0} -> data_combiner.input_a
    add_mul_data_filter.data_output |= data_combiner.input_a

    # 5.0 -> 6.0 -> data_combiner.input_b
    float_input_filter.output |= DataMassagerFilter(massage) | data_combiner.input_b

    # Run the pipeline - outputs the value of
    # {
    # "addend": 7.0,  <- the original value + 5.0 + 1.0
    # "multiplier": 2.0
    # }
    return data_combiner
