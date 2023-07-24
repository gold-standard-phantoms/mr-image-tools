"""A pipeline that uses the data combiner filter"""

import os
from pathlib import Path
from typing import Any

from mrimagetools.filters.base import ContainerInputFilter, DataCombinerFilter

from ..filters.add_multiply_filter import AddMultiplyParams
from ..filters.json_loader import JsonLoaderFilter

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
JSON_FILE = Path(THIS_DIR) / ".." / "data" / "add_multiply.json"


def data_combiner_pipeline() -> DataCombinerFilter[Any, Any, AddMultiplyParams]:
    """A demo of the data combiner filter"""

    # This contains a signal which have the fields of the JSON file
    # {
    # "addend": 1,
    # "multiplier": 2
    # }
    add_mul_data_filter = JsonLoaderFilter(JSON_FILE, AddMultiplyParams)

    float_input_filter = ContainerInputFilter(5.0)

    # A function that combines data from any two signals
    def addend_multiplier_combiner(
        input_a: AddMultiplyParams, input_b: float
    ) -> AddMultiplyParams:
        """A silly example that arbitrarily combines two inputs"""
        return AddMultiplyParams(
            addend=input_a.addend + input_b, multiplier=input_a.multiplier
        )

    data_combiner = DataCombinerFilter(addend_multiplier_combiner)

    # Connect the signals to the combiner
    add_mul_data_filter.data_output |= data_combiner.input_a
    float_input_filter.output |= data_combiner.input_b

    # Run the pipeline - outputs the value of
    # {
    # "addend": 6.0,  <- the original value + 5.0
    # "multiplier": 2.0
    # }
    return data_combiner


def main():
    pipeline = data_combiner_pipeline()
    print(pipeline.solve().output.value)
    pipeline.visualise()


if __name__ == "__main__":
    main()
