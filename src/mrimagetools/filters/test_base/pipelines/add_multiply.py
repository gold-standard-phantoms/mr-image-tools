"""A pipeline for adding and multiplying numbers."""

import os
from pathlib import Path

from mrimagetools.filters.base import (
    ContainerInputFilter,
    DataCombinerFilter,
    DataMassagerFilter,
)

from ..filters.add_multiply_filter import AddMultiplyFilter, AddMultiplyParams
from ..filters.json_loader import JsonLoaderFilter

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
JSON_FILE = Path(THIS_DIR) / ".." / "data" / "add_multiply.json"


def simple_pipeline(use_optional_input: bool = False) -> None:
    """A very simple pipeline that adds and multiplies numbers loaded
    from a JSON file"""

    # This contains a signal which have the fields of the JSON file
    # {
    # "addend": 1,
    # "multiplier": 2
    # }
    add_mul_data_filter = JsonLoaderFilter(JSON_FILE, AddMultiplyParams)

    # This contains a slot and a signal (which is not connected)
    add_multiply_filter = AddMultiplyFilter()

    float_input_filter = ContainerInputFilter(5.0)

    # Populate the "input_params" slot of the add_multiply_filter
    add_mul_data_filter.data_output |= add_multiply_filter.append_multiplier_input
    float_input_filter.output |= add_multiply_filter.float_input

    if use_optional_input:
        # Populate the "second_float_input" slot of the add_multiply_filter
        second_float_input_filter = ContainerInputFilter(10.0)
        second_float_input_filter.output |= add_multiply_filter.second_float_input

    # (5.0 + 1.0 + (10.0?)) * 2.0 = 12.0 or 32.0
    # Solve and output the result of 12.0 or 32.0

    # Show the graph of the pipeline
    add_multiply_filter.visualise()
    print(add_multiply_filter.solve().float_output.value)


def data_massager_pipeline() -> None:
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
    data_massager.visualise()
    print(data_massager.solve().output.value)


def data_combiner_pipeline() -> None:
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
    data_combiner.visualise()
    print(data_combiner.solve().output.value)


def complex_pipeline() -> None:
    """For showing the graphing"""

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

    def massage(
        input_value: float,
    ) -> float:
        return input_value + 1

    data_combiner = DataCombinerFilter(addend_multiplier_combiner)

    # Connect the signals to the combiner
    add_mul_data_filter.data_output |= data_combiner.input_a
    float_input_filter.output |= DataMassagerFilter(massage) | data_combiner.input_b

    # Run the pipeline - outputs the value of
    # {
    # "addend": 6.0,  <- the original value + 5.0
    # "multiplier": 2.0
    # }
    data_combiner.visualise()
    print(data_combiner.solve().output.value)


def main() -> None:
    """Main function for module"""
    simple_pipeline()
    data_massager_pipeline()
    data_combiner_pipeline()
    complex_pipeline()


if __name__ == "__main__":
    main()
