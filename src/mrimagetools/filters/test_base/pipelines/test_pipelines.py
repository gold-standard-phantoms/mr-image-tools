"""Tests for the example pipelines.
For all tests, see a full example of the pipeline in the corresponding
example file.
"""

import gc
from dataclasses import dataclass

import pytest

from mrimagetools.filters.base import ContainerInputFilter, SlotValidationError

from ..filters.add_multiply_filter import AddMultiplyFilter, AddMultiplyParams
from ..pipelines.complex_pipeline import complex_pipeline
from ..pipelines.data_combiner_pipeline import data_combiner_pipeline
from ..pipelines.data_massager_pipeline import data_massager_pipeline
from ..pipelines.simple_pipeline import simple_pipeline


def test_garbage_collection() -> None:
    """Test the garbage collection and weak references.
    This test is here to ensure that that references to garbage collected
    objects are not kept in the object register.
    """
    for _ in range(100):
        # We don't care about the output, just that the pipeline can be solved
        # pylint: disable=expression-not-assigned
        pipeline = simple_pipeline()
        # Force the garbage collector to run and check the object register
        # behaves as expected when data might be put in the same memory location
        # This has triggered a bug in the past
        gc.collect()
        pipeline.solve().float_output.value


def test_simple_pipeline() -> None:
    """Test the simple pipeline"""
    pipeline = simple_pipeline(use_optional_input=False)
    value = pipeline.solve().float_output.value
    assert value == 12.0
    value = simple_pipeline(use_optional_input=True).solve().float_output.value
    assert value == 32.0


def test_data_massager_pipeline() -> None:
    """Test the data massager pipeline"""
    assert data_massager_pipeline().solve().output.value == 1.0


def test_data_combiner_pipeline() -> None:
    """Test the data combiner pipeline"""
    assert data_combiner_pipeline().solve().output.value == AddMultiplyParams(
        addend=6.0, multiplier=2.0
    )


def test_complex_pipeline() -> None:
    """Test the complex pipeline"""
    assert complex_pipeline().solve().output.value == AddMultiplyParams(
        addend=7.0, multiplier=2.0
    )


def test_slot_validation() -> None:
    """Test the slot validation"""

    @dataclass
    class BadParams:
        addend: str
        multiplier: float

    bad_params = BadParams(addend="a", multiplier=1.0)
    add_multiply_filter = AddMultiplyFilter()

    ContainerInputFilter(1.0).output |= add_multiply_filter.float_input
    ContainerInputFilter(
        bad_params
    ).output |= add_multiply_filter.append_multiplier_input

    # This should raise a "SlotValidationError" as the addend should be a float
    # and is a string
    with pytest.raises(SlotValidationError):
        add_multiply_filter.solve()
