"""A filter that will add and then multiply an input number"""


from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from mrimagetools.filters.base import BaseFilter, SignalDescriptor, SlotDescriptor


@runtime_checkable
class AddMultiplyProtocol(Protocol):
    addend: float
    multiplier: float


class AddMultiplyParams(BaseModel):
    """Parameters for adding and multiplying an input number"""

    addend: float = 0.0
    """Addend to apply to the input value"""

    multiplier: float = 1.0
    """Multiplier to apply to the input value"""


class AddMultiplyFilter(BaseFilter):
    """A filter that will add and then multiply an input number"""

    # Input parameters for the filter
    append_multiplier_input = SlotDescriptor[AddMultiplyProtocol](
        pydantic_model=AddMultiplyParams,
    )

    # Input value for the filter (to be added and multiplied)
    float_input = SlotDescriptor[float]()

    second_float_input = SlotDescriptor[float](optional=True)

    # Output results for the filter
    float_output = SignalDescriptor[float]()

    def _run(self) -> None:
        """Run the filter and populate the `float_output` signal"""
        print(f"float_input: {self.float_input.value}")
        print(f"append_multiplier_input: {self.append_multiplier_input.value}")

        print(f"Running {self}")
        self.float_output.set_value(
            (
                self.float_input.value
                + (
                    # Use the optional parameter "second_float_input" if it is connected
                    self.second_float_input.value
                    if self.second_float_input.is_connected
                    else 0
                )
                + self.append_multiplier_input.value.addend
            )
            * self.append_multiplier_input.value.multiplier
        )
        print(f"float_output: {self.float_output.value}")
