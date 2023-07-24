"""Base module for all other filters. Contains the base class, all exceptions and the
object register class which is used to hold the state of the filters/slots/signals
networks
"""

from __future__ import annotations

import logging
import pprint
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel, ValidationError
from typing_extensions import Self

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_BaseFilterT = TypeVar("_BaseFilterT", bound="BaseFilter")


class FilterError(Exception):
    """Base class for all filter errors"""


class FilterNotRegisteredError(FilterError):
    """Raised when a filter has not been registered"""


class SlotError(FilterError):
    """Base class for all slot errors"""


class SlotAlreadyConnectedError(SlotError):
    """Raised when a slot is already connected to a signal but
    is being connected to another one"""


class SlotValidationError(SlotError):
    """Raised when a slot value is evaluated and doesn't meet the
    validation requirements"""


class SlotNotConnectedError(SlotError):
    """Raised when a slot is required but
    not connected to a signal"""


class SlotNotRegisteredError(SlotError):
    """Raised when a slot has not been registered"""


class SignalError(FilterError):
    """Base class for all signal errors"""


class SignalNotRegisteredError(SignalError):
    """Raised when a signal has not been registered"""


class SignalNotPopulatedError(SignalError):
    """Raised when a signal has not been populated (and is required and/or
    being read)"""


class SlotNotPopulatedError(SlotError):
    """Raised when a slot has not been populated (and is required and/or
    being read)"""


@dataclass
class BaseFilter(ABC):
    # pylint: disable=too-few-public-methods
    """Abstract base class for all filters"""

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseFilter:
        """Manually initialise slots and signals via the SlotDescriptor and
        SignalDescriptor"""
        obj = super().__new__(cls)

        for key, value in cls.__dict__.items():
            if isinstance(value, SlotDescriptor):
                logger.debug(
                    "Found slot descriptor %s on instance of %s", key, cls.__name__
                )
                getattr(obj, key)  # call the `get` method to initialise the slot
            if isinstance(value, SignalDescriptor):
                logger.debug(
                    "Found signal descriptor %s on instance of %s", key, cls.__name__
                )
                getattr(obj, key)  # call the `get` method to initialise the signal
        return obj

    @abstractmethod
    def _run(self) -> None:
        """Run the filter. This method must be implemented by each individual filter.
        The method should populate ALL signals that are contained within the filter
        UNLESS they are marked as 'optional'. To populate a signal, use the
        `SignalDescriptor.set_value` method, e.g.:
            `self.output_signal.set_value(value)`
        """

    def solve(self) -> Self:
        """Run the filter (and all required previous filters) and return itself.
        After running this, you can access this filter's signals as they will be
        populated and not cleared. Other filters will not retain their signals.
        This is for memory efficiency. An example of how you might use this is:
            `value = filter_a.solve().output_signal.value`

        :return: The filter itself"""
        ObjRegister.solve_for(self)
        return self

    def visualise(self) -> None:
        """Visualise the graph of all filters, slots and signals.
        This method is only available when the `networkx` and
        `matplotlib` packages are installed."""
        ObjRegister.show_graph(self)

    def __hash__(self) -> int:
        return id(self)


@dataclass
class RegisteredSlot:
    """A slot, its filter and its connected signals. This class is used to keep
    track of which slots are connected to which signals. Hard references are kept so
    that if a single reference to a filter is kept, the referenced filters will not be
    garbage collected
    """

    # The filter that this registered slot is connected to
    filter: RegisteredFilter
    # The actual slot that this registered slot is connected to
    slot: Slot
    # An (optional) signal that this slot is connected to
    connected_signal: Optional[RegisteredSignal]

    def __hash__(self) -> int:
        return id(self.filter)


@dataclass
class RegisteredSignal:
    """A signal, its filter and its connected slots. This class is used to keep
    track of which signals are connected to which slots. Hard references are kept so
    that if a single reference to a filter is kept, the referenced filters will not be
    garbage collected
    """

    filter: RegisteredFilter
    signal: Signal
    connected_slots: list[RegisteredSlot]

    def __hash__(self) -> int:
        return id(self.filter)


@dataclass
class RegisteredFilter:
    """A filter and its connected slots and signals. This class is used to keep
    track of which filters are connected to which slots and signals. Hard references
    are kept so that if a single reference to a filter is kept, the referenced filters
    will not be garbage collected
    """

    filter: BaseFilter
    slots: dict[Slot, RegisteredSlot] = field(default_factory=dict)
    signals: dict[Signal, RegisteredSignal] = field(default_factory=dict)

    def __hash__(self) -> int:
        return id(self.filter)


class ObjRegister:
    """Class for registering objects"""

    # List of objects that have been registered
    # As this is global (class) level state, only hold weak references to the objects
    # so that they can be garbage collected when all references to filters are out of
    # scope. The keys are the id (address) of the object

    _registered_filters: dict[BaseFilter, weakref.ref[RegisteredFilter]] = {}
    _registered_slots: dict[Slot, weakref.ref[RegisteredSlot]] = {}
    _registered_signals: dict[Signal, weakref.ref[RegisteredSignal]] = {}

    @classmethod
    def gc_status(cls) -> str:
        """Return the status of the objects stored in the registered filters,
        slots and signals. In particular, whether they have been garbage collected"""
        filters = [
            s for s in (f() for f in cls._registered_filters.values()) if s is not None
        ]
        slots = [
            s for s in (f() for f in cls._registered_slots.values()) if s is not None
        ]
        signals = [
            s for s in (f() for f in cls._registered_signals.values()) if s is not None
        ]

        return (
            "Registered objects summary:\nFilters:"
            f" {len(cls._registered_filters)} total"
            f" {len(filters)}\nSlots:"
            f" {len(cls._registered_slots)} total"
            f" {len(slots)}\nSignals:"
            f" {len(cls._registered_signals)} total"
            f" {len(signals)}\n"
            f"Filters: {[hex(id(f.filter)) for f in filters]}\n"
            f"Slots: {[hex(id(s.slot)) for s in slots]}\n"
            f"Signals: {[hex(id(s.signal)) for s in signals]}\n"
        )

    @classmethod
    def show_graph(cls, this_filter: BaseFilter) -> None:
        """Show the graph of all registered filters, slots and signals. Requires
        the `networkx` and `matplotlib` packages to be installed.

        :param this_filter: The filter to show the graph for. Will show all connected
        filters"""
        graph: list[tuple[str, str, dict]] = []

        for registered_filter in cls.get_connected_filters(
            cls.register_filter(this_filter)
        ):
            # Show the incoming connections
            for slot, registered_slot in registered_filter.slots.items():
                if registered_slot.connected_signal is None:
                    continue

                connected_signal = registered_slot.connected_signal.signal
                filter_a = registered_slot.connected_signal.filter.filter
                filter_b = registered_slot.filter.filter
                graph.append(
                    (
                        filter_a.__class__.__name__ + "\n" + hex(id(filter_a)),
                        filter_b.__class__.__name__ + "\n" + hex(id(filter_b)),
                        {
                            "slot": slot._descriptor.name,
                            "signal": connected_signal._descriptor.name,
                        },
                    )
                )

        logger.debug(pprint.pformat(graph))

        import matplotlib.pyplot as plt
        import networkx as nx

        di_graph = nx.DiGraph(graph)

        for layer, nodes in enumerate(nx.topological_generations(di_graph)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                di_graph.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(di_graph, subset_key="layer", align="horizontal")

        fig, axis = plt.subplots()
        nx.draw_networkx(di_graph, pos=pos, ax=axis)
        nx.draw_networkx_edge_labels(
            di_graph,
            pos,
            edge_labels={(u, v): d["slot"] for u, v, d in di_graph.edges(data=True)},
            label_pos=0.3,
        )
        nx.draw_networkx_edge_labels(
            di_graph,
            pos,
            edge_labels={(u, v): d["signal"] for u, v, d in di_graph.edges(data=True)},
            label_pos=0.7,
        )
        axis.set_title(f"DAG layout of {this_filter.__class__.__name__}")
        fig.tight_layout()
        plt.show()

    @classmethod
    def register_filter(cls, obj: BaseFilter) -> RegisteredFilter:
        """Register a filter (if not registered already). Uses the id of the object as
        the hash key.

        :param obj: The filter to register
        :return: The registered filter"""
        logger.debug("Registering filter %s with address %s", obj, hex(id(obj)))

        new_registered_filter = RegisteredFilter(filter=obj)

        if (
            previous_registered_filter := cls._registered_filters.get(obj, None)
        ) is not None:
            stored_filter = previous_registered_filter()
            if stored_filter is None:
                # The filter has been garbage collected, so replace it with the new
                # filter
                cls._registered_filters[obj] = weakref.ref(new_registered_filter)
                # Keep the garbage collector from collecting the registered filter
                # by setting it as an attribute on the filter
                setattr(obj, "_registered_filter", new_registered_filter)
                return new_registered_filter

            # The filter has already been registered, so return the stored filter
            return stored_filter

        # Register the new filter
        cls._registered_filters[obj] = weakref.ref(new_registered_filter)

        # Keep the garbage collector from collecting the registered filter
        # by setting it as an attribute on the filter
        setattr(obj, "_registered_filter", new_registered_filter)
        return new_registered_filter

    @classmethod
    def register_slot(cls, obj: BaseFilter, slot: Slot) -> None:
        """Register a slot"""
        logger.debug("Registering slot %s on filter %s", slot, obj)
        registered_filter = cls.register_filter(obj)
        registered_slot = RegisteredSlot(
            filter=registered_filter,
            slot=slot,
            connected_signal=None,
        )
        registered_filter.slots[slot] = registered_slot
        cls._registered_slots[slot] = weakref.ref(registered_slot)

    @classmethod
    def register_signal(cls, obj: BaseFilter, signal: Signal) -> None:
        """Register a signal"""
        logger.debug("Registering signal %s on filter %s", signal, obj)
        registered_filter = cls.register_filter(obj)
        registered_signal = RegisteredSignal(
            filter=registered_filter,
            signal=signal,
            connected_slots=[],
        )
        registered_filter.signals[signal] = registered_signal
        cls._registered_signals[signal] = weakref.ref(registered_signal)

    @classmethod
    def find_registered_filter_from_signal(
        cls, _signal: Union[Signal, RegisteredSignal]
    ) -> RegisteredFilter:
        """Find the registered filter for the signal"""

        registered_signal: RegisteredSignal = (
            _signal
            if isinstance(_signal, RegisteredSignal)
            else cls.get_registered_signal(_signal)
        )

        garbage_collected_registered_filters: list[BaseFilter] = []
        for key, registered_filter in cls._registered_filters.items():
            unwrapped_filter = registered_filter()
            if unwrapped_filter is None:
                garbage_collected_registered_filters.append(key)
                continue
            if registered_signal in unwrapped_filter.signals.values():
                # We're done with the garbage collected registered filters, so remove
                # them
                for (
                    garbage_collected_registered_filter
                ) in garbage_collected_registered_filters:
                    del cls._registered_filters[garbage_collected_registered_filter]
                return unwrapped_filter

        raise SignalNotRegisteredError(f"Signal {_signal} has not been registered")

    @classmethod
    def find_registered_filter_from_slot(
        cls, _slot: Union[Slot, RegisteredSlot]
    ) -> RegisteredFilter:
        """Find the registered filter for the slot"""
        registered_slot: RegisteredSlot = (
            _slot
            if isinstance(_slot, RegisteredSlot)
            else cls.get_registered_slot(_slot)
        )

        garbage_collected_registered_filters: list[BaseFilter] = []
        for key, registered_filter in cls._registered_filters.items():
            unwrapped_filter = registered_filter()
            if unwrapped_filter is None:
                garbage_collected_registered_filters.append(key)
                continue
            if registered_slot in unwrapped_filter.slots.values():
                # We're done with the garbage collected registered filters, so remove
                # them
                for (
                    garbage_collected_registered_filter
                ) in garbage_collected_registered_filters:
                    del cls._registered_filters[garbage_collected_registered_filter]
                return unwrapped_filter

        raise SlotNotRegisteredError(f"Slot {_slot} has not been registered")

    @classmethod
    def get_registered_slot(cls, slot: Slot) -> RegisteredSlot:
        """Get the registered slot"""

        if (registered_slot := cls._registered_slots.get(slot, None)) is None:
            raise SlotNotRegisteredError(f"Slot {slot} has not been registered")

        unwrapped_slot = registered_slot()
        if unwrapped_slot is None:
            raise SlotNotRegisteredError(
                f"Slot {registered_slot} has been garbage collected"
            )

        return unwrapped_slot

    @classmethod
    def get_registered_signal(cls, signal: Signal) -> RegisteredSignal:
        """Get the registered signal"""

        if (registered_signal := cls._registered_signals.get(signal, None)) is None:
            raise SignalNotRegisteredError(f"Signal {signal} has not been registered")

        unwrapped_signal = registered_signal()
        if unwrapped_signal is None:
            raise SignalNotRegisteredError(
                f"Signal {registered_signal} has been garbage collected"
            )

        return unwrapped_signal

    @classmethod
    def _connect_helper(
        cls, _slot_fn: Callable[[_T], None], _signal_fn: Callable[[], _T]
    ) -> None:
        """Helper function for type checking when connecting a signal to a slot"""

    @classmethod
    def connect(
        cls,
        signal: Signal[_T],
        slot: Slot[_T],
    ) -> None:
        """Connect a signal to a slot"""
        if TYPE_CHECKING:
            # This is only used for type checking
            cls._connect_helper(slot.set_value, signal.get_value)

        slot_filter = cls.find_registered_filter_from_slot(slot)
        signal_filter = cls.find_registered_filter_from_signal(signal)
        registered_slot = cls.get_registered_slot(slot)
        registered_signal = cls.get_registered_signal(signal)

        # Add the slot to the signal's list of slots
        signal_filter.signals[signal].connected_slots.append(registered_slot)

        # Only allow a single signal to be connected to a slot
        if slot_filter.slots[slot].connected_signal is not None:
            raise SlotAlreadyConnectedError(
                f"Slot {slot} on filter {slot_filter.filter} is already connected to a"
                "signal"
            )

        # Add the signal to the slot's list of signals
        slot_filter.slots[slot].connected_signal = registered_signal

    @classmethod
    def get_connected_filters(
        cls,
        this_registered_filter: RegisteredFilter,
        visited_filters: Optional[set[RegisteredFilter]] = None,
    ) -> set[RegisteredFilter]:
        """Get all filters connected to this filter"""
        if visited_filters is None:
            visited_filters = set()

        if this_registered_filter in visited_filters:
            # Already visited this slot
            return visited_filters

        visited_filters.add(this_registered_filter)

        # Get all filters with signals to this filter's slots
        for slot in this_registered_filter.slots.values():
            connected_signal = slot.connected_signal

            if connected_signal is not None:
                referenced_filter = cls.find_registered_filter_from_signal(
                    connected_signal
                )
                cls.get_connected_filters(referenced_filter, visited_filters)

        # Get all filters with slots connected to this filter's signals
        for signal in this_registered_filter.signals.values():
            for connected_slot in signal.connected_slots:
                referenced_filter = cls.find_registered_filter_from_slot(connected_slot)
                cls.get_connected_filters(referenced_filter, visited_filters)

        return visited_filters

    @classmethod
    def recursively_clear_network(
        cls,
        this_registered_filter: RegisteredFilter,
        visited_filters: set[RegisteredFilter],
    ) -> None:
        """Recursively clear a slot and all connected signals"""
        if this_registered_filter in visited_filters:
            # Already visited this slot
            return
        logger.debug("Clearing filter %s", this_registered_filter.filter)
        visited_filters.add(this_registered_filter)

        # Clear all slots
        for slot in this_registered_filter.slots.values():
            slot.slot.clear_value()

        # Clear all signals
        for signal in this_registered_filter.signals.values():
            signal.signal.clear_value()

        # Clear filters that a signal connected to one of this filter's slots
        for slot in this_registered_filter.slots.values():
            if slot.connected_signal is not None:
                cls.recursively_clear_network(
                    slot.connected_signal.filter, visited_filters
                )

        # Clear filters that a slot connected to one of this filter's signals
        for signal in this_registered_filter.signals.values():
            for slot in signal.connected_slots:
                cls.recursively_clear_network(slot.filter, visited_filters)

    @classmethod
    def recursively_solve_for_filter(
        cls,
        this_registered_filter: RegisteredFilter,
        visited_filters: set[RegisteredFilter],
    ) -> None:
        """Recursively solve for a filter and all connected slots/signals"""
        if this_registered_filter in visited_filters:
            # Already visited this filter
            return
        logger.debug("Solving for filter %s", this_registered_filter.filter)
        visited_filters.add(this_registered_filter)

        this_filter = this_registered_filter.filter

        # Check if all non optional slots have been connected
        for registered_slot in this_registered_filter.slots.values():
            slot = registered_slot.slot
            if not slot.is_optional:
                if registered_slot.connected_signal is None:
                    raise SlotNotConnectedError(
                        f"Slot {slot} on filter {this_filter} is required but is not"
                        " connected to a signal"
                    )

        # Populate all slots
        for registered_slot in this_registered_filter.slots.values():
            if registered_slot.connected_signal is not None:
                cls.recursively_solve_for_filter(
                    registered_slot.connected_signal.filter, visited_filters
                )

        # Finally, populate the signals of this filter
        this_filter._run()

        # Check all the filter's signals have been populated
        for registered_signal in this_registered_filter.signals.values():
            signal = registered_signal.signal
            if signal.value is None:
                raise SignalNotPopulatedError(
                    f"Signal {signal} on filter {this_filter} has not been populated"
                )

        # Clear all slots
        for registered_slot in this_registered_filter.slots.values():
            registered_slot.slot.clear_value()

        # Copy values from signals to any corresponding slots
        for registered_signal in this_registered_filter.signals.values():
            filters_signal = registered_signal.signal
            for registered_slot in registered_signal.connected_slots:
                registered_slot.slot.set_value(filters_signal.get_value())
            # Finally, clear the signal, unless it has been marked to be stored
            if not filters_signal._store_value:
                filters_signal.clear_value()

    @classmethod
    def solve_for(cls, this_filter: _BaseFilterT) -> _BaseFilterT:
        """Solve the network for a filter - populate all signals"""
        registered_filter = cls.register_filter(this_filter)

        # First clear all the values of connected signals/slots
        cls.recursively_clear_network(registered_filter, set())

        # Don't clear this filter's signals when calculated
        for registered_signal in registered_filter.signals.values():
            registered_signal.signal._store_value = True

        cls.recursively_solve_for_filter(registered_filter, set())
        return this_filter


_DescriptorT = TypeVar(
    "_DescriptorT", bound=Union["SlotDescriptor", "SignalDescriptor"]
)


class SignalSlot(ABC, Generic[_T_co, _DescriptorT]):
    """Base class for signals and slots.
    Signals and slots share a lot of functionality, so this class provides a
    common base class for them.
    """

    def __init__(
        self,
        owner: BaseFilter,
        descriptor: _DescriptorT,
        optional: bool,
        validator: Optional[Callable[[_T_co], _T_co]] = None,
        pydantic_model: Optional[type[BaseModel]] = None,
    ) -> None:
        """Initialise the signal/slot

        :param owner: The filter that owns this signal/slot
        :param slot_descriptor: The slot descriptor for this signal/slot
        :param optional: Whether this signal/slot is optional
        """
        self._value: Optional[_T_co] = None
        self._owner: BaseFilter = owner
        self._descriptor: _DescriptorT = descriptor
        self._optional = optional
        self._validator = validator
        self._pydantic_model = pydantic_model
        # Whether the signal/slot value should be stored after it has been used
        # to populate a connected slot/signal
        self._store_value = False

    def __hash__(self) -> int:
        return id(self)

    def _validate_value(self, value: _T_co) -> _T_co:  # type: ignore
        # Run the validator if one is set
        if self._validator is not None:
            try:
                # There might be some data sanitisation occuring, so return the value
                # after running it through the validator
                return self._validator(value)
            except Exception as exception:
                raise SlotValidationError(str(exception)) from exception

        # Run the pydantic model if one is set
        if self._pydantic_model is not None:
            try:
                self._pydantic_model.model_validate(
                    value, from_attributes=True, strict=True
                )
            except ValidationError as exception:
                raise SlotValidationError(str(exception)) from exception

        return value

    def clear_value(self) -> None:
        """Clears the signal/slot value. This is called after the network has been
        solved and if the value of a signal/slot has not been labelled to be stored."""
        self._value = None
        self._store_value = False

    @property
    def is_optional(self) -> bool:
        """Returns whether this signal/slot is optional

        :return bool: Whether this signal/slot is optional"""
        return self._optional

    def set_value(self, value: _T_co) -> None:  # type: ignore
        """Sets the signal/slot value

        :param value: The value to set"""
        self._value = value

    def get_value(self) -> _T_co:
        """Returns the value of the signal/slot"""
        return self.value

    @property
    def value(self) -> _T_co:
        """Returns the signal/slot value
        :raises SignalNotPopulatedError: If the signal has not been populated
        :raises SlotNotPopulatedError: If the slot has not been populated
        :raises TypeError: If this is neither a signal nor a slot"""

        if self._value is None:
            if isinstance(self._descriptor, SlotDescriptor):
                raise SlotNotPopulatedError(
                    f"Value for slot {self._descriptor.name} for {self._owner} has"
                    " not been initialised or has been cleared"
                )
            if isinstance(self._descriptor, SignalDescriptor):
                raise SignalNotPopulatedError(
                    f"Value for signal {self._descriptor.name} for {self._owner} has"
                    " not been initialised or has been cleared"
                )
            raise TypeError(f"Unknown type of signal/slot {type(self)}")

        return self._validate_value(self._value)


class Slot(SignalSlot[_T, "SlotDescriptor"], Generic[_T]):
    """A slot on a filter. Slots are used to connect filters together. Slots are
    connected to signals, and when a signal is populated and the filter is run/solved
    the slot is populated with the same value. Slots can be optional, in which case
    they do not need to be connected to a signal. If an optional slot is not connected
    to a signal, it will not be populated when the filter is solved for."""

    @property
    def is_connected(self) -> bool:
        """Returns whether this slot is connected to a signal

        :return bool: Whether this slot is connected to a signal"""
        return ObjRegister.get_registered_slot(self).connected_signal is not None


class SignalSlotDescriptor(ABC, Generic[_T]):
    def __init__(
        self,
        *,
        optional: bool = False,
        validator: Optional[Callable[[_T], _T]] = None,
        pydantic_model: Optional[type[BaseModel]] = None,
    ) -> None:
        """Initialise the slot descriptor

        :param optional: Whether this slot is optional
        """

        self.name: Optional[str] = None
        self._optional = optional
        self._validator = validator
        self._pydantic_model = pydantic_model

    def __set_name__(self, owner: BaseFilter, name: str) -> None:
        """Set the name of the variable that this descriptor is assigned to

        :param owner: The owner (instance) of the descriptor
        :param name: The name of the variable that this descriptor is assigned to"""

        self.name = name

    @property
    def _key(self) -> str:
        """Returns the key to be used for storing the connected slot/slot. This is used
        internall by the descriptor to attach the slot to the filter instance"""

        if self.name is None:
            raise ValueError("The name of the descriptor variable has not been set")
        if isinstance(self, SignalDescriptor):
            return f"__{self.name}_signal"
        if isinstance(self, SlotDescriptor):
            return f"__{self.name}_slot"
        raise TypeError(f"Unknown type of descriptor {type(self)}")


class SlotDescriptor(SignalSlotDescriptor[_T], Generic[_T]):
    """A descriptor for a slot"""

    def __get__(self, instance: BaseFilter, owner: type[BaseFilter]) -> Slot[_T]:
        """Get the slot held by this descriptor.
        This should be called on object initialisation, otherwise the Slot instance
        might be missing. This is done automatically by the `BaseFilter.__new__` method

        :param instance: The instance of the filter
        :param owner: The owner (type) of the descriptor
        :return Slot: The slot held by this descriptor
        """
        if instance is None:
            raise ValueError("Slot descriptor can only be accessed via an instance")
        if not isinstance(instance, BaseFilter):
            raise TypeError(
                "Slot descriptor can only be accessed via a BaseFilter instance, not"
                f" {type(instance)}"
            )

        slot = getattr(instance, self._key, None)
        if slot is None:
            logger.debug("Initialising slot for %s", self._key)
            slot = Slot(
                owner=instance,
                descriptor=self,
                optional=self._optional,
                validator=self._validator,
                pydantic_model=self._pydantic_model,
            )
            setattr(instance, self._key, slot)
            # register the slot with the register
            ObjRegister.register_slot(instance, slot)

        if not isinstance(slot, Slot):
            raise TypeError(
                f"Slot descriptor {self.name} for {instance} is not a Slot instance, is"
                f"{type(slot)}: {slot}"
            )

        return slot

    def __set__(self, instance: BaseFilter, value: Slot[_T]) -> None:
        """Set the slot held by this descriptor

        :param instance: The instance of the filter
        :param value: The slot to set
        """

        if instance is None:
            raise ValueError("Slot descriptor can only be accessed via an instance")

        setattr(instance, self._key, value)


class Signal(SignalSlot[_T_co, "SignalDescriptor"], Generic[_T_co]):
    """A signal on a filter. Signals are used to connect filters together. Signals are
    connected to slots, and when a signal is populated and the filter is run/solved
    the slot is populated with the same value. Signals can be optional, in which case
    they do not need to be populated when a filter is run. If the signal is not optional
    and it does not get populated, an error will be raised."""

    @overload
    def __or__(self, other: Slot) -> Self:
        """Connect a signal to a slot via the `|` operator"""

    @overload
    def __or__(self, other: DataMassagerFilter[_T_co, Any]) -> Self:
        """Connect a signal to a data massager via the `|` operator"""

    def __or__(
        self,
        other: Union[
            Slot,
            DataMassagerFilter[_T_co, Any],
        ],
    ) -> Self:
        """Connect a signal to a slot or a data massager via the `|` operator"""
        return self.connect(other)

    @overload
    def connect(self, other: DataMassagerFilter[_T_co, Any]) -> Self:
        """Connect a signal to a data massager

        :param other: The data massager to connect to
        :return: This signal
        """

    @overload
    def connect(self, other: Slot) -> Self:
        """Connect a signal to a slot

        :param other: The slot to connect to
        :return: This signal
        """

    def connect(self, other: Union[Slot, DataMassagerFilter[_T_co, Any]]) -> Self:
        """Connect a signal to a slot or a data massager

        :param other: The slot or data massager to connect to
        :return: This signal"""

        if isinstance(other, DataMassagerFilter):
            ObjRegister.connect(self, other.input)
        else:
            ObjRegister.connect(self, other)
        return self


class SignalDescriptor(SignalSlotDescriptor[_T], Generic[_T]):
    """A descriptor for a signal on a filter. This is used to create a signal on a
    filter. The signal can then be accessed via the descriptor"""

    def __get__(self, instance: BaseFilter, owner: type[BaseFilter]) -> Signal[_T]:
        """Get the signal - this should be called on object initialisation,
        otherwise the Signal instance might be missing. This is called automatically
        by `BaseFilter.__new__`

        :param instance: The instance of the filter
        :param owner: The owner (type) of the descriptor"""

        if instance is None:
            raise ValueError("Signal descriptor can only be accessed via an instance")
        if not isinstance(instance, BaseFilter):
            raise TypeError(
                "Signal descriptor can only be accessed via a BaseFilter instance, not"
                f" {type(instance)}"
            )

        signal = getattr(instance, self._key, None)

        if signal is None:
            logger.debug("Initialising signal for %s", self._key)
            signal = Signal(
                owner=instance,
                descriptor=self,
                optional=self._optional,
                validator=self._validator,
                pydantic_model=self._pydantic_model,
            )
            setattr(instance, self._key, signal)
            # register the signal with the register
            ObjRegister.register_signal(instance, signal)

        if not isinstance(signal, Signal):
            raise TypeError(
                f"Signal descriptor {self.name} for {instance} is not a Signal"
                f" instance, is{type(signal)}: {signal}"
            )

        return signal

    def __set__(self, instance: BaseFilter, value: Signal[_T]) -> None:
        """Set the signal value
        :param instance: The instance of the filter
        :param value: The value to set the signal to"""

        if instance is None:
            raise ValueError("Signal descriptor can only be accessed via an instance")

        setattr(instance, self._key, value)


_InputT = TypeVar("_InputT")
_FirstInputT = TypeVar("_FirstInputT")
_SecondInputT = TypeVar("_SecondInputT")
_OutputT = TypeVar("_OutputT")


class DataMassagerFilter(BaseFilter, Generic[_InputT, _OutputT]):
    """A filter that will transfer a parameter from one type to another. The
    filter is generic and the input and output (value) types will be determined
    by the function passed to the constructor. _InputT is the type of the input
    parameter and _OutputT is the type of the output parameter.
    """

    input = SlotDescriptor[_InputT]()
    output = SignalDescriptor[_OutputT]()

    def __init__(self, fn: Callable[[_InputT], _OutputT]) -> None:
        """Initialise the filter

        :param fn: The function to use to convert the input to the output. Must
        have a single parameter and single output parameter. The type of the
        `output` signal will be the same as the type of the output parameter of
        `fn`. The type of the `input` slot will be the same as the type of the
        input parameter of `fn`"""

        self._fn = fn
        super().__init__()

    def _run(self) -> None:
        """Run the filter and populate the output signal"""

        logger.debug("Running %s", self)
        self.output.set_value(self._fn(self.input.value))

    def connect(self, slot: Slot[_OutputT]) -> Self:
        """Connect a signal to a slot

        :param slot: The slot to connect to
        :return: This filter"""

        ObjRegister.connect(self.output, slot)
        return self

    def __or__(self, other: Slot[_OutputT]) -> Self:
        """Connect a signal to a slot via the `|` operator. This is equivalent
        to calling `connect` on the filter

        :param other: The slot to connect to
        :return: This filter"""

        return self.connect(other)


class DataCombinerFilter(BaseFilter, Generic[_FirstInputT, _SecondInputT, _OutputT]):
    """A filter that will combine two signals into one. The filter is generic
    and the input and output (value) types will be determined by the function
    passed to the constructor. _FirstInputT is the type of the first input
    parameter, _SecondInputT is the type of the second input parameter and
    _OutputT is the type of the output parameter.
    """

    input_a = SlotDescriptor[_FirstInputT]()
    input_b = SlotDescriptor[_SecondInputT]()
    output = SignalDescriptor[_OutputT]()

    def __init__(self, fn: Callable[[_FirstInputT, _SecondInputT], _OutputT]) -> None:
        """Initialise the filter

        :param fn: The function to use to combine the inputs to the output. Must have
        two parameters and a single output parameter. The type of the `output` signal
        will be the same as the type of the output parameter of `fn`. The type of the
        `input_a` slot will be the same as the type of the first input parameter of
        `fn` and the type of the `input_b` slot will be the same as the type of the
        second input parameter of `fn`"""
        self._fn = fn
        super().__init__()

    def _run(self) -> None:
        """Run the filter and populate the output signal"""
        logger.debug("Running %s", self)
        self.output.set_value(self._fn(self.input_a.value, self.input_b.value))

    def connect(self, slot: Slot[_OutputT]) -> Self:
        """Connect a signal to a slot.

        :param slot: The slot to connect to
        :return: This filter"""
        ObjRegister.connect(self.output, slot)
        return self

    def __or__(self, other: Slot[_OutputT]) -> Self:
        """Connect a signal to a slot via the `|` operator. This is equivalent
        to calling `connect` on the filter

        :param other: The slot to connect to
        :return: This filter"""

        return self.connect(other)


class ContainerInputFilter(BaseFilter, Generic[_T]):
    """A filter that will take a statically defined input and output that input as a
    signal. This filter is generic and will be typed based on the input type. e.g.
    `ContainerInputFilter(1.0)` will create a signal of type `float`"""

    output = SignalDescriptor[_T]()

    def __init__(self, value: _T) -> None:
        """Initialise the filter.

        :param value: The value to output as a signal"""
        self.value = value
        super().__init__()

    def _run(self) -> None:
        """Run the filter and populate the output signal"""
        logger.debug("Running %s", self)
        self.output.set_value(self.value)

    def connect(self, slot: Slot[_T]) -> Self:
        """Connect a signal to a slot.

        :param slot: The slot to connect to
        :return: This filter"""
        ObjRegister.connect(self.output, slot)
        return self

    def __or__(self, other: Slot[_T]) -> Self:
        """Connect a signal to a slot via the `|` operator.

        :param other: The slot to connect to
        :return: This filter"""
        return self.connect(other)
