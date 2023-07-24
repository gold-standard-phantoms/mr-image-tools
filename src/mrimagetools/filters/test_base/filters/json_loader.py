""" JSON file loader filter """

from pathlib import Path
from typing import Generic, TypeVar, Union

from pydantic import BaseModel

from mrimagetools.filters.base import BaseFilter, SignalDescriptor

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonLoaderFilter(BaseFilter, Generic[ModelT]):
    """A filter for loading a JSON file."""

    data_output = SignalDescriptor[ModelT]()

    def __init__(self, filename: Union[str, Path], model_type: type[ModelT]):
        """Create a JsonLoader filter.

        :param filename: The filename to load
        :param model_type: The Pydantic model type to parse the JSON file"""

        self.filename: Union[str, Path] = filename
        self.model_type: type[ModelT] = model_type

    def _run(self) -> None:
        """Load the input `filename`. Use the supplied `model_type` to parse.
        The `data_output` signal will be populated with the parsed data and will
        be of type `model_type`."""

        print(f"Running {self}")
        print(f"Loading {self.filename}...")

        with open(self.filename, encoding="utf-8") as json_file:
            results = self.model_type.model_validate_json(json_file.read())
        self.data_output.set_value(results)
