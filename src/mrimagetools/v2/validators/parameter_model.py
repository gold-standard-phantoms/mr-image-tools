"""Parameter model classes and function.
Used for parameter validation and typing in the filters"""
from pydantic import BaseModel, ConfigDict


class ParameterModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )


class GenericParameterModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )
