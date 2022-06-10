"""
Used to perform parameter validation - additional validators.
"""
import os
from typing import List, Union

from asldro.validators.parameters import Validator


def isfile_validator(
    extensions: Union[str, List[str]] = None, must_exist: bool = False
) -> Validator:
    """
    Validates that a given value is a path to a file.
    :param extension: string of file extensions that must match
    :param must_exist: the file must exist on disk"""

    if extensions is not None:
        for extension in extensions:
            if not isinstance(extension, str):
                raise TypeError(f"All extensions must be strings")

    def validate(value: str) -> bool:
        # check that value is a str
        if isinstance(value, str):
            # check that the value isn't a path
            is_not_dir = not os.path.isdir(value)
            # check that the path actually exists
            root_folder_exists = os.path.exists(os.path.dirname(value))

            # check that the file has a valid extension
            if extensions is not None:
                valid_extension = any(
                    [
                        value.lower().endswith(extension.lower())
                        for extension in extensions
                    ]
                )
            else:
                valid_extension = True
            if must_exist:
                return (
                    is_not_dir
                    and root_folder_exists
                    and valid_extension
                    and os.path.exists(value)
                )
            else:
                return is_not_dir and root_folder_exists and valid_extension
        else:
            return False

    return Validator(
        validate,
        "Value must be a valid filename" + f"with extension(s) {extensions}"
        if extensions is not None
        else "" + "and must exist"
        if must_exist
        else "",
    )
