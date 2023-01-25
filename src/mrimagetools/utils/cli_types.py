"""Types to be used with argparse"""

import argparse
import os
from typing import Optional

from mrimagetools.data.filepaths import GROUND_TRUTH_DATA


class HrgtType:  # pylint: disable=too-few-public-methods
    """A HRGT string checker.
    Will determine if the input is a valid HRGT string label.
    """

    def __call__(self, hrgt_label: str) -> str:
        """
        Do the checking
        :param hrgt_label: the HRGT label string
        """
        if hrgt_label not in GROUND_TRUTH_DATA:
            raise argparse.ArgumentTypeError(
                f"{hrgt_label} is not valid, must be one of"
                f" {', '.join(GROUND_TRUTH_DATA.keys())}"
            )
        return hrgt_label


class DirType:  # pylint: disable=too-few-public-methods
    """A directory checker.
    Will determine if the input is a directory and
    optionally, whether it exists
    """

    def __init__(self, should_exist: bool = False) -> None:
        """
        :param should_exist: does the directory have to exist
        """
        self.should_exist: bool = should_exist

    def __call__(self, path: str) -> str:
        """
        Do the checking
        :param path: the path to the directory
        """
        # Always check the file is a directory

        if self.should_exist:
            # Check whether the file exists
            if not os.path.exists(path):
                raise argparse.ArgumentTypeError(f"{path} does not exist")
            if not os.path.isdir(path):
                raise argparse.ArgumentTypeError(f"{path} is not a directory")
        return path


class FileType:  # pylint: disable=too-few-public-methods
    """A file checker.
    Will determine if the input is a valid file name (or path)
    and optionally, whether it has a particular extension and/or exists
    """

    def __init__(
        self, extensions: Optional[list[str]] = None, should_exist: bool = False
    ) -> None:
        """
        :param extensions: a list of allowed file extensions.
        :param should_exist: does the file have to exist
        """
        if not isinstance(extensions, list) and extensions is not None:
            raise TypeError("extensions should be a list of string extensions")

        if extensions is not None:
            for extension in extensions:
                if not isinstance(extension, str):
                    raise TypeError("All extensions must be strings")

        self.extensions: list[str] = []
        if extensions is not None:
            # Strip any proceeding dots
            self.extensions = [
                extension if not extension.startswith(".") else extension[1:]
                for extension in extensions
            ]
        self.should_exist: bool = should_exist

    def __call__(self, path: str) -> str:
        """
        Do the checkstructing
        :param path: the path to the file
        """
        # Always check the file is not a directory
        if os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"{path} is a directory")

        if self.should_exist:
            # Check whether the file exists
            if not os.path.exists(path):
                raise argparse.ArgumentTypeError(f"{path} does not exist")

        if self.extensions:
            valid_extension = False
            for extension in self.extensions:
                if path.lower().endswith(extension.lower()):
                    valid_extension = True
            if not valid_extension:
                raise argparse.ArgumentTypeError(
                    f"{path} is does not have a valid extension "
                    f"(from {', '.join(self.extensions)})"
                )
        return path
