import os
from tempfile import TemporaryDirectory

import pytest

from neuroqa.validators.parameters import isfile_validator


def test_isfile_validator_creator():
    """Check the isfile validator raises errors when extension is
    not a string or list of strings
    """
    with pytest.raises(TypeError):
        isfile_validator(1)
        isfile_validator(
            [
                1,
                1,
            ]
        )
        isfile_validator([".ext", ".nii", 1])


def test_isfile_validator():
    """Tests the isfile validator with some values"""

    with TemporaryDirectory() as temp_dir:

        file_name = os.path.join(temp_dir, "file.txt")
        validator = isfile_validator()
        assert not validator(1)  # non-string
        assert not validator("a string")  # a string that isn't a path
        assert validator(file_name)  # valid filename
        assert not validator(temp_dir)  # just a directory
        assert not validator(
            os.path.join(temp_dir, "a_folder", "file.txt")
        )  # dir doesn't exist
        validator = isfile_validator(extensions=".json")
        assert not validator(file_name)  # wrong extension
        validator = isfile_validator(extensions=[".json", ".txt"])
        assert validator(file_name)  # correct extension
        validator = isfile_validator(must_exist=True)
        assert not validator(file_name)

        f = open(file_name, "w")
        f.write("some text")
        f.close()
        assert validator(file_name)
