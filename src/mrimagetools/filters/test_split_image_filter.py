"""Tests for split image filter"""

import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from mrimagetools.containers.image import NiftiImageContainer
from mrimagetools.filters.split_image_filter import SplitImageFilter
from mrimagetools.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="test_data")
def test_data_fixture() -> dict:
    """Returns a dictionary with data for testing"""
    data = {
        f"image_{n+1}d": NiftiImageContainer(
            nib.Nifti1Image(
                np.stack([i * np.ones(([8] * n)) for i in range(10)], axis=n),
                affine=np.eye(4),
            ),
            metadata={"field_1": "one", "field_2": 2},
        )
        for n in range(1, 4)
    }
    data["image_1d"] = NiftiImageContainer(
        nib.Nifti1Image(np.arange(10), np.eye(4)),
        metadata={"field_1": "one", "field_2": 2},
    )
    return data


@pytest.fixture(name="validation_data")
def validation_data_fixture(test_data) -> dict:
    """Returns a dictionary for input validation testing"""

    return {
        "validation_1d": {
            "image": [False, test_data["image_1d"], np.ones((4, 4, 4, 4)), 1, "str"],
            "axis": [False, 0, 5, 3.0, "str"],
            "indices": [False, [2, 4, 6], [], 10, "str"],
        },
        "validation_2d": {
            "image": [False, test_data["image_2d"], np.ones((4, 4, 4, 4)), 1, "str"],
            "axis": [False, 1, 5, 3.0, "str"],
            "indices": [False, [2, 4, 6], [], 10, "str"],
        },
        "validation_3d": {
            "image": [False, test_data["image_3d"], np.ones((4, 4, 4, 4)), 1, "str"],
            "axis": [False, 2, 5, 3.0, "str"],
            "indices": [False, [2, 4, 6], [], 10, "str"],
        },
        "validation_4d": {
            "image": [False, test_data["image_4d"], np.ones((4, 4, 4, 4)), 1, "str"],
            "axis": [False, 3, 5, 3.0, "str"],
            "indices": [False, [2, 4, 6], [], 10, "str"],
        },
    }


def test_split_image_filter_validate_inputs(validation_data) -> None:
    """Checkes that a FilterInputValidationError is raised when the inputs
    to the SplitImageFilter are incorrect or missing."""

    for key in validation_data.keys():
        validate_filter_inputs(SplitImageFilter, validation_data[key])


def test_split_image_filter_mock_data_1d(test_data) -> None:
    """Tests the SplitImageFilter with some 1d mock data"""

    split_image_filter = SplitImageFilter()
    split_image_filter.add_input("image", test_data["image_1d"])
    split_image_filter.add_input("axis", 0)
    split_image_filter.add_input("indices", [3])
    split_image_filter.run()

    assert list(split_image_filter.outputs.keys()) == ["image_0", "image_1"]
    assert split_image_filter.outputs["image_0"].shape == (3,)
    assert split_image_filter.outputs["image_1"].shape == (7,)

    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_0"].image, [0, 1, 2]
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_1"].image, [3, 4, 5, 6, 7, 8, 9]
    )

    assert split_image_filter.outputs["image_0"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }
    assert split_image_filter.outputs["image_1"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }


def test_split_image_filter_mock_data_2d(test_data) -> None:
    """Tests the SplitImageFilter with some 2d mock data"""

    split_image_filter = SplitImageFilter()
    split_image_filter.add_input("image", test_data["image_2d"])
    split_image_filter.add_input("axis", 1)
    split_image_filter.add_input("indices", [3])
    split_image_filter.run()

    assert list(split_image_filter.outputs.keys()) == ["image_0", "image_1"]
    assert split_image_filter.outputs["image_0"].shape == (8, 3)
    assert split_image_filter.outputs["image_1"].shape == (8, 7)

    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_0"].image, test_data["image_2d"].image[:, :3]
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_1"].image, test_data["image_2d"].image[:, 3:]
    )

    assert split_image_filter.outputs["image_0"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }
    assert split_image_filter.outputs["image_1"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }


def test_split_image_filter_mock_data_3d(test_data) -> None:
    """Tests the SplitImageFilter with some 4d mock data"""

    split_image_filter = SplitImageFilter()
    split_image_filter.add_input("image", test_data["image_3d"])
    split_image_filter.add_input("axis", 2)
    split_image_filter.add_input("indices", [3, 6])
    split_image_filter.run()

    assert list(split_image_filter.outputs.keys()) == ["image_0", "image_1", "image_2"]
    assert split_image_filter.outputs["image_0"].shape == (8, 8, 3)
    assert split_image_filter.outputs["image_1"].shape == (8, 8, 3)
    assert split_image_filter.outputs["image_2"].shape == (8, 8, 4)

    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_0"].image,
        test_data["image_3d"].image[:, :, :3],
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_1"].image,
        test_data["image_3d"].image[:, :, 3:6],
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_2"].image,
        test_data["image_3d"].image[:, :, 6:],
    )

    assert split_image_filter.outputs["image_0"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }
    assert split_image_filter.outputs["image_1"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }
    assert split_image_filter.outputs["image_2"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }


def test_split_image_filter_mock_data_4d(test_data) -> None:
    """Tests the SplitImageFilter with some 4d mock data"""

    split_image_filter = SplitImageFilter()
    split_image_filter.add_input("image", test_data["image_4d"])
    split_image_filter.add_input("axis", 3)
    split_image_filter.add_input("indices", [3, 6])
    split_image_filter.run()

    assert list(split_image_filter.outputs.keys()) == ["image_0", "image_1", "image_2"]
    assert split_image_filter.outputs["image_0"].shape == (8, 8, 8, 3)
    assert split_image_filter.outputs["image_1"].shape == (8, 8, 8, 3)
    assert split_image_filter.outputs["image_2"].shape == (8, 8, 8, 4)

    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_0"].image,
        test_data["image_4d"].image[:, :, :, :3],
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_1"].image,
        test_data["image_4d"].image[:, :, :, 3:6],
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_2"].image,
        test_data["image_4d"].image[:, :, :, 6:],
    )

    assert split_image_filter.outputs["image_0"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }
    assert split_image_filter.outputs["image_1"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }
    assert split_image_filter.outputs["image_2"].metadata == {
        "field_1": "one",
        "field_2": 2,
    }


def test_split_image_filter_mock_data_4d_unsorted_indices(test_data) -> None:
    """Tests the SplitImageFilter with some 4d mock data where
    the indices are not sorted"""

    split_image_filter = SplitImageFilter()
    split_image_filter.add_input("image", test_data["image_4d"])
    split_image_filter.add_input("axis", 3)
    split_image_filter.add_input("indices", [7, 4, 2])
    split_image_filter.run()

    assert list(split_image_filter.outputs.keys()) == [
        "image_0",
        "image_1",
        "image_2",
        "image_3",
    ]
    assert split_image_filter.outputs["image_0"].shape == (8, 8, 8, 2)
    assert split_image_filter.outputs["image_1"].shape == (8, 8, 8, 2)
    assert split_image_filter.outputs["image_2"].shape == (8, 8, 8, 3)
    assert split_image_filter.outputs["image_3"].shape == (8, 8, 8, 3)

    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_0"].image,
        test_data["image_4d"].image[:, :, :, :2],
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_1"].image,
        test_data["image_4d"].image[:, :, :, 2:4],
    )
    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_2"].image,
        test_data["image_4d"].image[:, :, :, 4:7],
    )

    numpy.testing.assert_array_equal(
        split_image_filter.outputs["image_3"].image,
        test_data["image_4d"].image[:, :, :, 7:],
    )
