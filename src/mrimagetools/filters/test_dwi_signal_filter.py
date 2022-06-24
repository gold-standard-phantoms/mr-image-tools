"""Test DwiSignalFilter"""

import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from mrimagetools.containers.image import NiftiImageContainer
from mrimagetools.filters.dwi_signal_filter import DwiSignalFilter
from mrimagetools.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="test_data")
def test_data_fixture() -> dict:
    """Returns a dictionary with data for testing"""
    b_values = [1]
    b_vectors = [[1, 1, 1]]  # what if 0 values in b_vector

    test_dims = (2, 2, 2, 1)  # shape (x, x, x, len(b_values))
    attenuation = np.arange(8, dtype=np.float32).reshape(test_dims)
    # this is the different attenuation coefficient we simulate
    adc_img_val = np.zeros((2, 2, 2, 3))
    s0_img_val = np.ones((2, 2, 2))
    # supposidely you can get adc only with one of the 3D element of image
    # hence why I only used one b_value
    for i in range(0, 3):
        adc_img_val[:, :, :, i] = i
    adc = NiftiImageContainer(
        nib.Nifti1Image(
            adc_img_val,
            affine=np.eye(4),
        )
    )  # shape (x, x, x, 3)
    s0 = NiftiImageContainer(
        nib.Nifti1Image(
            s0_img_val,
            affine=np.eye(4),
        )
    )
    attenuation[:, :, :, 0] = [
        [
            [0.04978707, 0.04978707],  # 0.04978707
            [0.04978707, 0.04978707],
        ],
        [
            [0.04978707, 0.04978707],
            [0.04978707, 0.04978707],
        ],
    ]

    dwi = attenuation

    return {
        "adc": adc,
        "b_values": b_values,
        "b_vectors": b_vectors,
        "s0": s0,
        "attenuation": attenuation,
        "dwi": dwi,
    }


@pytest.fixture(name="test_data_s0_null")
def test_data_s0_null_fixture() -> dict:
    """Returns a dictionary with data for testing"""
    b_values = [1]
    b_vectors = [[1, 1, 1]]  # what if 0 values in b_vector

    test_dims = (2, 2, 2, 1)  # shape (x, x, x, len(b_values))
    attenuation = np.arange(8, dtype=np.float32).reshape(test_dims)
    # this is the different attenuation coefficient we simulate
    adc_img_val = np.zeros((2, 2, 2, 3))
    s0_img_val = np.zeros((2, 2, 2))
    # supposidely you can get adc only with one of the 3D element of image
    # hence why I only used one b_value
    for i in range(0, 3):
        adc_img_val[:, :, :, i] = i
    adc = NiftiImageContainer(
        nib.Nifti1Image(
            adc_img_val,
            affine=np.eye(4),
        )
    )  # shape (x, x, x, 3)
    s0 = NiftiImageContainer(
        nib.Nifti1Image(
            s0_img_val,
            affine=np.eye(4),
        )
    )
    attenuation[:, :, :, 0] = [
        [
            [0.04978707, 0.04978707],  # 0.04978707
            [0.04978707, 0.04978707],
        ],
        [
            [0.04978707, 0.04978707],
            [0.04978707, 0.04978707],
        ],
    ]

    dwi = np.zeros((2, 2, 2, 1))

    return {
        "adc": adc,
        "b_values": b_values,
        "b_vectors": b_vectors,
        "s0": s0,
        "attenuation": attenuation,
        "dwi": dwi,
    }


@pytest.fixture(name="test_data_no_s0")
def test_data_no_s0_fixture() -> dict:
    """Returns a dictionary with data for testing"""
    b_values = [1]
    b_vectors = [[1, 1, 1]]  # what if 0 values in b_vector

    test_dims = (2, 2, 2, 1)  # shape (x, x, x, len(b_values))
    attenuation = np.arange(8, dtype=np.float32).reshape(test_dims)
    # this is the different attenuation coefficient we simulate
    adc_img_val = np.zeros((2, 2, 2, 3))
    # supposidely you can get adc only with one of the 3D element of image
    # hence why I only used one b_value
    for i in range(0, 3):
        adc_img_val[:, :, :, i] = i
    adc = NiftiImageContainer(
        nib.Nifti1Image(
            adc_img_val,
            affine=np.eye(4),
        )
    )  # shape (x, x, x, 3)

    attenuation[:, :, :, 0] = [
        [
            [0.04978707, 0.04978707],  # 0.04978707
            [0.04978707, 0.04978707],
        ],
        [
            [0.04978707, 0.04978707],
            [0.04978707, 0.04978707],
        ],
    ]

    dwi = attenuation

    return {
        "adc": adc,
        "b_values": b_values,
        "b_vectors": b_vectors,
        "attenuation": attenuation,
        "dwi": dwi,
    }


@pytest.fixture(name="test_data_complex")
def test_data_complex() -> dict:
    """Returns a dictionary with data for testing"""
    b_values = [1, 2]
    b_vectors = [[1, 1, 1], [1, 2, 3]]  # what if 0 values in b_vector

    test_dims = (2, 2, 2, 2)  # shape (x, x, x, len(b_values))
    attenuation = np.arange(16, dtype=np.float32).reshape(test_dims)
    # this is the different attenuation coefficient we simulate
    adc_img_val = np.zeros((2, 2, 2, 3))
    s0_img_val = np.zeros((2, 2, 2))
    # supposidely you can get adc only with one of the 3D element of image
    # hence why I only used one b_value
    for i in range(0, 3):
        adc_img_val[:, :, :, i] = i
    s0_img_val[:, :, 0] = 0
    s0_img_val[:, :, 1] = 20
    adc = NiftiImageContainer(
        nib.Nifti1Image(
            adc_img_val,
            affine=np.eye(4),
        )
    )  # shape (x, x, x, 3)
    s0 = NiftiImageContainer(
        nib.Nifti1Image(
            s0_img_val,
            affine=np.eye(4),
        )
    )
    attenuation[:, :, :, 0] = [
        [
            [0.04978707, 0.04978707],  # 0.04978707
            [0.04978707, 0.04978707],
        ],
        [
            [0.04978707, 0.04978707],
            [0.04978707, 0.04978707],
        ],
    ]
    attenuation[:, :, :, 1] = [
        [
            [1.12535e-07, 1.12535e-07],
            [1.12535e-07, 1.12535e-07],
        ],
        [
            [1.12535e-07, 1.12535e-07],
            [1.12535e-07, 1.12535e-07],
        ],
    ]
    dwi = np.zeros((2, 2, 2, 2))
    dwi[:, :, :, 0] = [
        [
            [0, 0.995741],
            [0, 0.995741],
        ],
        [
            [0, 0.995741],  # 20*0.049787
            [0, 0.995741],
        ],
    ]
    dwi[:, :, :, 1] = [
        [
            [0, 2.2507e-06],
            [0, 2.2507e-06],
        ],
        [
            [0, 2.2507e-06],  # 20*1.12535e-7
            [0, 2.2507e-06],
        ],
    ]

    return {
        "adc": adc,
        "b_values": b_values,
        "b_vectors": b_vectors,
        "s0": s0,
        "attenuation": attenuation,
        "dwi": dwi,
    }


@pytest.fixture(name="validation_data")
def validation_data(test_data) -> dict:
    """returns validation data to test input validation"""
    im_wrong_size = NiftiImageContainer(
        nib.Nifti1Image(np.ones((3, 3, 3, 1)), np.eye(4))
    )
    nifti = nib.Nifti1Image(np.ones((3, 3, 3, 5)), np.eye(4))
    b_vectors_wrong_size = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0, 0]]

    return {
        "adc": [
            False,
            test_data["adc"],
            im_wrong_size,
            nifti,
            np.ones((3, 3, 3, 1)),
            1.0,
            4 + 4j,
            "str",
        ],
        "b_values": [
            False,
            test_data["b_values"],
            [100, 200, 300],
            [0, 100, 200, 300],
            1.0,
            4 + 4j,
            "str",
        ],
        "b_vectors": [
            False,
            test_data["b_vectors"],
            b_vectors_wrong_size,
            1.0,
            4 + 4j,
            "str",
        ],
        "s0": [
            True,
            test_data["s0"],
            im_wrong_size,
            nifti,
            np.ones((3, 3, 3, 1)),
            1.0,
            4 + 4j,
            "str",
        ],
    }


def test_value(test_data) -> None:
    """test formula"""
    dwi_signal_filter = DwiSignalFilter()
    dwi_signal_filter.add_inputs(test_data)
    dwi_signal_filter.run()

    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["attenuation"].image, test_data["attenuation"]
    )
    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["dwi"].image, test_data["dwi"]
    )
    # check metadata #TODO update metadata
    assert dwi_signal_filter.outputs["attenuation"].metadata == {
        "image_flavor": "DWI",
        "b_values": test_data["b_values"],
        "b_vectors": test_data["b_vectors"],
    }

    assert dwi_signal_filter.outputs["dwi"].metadata == {
        "image_flavor": "DWI",
        "b_values": test_data["b_values"],
        "b_vectors": test_data["b_vectors"],
    }


def test_value_no_s0(test_data_no_s0) -> None:
    """test that no s0 returns dwi = attenuation"""
    dwi_signal_filter = DwiSignalFilter()
    dwi_signal_filter.add_inputs(test_data_no_s0)
    dwi_signal_filter.run()

    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["attenuation"].image, test_data_no_s0["attenuation"]
    )
    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["dwi"].image, test_data_no_s0["dwi"]
    )


def test_value_s0_null(test_data_s0_null) -> None:
    """test that no s0 returns dwi = attenuation"""
    dwi_signal_filter = DwiSignalFilter()
    dwi_signal_filter.add_inputs(test_data_s0_null)
    dwi_signal_filter.run()

    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["attenuation"].image, test_data_s0_null["attenuation"]
    )
    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["dwi"].image, test_data_s0_null["dwi"]
    )


def test_value_complex(test_data_complex) -> None:
    """test formula"""
    dwi_signal_filter = DwiSignalFilter()
    dwi_signal_filter.add_inputs(test_data_complex)
    dwi_signal_filter.run()

    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["attenuation"].image, test_data_complex["attenuation"]
    )
    numpy.testing.assert_array_almost_equal(
        dwi_signal_filter.outputs["dwi"].image, test_data_complex["dwi"]
    )


def test_dwi_signal_filter_validate_inputs(validation_data) -> None:
    """Check a FilterInputValidationError is raised when the inputs to the
    DwiSignalFilter are incorrect or missing"""

    validate_filter_inputs(DwiSignalFilter, validation_data)
