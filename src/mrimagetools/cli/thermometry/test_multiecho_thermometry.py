"""Tests for multiecho_thermometry.py CLI."""

from operator import mul
import os

import uuid
from typing import Union, Callable, Tuple, Any, List, cast
import pdb
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from mrimagetools.cli.thermometry.multiecho_thermometry import (
    multiecho_thermometry,
    remove_suffix,
    app,
)
from mrimagetools.filters.test_multiecho_thermometry_filter import (
    thermometry_test_volume_factory,
)
from mrimagetools.v2.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.filters.multiecho_thermometry_filter import GAMMA_H


def test_remove_suffix() -> None:
    """Test the remove_suffix function."""
    assert remove_suffix(Path("file.nii.gz"), ".gz") == Path("file.nii")
    assert remove_suffix(Path("file.nii"), ".nii") == Path("file")
    assert remove_suffix(Path("file.nii"), ".txt") == Path("file.nii")
    assert remove_suffix(Path("file"), ".nii") == Path("file")


@pytest.fixture(name="thermometry_test_data")
@pytest.mark.usefixtures("thermometry_test_volume")
def thermometry_test_data_fixture(
    thermometry_test_volume_factory: Callable, tmp_path: Path
) -> Callable:
    """ "Fixture that provides test data for thermometry tests."""

    def _make_thermometry_test_data_fixture(
        echo_times: np.ndarray = np.linspace(0.001, 0.024, 24),
        sidecar_fields: List[str] = [
            "ImagingFrequency",
            "MagneticFieldStrength",
            "SomeOtherField",
        ],
    ) -> Tuple[Any, ...]:
        """Returns a tuple with test data for testing."""
        # unpack the test fixture tuple
        (
            magnetic_field_tesla,
            true_temperature_map,
            segmentation_image,
            multiecho_image,
            noisy_multiecho_image,
            echo_times,
            region_id,
            region_temperature_celsius,
        ) = thermometry_test_volume_factory(echo_times)

        output_path = tmp_path / str(uuid.uuid4())
        os.makedirs(output_path, exist_ok=True)

        full_json_sidecar = {
            "ImagingFrequency": magnetic_field_tesla * GAMMA_H / 1e6,  # in MHz
            "MagneticFieldStrength": magnetic_field_tesla,  # in Tesla
            "SomeOtherField": 123.456,
        }
        # only keep the requested fields in the sidecar
        json_sidecar = {
            k: full_json_sidecar[k] for k in sidecar_fields if k in full_json_sidecar
        }

        # save the test data to temporary files
        multiecho_image_file = output_path / "multiecho_image.nii.gz"
        nib.nifti1.save(multiecho_image.nifti_image, multiecho_image_file)

        # truncated version with one less echo
        multiecho_image_truncated_file = (
            output_path / "multiecho_image_truncated.nii.gz"
        )
        nib.nifti1.save(
            nib.Nifti1Image(
                multiecho_image.image[..., :-1],
                multiecho_image.affine,
            ),
            multiecho_image_truncated_file,
        )

        # correct number of echoes, wrong shape
        multiecho_image_wrong_shape_file = (
            output_path / "multiecho_image_wrong_shape.nii.gz"
        )
        nib.nifti1.save(
            nib.Nifti1Image(
                multiecho_image.image[:-1, :, :, :],
                multiecho_image.affine,
            ),
            multiecho_image_wrong_shape_file,
        )

        json_sidecar_file = output_path / "multiecho_image.json"
        BidsOutputFilter.save_json(json_sidecar, str(json_sidecar_file))
        # also save json sidecar file for truncated and wrong shape images
        BidsOutputFilter.save_json(
            json_sidecar, str(output_path / "multiecho_image_truncated.json")
        )
        BidsOutputFilter.save_json(
            json_sidecar, str(output_path / "multiecho_image_wrong_shape.json")
        )

        segmentation_image_file = output_path / "segmentation_image.nii.gz"
        nib.nifti1.save(segmentation_image.nifti_image, segmentation_image_file)

        # save echo times to temporary files
        echo_times_file = output_path / "echo_times.txt"
        np.savetxt(echo_times_file, echo_times)

        echo_times_truncated_file = output_path / "echo_times_truncated.txt"
        np.savetxt(echo_times_truncated_file, echo_times[:-1])

        return (
            magnetic_field_tesla,
            echo_times,
            echo_times_file,
            echo_times_truncated_file,
            multiecho_image_file,
            json_sidecar_file,
            segmentation_image_file,
            true_temperature_map,
            multiecho_image_truncated_file,
            multiecho_image_wrong_shape_file,
        )

    return _make_thermometry_test_data_fixture


@pytest.mark.usefixtures("thermometry_test_data")
@pytest.mark.parametrize("method", ["regionwise", "voxelwise", "regionwise_bootstrap"])
def test_multiecho_thermometry_cli_basic(
    thermometry_test_data: Callable, tmp_path: Path, method: str
) -> None:
    """Basic test of the multiecho_thermometry CLI function.

    Uses a single multiecho volume, semgmentation mask, and echo times file.
    Method is 'regionwise'.
    Checks that output files are created and have reasonable values.

    """
    (
        _,
        _,
        echo_times_file,
        _,
        multiecho_image_file,
        _,
        segmentation_image_file,
        true_temperature_map,
        _,
        _,
    ) = thermometry_test_data()

    multiecho_thermometry(
        multiecho_nifti_files=[multiecho_image_file],
        segmentation_nifti_file=segmentation_image_file,
        echo_times_files=[echo_times_file],
        method=method,
        n_bootstrap=10,  # low number for speed
        output_dir=tmp_path,
    )

    # Check if output files were created
    filename_stem = remove_suffix(multiecho_image_file, ".gz").stem
    output_temperature_map_file = tmp_path / f"{filename_stem}_temperature_map.nii.gz"
    output_report_file = tmp_path / f"{filename_stem}_report.json"
    assert (
        output_temperature_map_file.exists()
    ), "Output temperature file was not created."
    assert output_report_file.exists(), "Output report file was not created."

    # Load the output image and check its shape
    output_img = cast(nib.Nifti1Image, nib.load(output_temperature_map_file))
    output_data = output_img.get_fdata()

    # Check that the output temperature map is close to the true temperature map
    # within a reasonable tolerance
    assert (
        output_data.shape == true_temperature_map.shape
    ), "Output temperature map has incorrect shape."
    np.testing.assert_array_almost_equal(
        output_data,
        true_temperature_map,
        err_msg="Output temperature map differs significantly from true temperature map.",
    )


def test_multiecho_thermometry_cli_validation_fails(
    thermometry_test_data: Callable, tmp_path: Path
) -> None:
    """Test that the CLI fails when the input files are invalid."""
    (
        magnetic_field_tesla,
        echo_times,
        echo_times_file,
        echo_times_truncated_file,
        multiecho_image_file,
        json_sidecar_file,
        segmentation_image_file,
        true_temperature_map,
        multiecho_image_truncated_file,
        multiecho_image_wrong_shape_file,
    ) = thermometry_test_data()

    # invalid analysis_method
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[multiecho_image_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_file],
            method="invalid_method",
            output_dir=tmp_path,
        )

    # invalid number of echo_times files
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[multiecho_image_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_file, echo_times_file],
            method="regionwise",
            output_dir=tmp_path,
        )

    # multiecho image shapes do not match
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[multiecho_image_file, segmentation_image_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_file, echo_times_file],
            method="regionwise",
            output_dir=tmp_path,
        )

    # multiecho image is not 4D
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[segmentation_image_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_file],
            method="regionwise",
            output_dir=tmp_path,
        )

    # segmentation image and multiecho image shapes do not match
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[multiecho_image_wrong_shape_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_file],
            method="regionwise",
            output_dir=tmp_path,
        )

    # missing magnetic field information in json sidecar files
    (
        magnetic_field_tesla,
        echo_times,
        echo_times_file,
        echo_times_truncated_file,
        multiecho_image_file,
        json_sidecar_file,
        segmentation_image_file,
        true_temperature_map,
        multiecho_image_truncated_file,
        multiecho_image_wrong_shape_file,
    ) = thermometry_test_data(sidecar_fields=["SomeOtherField"])
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[multiecho_image_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_file],
            method="regionwise",
            output_dir=tmp_path,
        )

    # only magnetic field strength is provided
    (
        magnetic_field_tesla,
        echo_times,
        echo_times_file,
        echo_times_truncated_file,
        multiecho_image_file,
        json_sidecar_file,
        segmentation_image_file,
        true_temperature_map,
        multiecho_image_truncated_file,
        multiecho_image_wrong_shape_file,
    ) = thermometry_test_data(sidecar_fields=["MagneticFieldStrength"])

    # should run without error
    multiecho_thermometry(
        multiecho_nifti_files=[multiecho_image_file],
        segmentation_nifti_file=segmentation_image_file,
        echo_times_files=[echo_times_file],
        method="regionwise",
        output_dir=tmp_path,
    )

    # echo times file has wrong number of entries
    with pytest.raises(typer.Exit) as e:
        multiecho_thermometry(
            multiecho_nifti_files=[multiecho_image_file],
            segmentation_nifti_file=segmentation_image_file,
            echo_times_files=[echo_times_truncated_file],
            method="regionwise",
            output_dir=tmp_path,
        )


def test_multiecho_thermometry_cli_multiple_inputs(
    thermometry_test_data: Callable, tmp_path: Path
) -> None:
    """Test the multiecho_thermometry CLI function with multiple multiecho images."""

    echo_times_list = [
        np.linspace(p[0], p[1], p[2])
        for p in [
            (0.002, 0.012, 6),
            (0.003, 0.013, 6),
            (0.014, 0.024, 6),
            (0.015, 0.025, 6),
        ]
    ]
    input_data = [thermometry_test_data(echo_times=et) for et in echo_times_list]
    multi_echo_files = [m[4] for m in input_data]
    segmentation_file = input_data[0][6]
    echo_times_files = [m[2] for m in input_data]

    # should run without error
    multiecho_thermometry(
        multiecho_nifti_files=multi_echo_files,
        segmentation_nifti_file=segmentation_file,
        echo_times_files=echo_times_files,
        method="regionwise",
        output_dir=tmp_path,
    )


def test_multiecho_thermometry_typer_app(
    thermometry_test_data: Callable, tmp_path: Path
) -> None:
    """Test that the typer app runs without error."""
    (
        _,
        _,
        echo_times_file,
        _,
        multiecho_image_file,
        _,
        segmentation_image_file,
        _,
        _,
        _,
    ) = thermometry_test_data()

    runner = CliRunner()
    args = []
    args = [str(f) for f in [multiecho_image_file]]
    args.extend(["--segmentation", f"{segmentation_image_file}"])
    args.extend(
        [item for file in [echo_times_file] for item in ["--echotimes", str(file)]]
    )
    args.extend(["--method", "regionwise"])
    args.extend(["--nb", "10"])  # low number for speed
    args.extend(["--output-dir", f"{tmp_path}"])

    pdb.set_trace()
    result = runner.invoke(app, args)
    # assert result.exit_code == 0
