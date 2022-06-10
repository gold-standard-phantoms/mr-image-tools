"""Tests for the MTR pipeline"""

import os
import sys
from unittest.mock import patch

import numpy as np
import numpy.testing
import pytest
from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.utils.general import splitext

from neuroqa.cli import main as cli
from neuroqa.filters.load_bids_filter import LoadBidsFilter
from neuroqa.filters.test_mtr_quantification_filter import test_data_fixture
from neuroqa.pipelines.mtr_pipeline import mtr_pipeline


@pytest.fixture(name="pipeline_test_data")
@pytest.mark.usefixtures("test_data")
def pipeline_test_data_fixture(test_data, tmp_path):
    """Returns a dictionary with data for testing"""
    filenames = {}
    # use the bids output filter to save BIDS files to tmp_path
    image_combined = test_data["image_nosat"].clone()
    image_combined.image = np.stack(
        [test_data["image_nosat"].image, test_data["image_sat"].image], axis=3
    )

    for image in ["image_nosat", "image_sat", "image_combined"]:
        bids_output_filter = BidsOutputFilter()

        bids_output_filter.add_input(
            "image", image_combined if image == "image_combined" else test_data[image]
        )
        bids_output_filter.add_input("output_directory", str(tmp_path))
        bids_output_filter.add_input("filename_prefix", image)
        bids_output_filter.run()
        filenames[image] = bids_output_filter.outputs

    return filenames


@pytest.mark.usefixtures("test_data")
def test_mtr_pipeline_mock_data_separate(pipeline_test_data, tmp_path, test_data):
    """Test the MTR pipeline with some mock data"""

    out = mtr_pipeline(
        pipeline_test_data["image_sat"]["filename"][0],
        pipeline_test_data["image_nosat"]["filename"][0],
    )

    assert out["filenames"] == {}
    numpy.testing.assert_array_almost_equal(
        out["image"].image, test_data["image_mtr"].image
    )

    out = mtr_pipeline(
        pipeline_test_data["image_sat"]["filename"][0],
        pipeline_test_data["image_nosat"]["filename"][0],
        tmp_path,
    )

    assert out["filenames"] == {
        "nifti": os.path.join(
            tmp_path,
            os.path.split(splitext(pipeline_test_data["image_sat"]["filename"][0])[0])[
                1
            ]
            + "_MTRmap.nii.gz",
        ),
        "json": os.path.join(
            tmp_path,
            os.path.split(splitext(pipeline_test_data["image_sat"]["filename"][0])[0])[
                1
            ]
            + "_MTRmap.json",
        ),
    }
    # load in the image to check
    bids_loader_filter = LoadBidsFilter()
    bids_loader_filter.add_input(
        LoadBidsFilter.KEY_NIFTI_FILENAME, out["filenames"]["nifti"]
    )
    bids_loader_filter.run()

    numpy.testing.assert_array_almost_equal(
        bids_loader_filter.outputs[LoadBidsFilter.KEY_IMAGE].image,
        test_data["image_mtr"].image,
    )


@pytest.mark.usefixtures("test_data")
def test_mtr_pipeline_mock_data_combined(pipeline_test_data, tmp_path, test_data):
    """Test the MTR pipeline with some mock data"""

    out = mtr_pipeline(
        pipeline_test_data["image_combined"]["filename"][0],
    )

    assert out["filenames"] == {}
    numpy.testing.assert_array_almost_equal(
        out["image"].image, np.expand_dims(test_data["image_mtr"].image, 3)
    )

    out = mtr_pipeline(
        pipeline_test_data["image_combined"]["filename"][0],
        output_dir=tmp_path,
    )


def test_mtr_pipeline_cli_separate_files(pipeline_test_data, tmp_path):
    """Tests the command line interface for the mtr_pipeline"""

    testargs = [
        "neuroqa",
        "mtr-quantify",
        "--nosat",
        str(pipeline_test_data["image_nosat"]["filename"][0]),
        str(pipeline_test_data["image_sat"]["filename"][0]),
        str(tmp_path),
    ]

    with patch.object(sys, "argv", testargs):
        # should run successfully
        cli()


def test_mtr_pipeline_cli_combined_file(pipeline_test_data, tmp_path):
    """Tests the command line interface for the mtr_pipeline"""

    testargs = [
        "neuroqa",
        "mtr-quantify",
        str(pipeline_test_data["image_combined"]["filename"][0]),
        str(tmp_path),
    ]

    with patch.object(sys, "argv", testargs):
        # should run successfully
        cli()
