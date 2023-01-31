"""Tests for the ADC pipeline"""

import csv
import os
import sys
from unittest.mock import patch

import nibabel as nib
import numpy.testing
import pytest

from mrimagetools.cli import main as cli
from mrimagetools.filters.bids_output_filter import BidsOutputFilter
from mrimagetools.filters.load_bids_filter import LoadBidsFilter

# pylint: disable=unused-import
from mrimagetools.filters.test_adc_quantification_filter import data_fixture
from mrimagetools.pipelines.adc_pipeline import adc_pipeline


@pytest.fixture(name="pipeline_test_data")
@pytest.mark.usefixtures("test_data")
def pipeline_test_data_fixture(test_data, tmp_path) -> dict:
    """Returns a dictionary with data for testing"""
    #  BIDS files to tmp_path
    base_filename = os.path.join(tmp_path, "dwi")
    dwi_nifti_filename = base_filename + ".nii.gz"
    dwi_json_filename = base_filename + ".json"
    nib.save(test_data["dwi"].nifti_image, dwi_nifti_filename)
    BidsOutputFilter.save_json(
        test_data["dwi"].metadata.dict(exclude_none=True),
        dwi_json_filename,
    )

    # output bval and bvec
    bval_filename = base_filename + ".bval"
    bvec_filename = base_filename + ".bvec"

    with open(bval_filename, "w", encoding="utf-8") as bval_file:
        tsv_writer = csv.writer(bval_file, delimiter=" ")
        tsv_writer.writerow(test_data["b_values"])

    with open(bvec_filename, "w", encoding="utf-8") as bvec_file:
        tsv_writer = csv.writer(bvec_file, delimiter=" ")
        [
            tsv_writer.writerow(
                [
                    test_data["b_vectors"][idx][comp]
                    for idx in range(len(test_data["b_vectors"]))
                ]
            )
            for comp in range(3)
        ]

    filenames = {
        "dwi": {
            "filename": dwi_nifti_filename,
            "sidecar": dwi_json_filename,
        },
        "bval": bval_filename,
        "bvec": bvec_filename,
    }

    return filenames


@pytest.mark.usefixtures("test_data")
def test_adc_pipeline_mock_data(pipeline_test_data, tmp_path, test_data) -> None:
    """Test the ADC pipeline with some mock data"""

    out = adc_pipeline(pipeline_test_data["dwi"]["filename"])

    assert not out["filenames"]
    for i in range(len(test_data["b_values"]) - 1):
        numpy.testing.assert_array_almost_equal(
            out["image"].image[:, :, :, i], test_data["adc"]
        )

    # run with output directory
    out = adc_pipeline(pipeline_test_data["dwi"]["filename"], tmp_path)
    assert out["filenames"] == {
        "nifti": os.path.join(tmp_path, "dwi_ADCmap.nii.gz"),
        "json": os.path.join(tmp_path, "dwi_ADCmap.json"),
    }

    # load in the image to check
    bids_loader_filter = LoadBidsFilter()
    bids_loader_filter.add_input(
        LoadBidsFilter.KEY_NIFTI_FILENAME, out["filenames"]["nifti"]
    )
    bids_loader_filter.run()
    for i in range(len(test_data["b_values"]) - 1):
        numpy.testing.assert_array_almost_equal(
            bids_loader_filter.outputs["image"].image[:, :, :, i], test_data["adc"]
        )


def test_adc_pipeline_cli(pipeline_test_data, tmp_path) -> None:
    """Tests the command line interface for the adc_pipeline"""
    testargs = [
        "neuroqa",
        "adc-quantify",
        str(pipeline_test_data["dwi"]["filename"]),
        str(tmp_path),
    ]
    with patch.object(sys, "argv", testargs):
        # should run successfully
        cli()
