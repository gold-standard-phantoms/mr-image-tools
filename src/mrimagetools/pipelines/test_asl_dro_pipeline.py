""" Example tests - only run when --runslow is passed to pytest """

from collections.abc import Sequence

import numpy as np
import numpy.testing
import pytest

from mrimagetools.containers.image_metadata import ImageMetadata
from mrimagetools.filters.asl_quantification_filter import AslQuantificationFilter
from mrimagetools.pipelines.asl_dro_pipeline import run_full_asl_dro_pipeline
from mrimagetools.validators.user_parameter_input import (
    ARRAY_PARAMS,
    get_example_input_params,
)


@pytest.mark.slow
def test_run_default_pipeline() -> None:
    """Runs the full ASL DRO pipeline"""
    droout = run_full_asl_dro_pipeline()

    # check the segmentation_mask resampled ground truth
    seg_label_index = [
        idx
        for idx, im in enumerate(droout.dro_output)
        if im.metadata.quantity == "seg_label"
    ]
    gt_seg_label = droout.dro_output[seg_label_index[0]]

    # interpolation is nearest for the default so no new values should be created, check the
    # unique values against the original ground truth
    numpy.testing.assert_array_equal(
        np.unique(gt_seg_label.image), np.unique(droout.hrgt.images["seg_label"].image)
    )


@pytest.mark.slow
def test_run_extended_pipeline() -> None:
    """Runs the full ASL DRO pipeline with modified input parameters"""
    input_params = get_example_input_params()

    asl_params = input_params["image_series"][0]["series_parameters"]

    asl_params["asl_context"] = "m0scan control label control label control label"
    # remove the array parameters so they are autogenerates
    for param in ARRAY_PARAMS:
        asl_params.pop(param)

    # remove the ground truth and structural image series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]
    run_full_asl_dro_pipeline(input_params=input_params)


@pytest.mark.slow
def test_run_asl_pipeline_multiphase() -> None:
    """Runs the full ASL DRO pipeline for muliphase ASL
    with the SNR=zero to check this works"""
    input_params = get_example_input_params()
    # remove the ground truth and structural image series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]
    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 0.0
    input_params["image_series"][0]["series_parameters"]["signal_time"] = [
        1.0,
        1.25,
        1.5,
    ]

    run_full_asl_dro_pipeline(input_params=input_params)


@pytest.mark.slow
def test_run_asl_pipeline_qasper_hrgt() -> None:
    """Runs the ASL pipeline using the qasper ground truth"""
    input_params = get_example_input_params()

    input_params["global_configuration"]["ground_truth"] = "qasper_3t"

    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 0.0
    # remove the ground truth and structural image series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]

    run_full_asl_dro_pipeline(input_params=input_params)


@pytest.mark.slow
def test_run_asl_pipeline_whitepaper() -> None:
    """Runs the ASL pipeline with gkm_model set to "whitepaper"
    Checks the results using the AslQuantificationFilter"""

    input_params = get_example_input_params()

    input_params["image_series"][0]["series_parameters"]["gkm_model"] = "whitepaper"
    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 0.0
    # set the TR of the M0 to very long so there's no real effect (important for
    # comparing values)
    input_params["image_series"][0]["series_parameters"]["repetition_time"] = [
        1.0e6,
        5.0,
        5.0,
    ]
    input_params["image_series"][0]["series_parameters"]["acq_matrix"] = input_params[
        "image_series"
    ][2]["series_parameters"]["acq_matrix"]
    # remove the ground truth and structural image series

    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]

    out = run_full_asl_dro_pipeline(input_params=input_params)

    asl_source = out.dro_output[0]
    assert asl_source.metadata.gkm_model == "whitepaper"
    asl_data = {}  # empty dictionary for asl data
    asl_context = asl_source.metadata.asl_context
    assert asl_context is not None
    for key in ["control", "label", "m0scan"]:
        volume_indices = [i for (i, val) in enumerate(asl_context) if val == key]
        if volume_indices is not None:
            asl_data[key] = asl_source.clone()
            asl_data[key].image = np.squeeze(asl_source.image[:, :, :, volume_indices])
            asl_data[key].metadata.asl_context = key * len(volume_indices)  # type: ignore

        # adjust metadata lists to correspond to one value per volume
        new_metadata = {
            metadata_key: [
                metadata_val[i]
                for i in volume_indices
                if isinstance(metadata_val, Sequence)
            ]
            for metadata_key, metadata_val in asl_data[key].metadata.__fields__.items()
            if (
                (metadata_key not in ["voxel_size"])
                and (
                    (len(metadata_val) == len(asl_context))
                    if isinstance(metadata_val, list)
                    else False
                )
            )
        }

        asl_data[key].metadata = ImageMetadata(
            **{**asl_data[key].metadata.dict(exclude_none=True), **new_metadata}
        )
    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(
        {
            key: getattr(asl_data["control"].metadata, key)
            for key in asl_data["control"].metadata.__fields__
            if key
            in [
                AslQuantificationFilter.KEY_LABEL_TYPE,
                AslQuantificationFilter.KEY_LABEL_DURATION,
                AslQuantificationFilter.KEY_POST_LABEL_DELAY,
                AslQuantificationFilter.KEY_LABEL_EFFICIENCY,
                AslQuantificationFilter.KEY_T1_ARTERIAL_BLOOD,
                AslQuantificationFilter.KEY_LAMBDA_BLOOD_BRAIN,
            ]
        }
    )
    asl_quantification_filter.add_inputs(
        asl_data, io_map={"m0scan": "m0", "control": "control", "label": "label"}
    )
    asl_quantification_filter.add_input("gkm_model", "whitepaper")

    asl_quantification_filter.run()

    # compare the quantified perfusion_rate with the ground truth to 12 d.p.
    np.testing.assert_array_almost_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        out.hrgt.images["perfusion_rate"].image,
        12,
    )
