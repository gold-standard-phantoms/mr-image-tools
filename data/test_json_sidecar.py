"""Regression tests for JSON sidecars"""
import json
import os
from pathlib import Path

from mrimagetools.v2.containers.image_metadata import ImageMetadata


def test_dataset_is_non_empty_folder(dataset: Path) -> None:
    """DICOM Dataset Loading Regression Tests."""
    assert dataset.exists() and dataset.is_dir(), f"{dataset}: Invalid Path"
    assert any(dataset.iterdir()), f"{dataset}: Empty Dataset"


def test_json_imagemetadata_loading(dataset: Path) -> None:
    """Try to load all the JSON sidecar files as ImageMetadata"""
    json_files = [file for file in os.listdir(dataset) if file.endswith(".json")]
    print(f"Parsing {len(json_files)} json files")

    for file in json_files:
        print(f"Parsing: {file}")
        with open(os.path.join(dataset, file), encoding="utf-8") as f:
            ImageMetadata.from_bids(json.load(f))
