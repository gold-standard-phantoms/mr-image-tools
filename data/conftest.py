"""Pytest config for regression tests"""

import os
from pathlib import Path
from typing import cast

from pytest import Metafunc, Parser

DEFAULT_DATASET_MARKERS = ["JSON-FIXTURES"]


def get_datasets(markers: set[str], data_dir: str) -> list[Path]:
    """Determine the datasets to be run based on the presence of marker files"""
    return [
        Path(path)
        for (path, dirs, files) in os.walk(data_dir)
        if any(marker in files or marker in dirs for marker in markers)
    ]


def pytest_addoption(parser: Parser) -> None:
    """Pytest hook to add command line options"""

    parser.addoption("--regression-tests", action="store_true", default=False)
    parser.addoption("--dataset-marker", action="append")
    parser.addoption("--data-dir", action="store", default="data")


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Hook to autodiscover datasets for regression testing

    Regression tests are enabled by discovering all datasets in the
    data directory (default: data), and parametrizing the 'dataset'
    fixture with those datasets.

    Datasets are determined by the presense of a marker file in the folder.
    The marker files are added via the --dataset-marker option.

    When the '--regression-tests' option is specified, the datasets are
    populated. Otherwise, an empty list is used for the parametrization,
    resulting in the tests being skipped.

    Note: In the future, the tests will be xfailed instead. See:
        - https://docs.pytest.org/en/7.1.x/reference/reference.html#confval-empty_parameter_set_mark
        - https://github.com/pytest-dev/pytest/issues/3155)

    """
    if "dataset" not in metafunc.fixturenames:
        return

    getopt = metafunc.config.getoption

    run_regression_tests = getopt("regression_tests")
    datasets = (
        []
        if not run_regression_tests
        else get_datasets(
            markers=set(getopt("dataset_marker") or DEFAULT_DATASET_MARKERS),
            data_dir=getopt("data_dir"),
        )
    )
    print(f"Datasets found: {len(datasets)}")
    metafunc.parametrize(
        argnames="dataset",
        argvalues=datasets,
        ids=(
            (lambda x: cast(Path, x).as_posix())
            if run_regression_tests
            else ["<dataset>"]
        ),
    )
