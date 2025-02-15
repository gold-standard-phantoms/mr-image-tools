""" filepaths.py tests """

import os

from mrimagetools.v2.data.filepaths import GROUND_TRUTH_DATA, QASPER_DATA


def test_file_paths_exist() -> None:
    """Check the ground truth file paths exist and are files"""
    for gt_value in GROUND_TRUTH_DATA.values():
        assert os.path.isfile(gt_value["json_file"])
        assert os.path.isfile(gt_value["nii_file"])

    for data_value in QASPER_DATA.values():
        assert os.path.isfile(data_value)
