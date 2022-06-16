""" filepaths.py tests """

import os

from mrimagetools.data.filepaths import GROUND_TRUTH_DATA, QASPER_DATA


def test_file_paths_exist() -> None:
    """Check the ground truth file paths exist and are files"""
    for gt_value in GROUND_TRUTH_DATA.values():
        assert os.path.isfile(gt_value["json"])
        assert os.path.isfile(gt_value["nii"])

    for data_value in QASPER_DATA.values():
        assert os.path.isfile(data_value)
