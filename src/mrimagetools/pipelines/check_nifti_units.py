"""Small python file to check for nifti units"""
# not sure why this isn't just a utils function

import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from mrimagetools.data.filepaths import GROUND_TRUTH_DATA


def check_nifti_units() -> None:
    """Checks the nifiti units are correct"""
    for key in GROUND_TRUTH_DATA.keys():
        img: Nifti1Image = nib.load(GROUND_TRUTH_DATA[key]["nii"])
        print(img.header.get_xyzt_units())
        if img.header.get_xyzt_units() is None:
            img.header.set_xyzt_units(xyz="mm", t="sec")
            nib.save(img, GROUND_TRUTH_DATA[key])


if __name__ == "__main__":
    check_nifti_units()
