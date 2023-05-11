"""Small python file to check for nifti units"""
# not sure why this isn't just a utils function


import nibabel as nib

from mrimagetools.data.filepaths import GROUND_TRUTH_DATA
from mrimagetools.utils.io import nifti_reader


def check_nifti_units() -> None:
    """Checks the nifiti units are correct"""
    for _, value in GROUND_TRUTH_DATA.items():
        img = nifti_reader(value["nii"])
        print(img.header.get_xyzt_units())
        if img.header.get_xyzt_units() is None:
            img.header.set_xyzt_units(xyz="mm", t="sec")
            nib.nifti2.save(img, value)


if __name__ == "__main__":
    check_nifti_units()
