""" Constants with data file paths """
import os

# The data directory for the asldro module
DATA_DIR = os.path.dirname(os.path.realpath(__file__))
REL_DATA_DIR = os.path.dirname(os.path.relpath(__file__))


GROUND_TRUTH_DATA = {
    "hrgt_icbm_2009a_nls_3t": {
        "json_file": os.path.join(REL_DATA_DIR, "hrgt_icbm_2009a_nls_3t.json"),
        "nii_file": os.path.join(REL_DATA_DIR, "hrgt_icbm_2009a_nls_3t.nii.gz"),
    },
    "hrgt_icbm_2009a_nls_1.5t": {
        "json_file": os.path.join(REL_DATA_DIR, "hrgt_icbm_2009a_nls_1.5t.json"),
        "nii_file": os.path.join(REL_DATA_DIR, "hrgt_icbm_2009a_nls_1.5t.nii.gz"),
    },
    "qasper_3t": {
        "json_file": os.path.join(REL_DATA_DIR, "qasper", "qasper_hrgt.json"),
        "nii_file": os.path.join(REL_DATA_DIR, "qasper", "qasper_hrgt.nii.gz"),
    },
}

ASL_BIDS_SCHEMA = os.path.join(DATA_DIR, "asl_bids_validator.json")

M0SCAN_BIDS_SCHEMA = os.path.join(DATA_DIR, "m0scan_bids_validator.json")

QASPER_DATA = {
    "inlet_fuzzy_mask": os.path.join(REL_DATA_DIR, "qasper", "inlet_fuzzy_mask.nii.gz"),
    "outlet_fuzzy_mask": os.path.join(
        REL_DATA_DIR, "qasper", "outlet_fuzzy_mask.nii.gz"
    ),
    "porous_fuzzy_mask": os.path.join(
        REL_DATA_DIR, "qasper", "porous_fuzzy_mask.nii.gz"
    ),
}
