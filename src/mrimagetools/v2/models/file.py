"""Pydantic models for file IO"""
from pydantic import FilePath

from mrimagetools.v2.validators.parameter_model import ParameterModel


class GroundTruthFiles(ParameterModel):
    """Ground truth files"""

    nii_file: FilePath
    json_file: FilePath

    # @classmethod
    # def __serialize__(cls) -> dict:
    #     return {
    #         "nii_file": cls.nii_file.as_posix(),
    #         "json_file": cls.json_file.as_posix(),
    #     }
