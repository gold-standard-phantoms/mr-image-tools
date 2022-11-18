"""Command Line Interface"""

import argparse
import json
import logging
import os
import sys
from enum import Enum
from typing import List, Optional

from pydantic import DirectoryPath, FilePath

from mrimagetools.data.filepaths import GROUND_TRUTH_DATA
from mrimagetools.pipelines.adc_pipeline import adc_pipeline
from mrimagetools.pipelines.asl_dro_pipeline import (
    PipelineReturnVariables,
    run_full_asl_dro_pipeline,
)
from mrimagetools.pipelines.generate_ground_truth import generate_hrgt
from mrimagetools.pipelines.mtr_pipeline import mtr_pipeline
from mrimagetools.validators.parameter_model import ParameterModel

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s %(message)s", level=logging.INFO
)  # Set the log level to INFO
logger = logging.getLogger(__name__)


class HrgtType:  # pylint: disable=too-few-public-methods
    """A HRGT string checker.
    Will determine if the input is a valid HRGT string label.
    """

    def __call__(self, hrgt_label: str) -> str:
        """
        Do the checking
        :param hrgt_label: the HRGT label string
        """
        if hrgt_label not in GROUND_TRUTH_DATA:
            raise argparse.ArgumentTypeError(
                f"{hrgt_label} is not valid, must be one of"
                f" {', '.join(GROUND_TRUTH_DATA.keys())}"
            )
        return hrgt_label


class DirType:  # pylint: disable=too-few-public-methods
    """A directory checker.
    Will determine if the input is a directory and
    optionally, whether it exists
    """

    def __init__(self, should_exist: bool = False) -> None:
        """
        :param should_exist: does the directory have to exist
        """
        self.should_exist: bool = should_exist

    def __call__(self, path: str) -> str:
        """
        Do the checking
        :param path: the path to the directory
        """
        # Always check the file is a directory

        if self.should_exist:
            # Check whether the file exists
            if not os.path.exists(path):
                raise argparse.ArgumentTypeError(f"{path} does not exist")
            if not os.path.isdir(path):
                raise argparse.ArgumentTypeError(f"{path} is not a directory")
        return path


class FileType:  # pylint: disable=too-few-public-methods
    """A file checker.
    Will determine if the input is a valid file name (or path)
    and optionally, whether it has a particular extension and/or exists
    """

    def __init__(
        self, extensions: Optional[list[str]] = None, should_exist: bool = False
    ) -> None:
        """
        :param extensions: a list of allowed file extensions.
        :param should_exist: does the file have to exist
        """
        if not isinstance(extensions, list) and extensions is not None:
            raise TypeError("extensions should be a list of string extensions")

        if extensions is not None:
            for extension in extensions:
                if not isinstance(extension, str):
                    raise TypeError("All extensions must be strings")

        self.extensions: list[str] = []
        if extensions is not None:
            # Strip any proceeding dots
            self.extensions = [
                extension if not extension.startswith(".") else extension[1:]
                for extension in extensions
            ]
        self.should_exist: bool = should_exist

    def __call__(self, path: str) -> str:
        """
        Do the checkstructing
        :param path: the path to the file
        """
        # Always check the file is not a directory
        if os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"{path} is a directory")

        if self.should_exist:
            # Check whether the file exists
            if not os.path.exists(path):
                raise argparse.ArgumentTypeError(f"{path} does not exist")

        if self.extensions:
            valid_extension = False
            for extension in self.extensions:
                if path.lower().endswith(extension.lower()):
                    valid_extension = True
            if not valid_extension:
                raise argparse.ArgumentTypeError(
                    f"{path} is does not have a valid extension "
                    f"(from {', '.join(self.extensions)})"
                )
        return path


def mtr_quantify(args: argparse.Namespace) -> None:
    """Parses the 'mtr-quantify' subcommand. Must have a:
    * 'nosat_nifti_path', which is the path to the image without
      bound pool saturation.
    * 'sat_nifti_path', which is the path to the image with
      bound pool saturation
    * 'output_dir', which is the path to a directory to save the output
      files to.
    """

    mtr_pipeline(args.sat, args.nosat, args.outputdir)


def adc_quantify(args: argparse.Namespace) -> None:
    """Parses the 'adc-quantify' subcommand. Must have a:
    * 'dwi_image_path', which is the path to the dwi image.
    * 'output_dir', which is the path to a directory to save the output
      files to.
    """
    adc_pipeline(args.dwi, args.outputdir)


class HrgtParams(ParameterModel):
    """CLI input paramaters for the HRGT generation"""

    hrgt_params_path: FilePath
    seg_mask_path: FilePath
    output_dir_path: DirectoryPath

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Construct the HRGT parameter model from the input arguments"""
        return cls(
            hrgt_params_path=args.hrgt_params_path,
            seg_mask_path=args.seg_mask_path,
            output_dir_path=args.output_dir,
        )


def create_hrgt(args: argparse.Namespace) -> None:
    """Parses the 'create-hrgt' subcommand. Must have a:
    * 'seg_mask_path' which is the path of the segmentation mask image
    * 'hrgt_params_path', which is the path of the hrgt generation parameters
    * 'output_dir', which is the directory to output to.
    """
    hrgt_params = HrgtParams.from_args(args)
    generate_hrgt(
        hrgt_params_filename=str(hrgt_params.hrgt_params_path),
        seg_mask_filename=str(hrgt_params.seg_mask_path),
        schema_name="generate_hrgt_params",
        output_dir=str(hrgt_params.output_dir_path),
    )


class Modality(str, Enum):
    """An MRI modality used in the selection of DRO types."""

    DWI = "dwi"
    ASL = "asl"


def generate(args: argparse.Namespace) -> PipelineReturnVariables:
    """Parses the 'generate' subcommand.
    :param args: the command line arguments. May optionally contain
    a 'params' value, which will be JSON filename to load for the model
    inputs (will use default if not present). Must contain 'output' which
    will contain the filename of a .zip or .tar.gz archive."""
    params = None
    if args.params is not None:
        with open(args.params, encoding="utf-8") as json_file:
            params = json.load(json_file)

    if args.modality == Modality.ASL:
        return run_full_asl_dro_pipeline(
            input_params=params, output_filename=args.output
        )
    raise ValueError(f"Modality {args.modality} unknown")


def main() -> None:
    """Main function for the Command Line Interface"""

    parser = argparse.ArgumentParser(
        description=(
            "A set of tools for performing the NeuroQA analysis. For help using the"
            " commands, use the -h flag, for example: neuroqa mtr-quantify -h"
        ),
        epilog="Enjoy the program! :)",
    )

    parser.set_defaults(func=lambda _: parser.print_help())

    # Generate subparser
    subparsers = parser.add_subparsers(
        title="command", help="Subcommand to run", dest="command"
    )

    # MTR Pipeline
    mtr_quantify_parser = subparsers.add_parser(
        name="mtr-quantify",
        description=(
            "Calculates the magnetisation transfer ratio based on"
            "two images, one with and one without bound pool saturation."
        ),
    )
    mtr_quantify_parser.add_argument(
        "sat",
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help=(
            "The path to the input image with bound pool saturation, or a"
            "mutli-volume image with both the saturated and unsaturated images. This"
            "should be accompanied by a corresponding *.json file in BIDS format"
        ),
    )

    mtr_quantify_parser.add_argument(
        "--nosat",
        required=False,
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help=(
            "The path to the input image without bound pool saturation. If not"
            "supplied then it is assumed the sat nifti is multi-volume containing"
            "both. This should be accompanied by a corresponding *.json file in"
            "BIDS format"
        ),
    )

    mtr_quantify_parser.add_argument(
        "outputdir",
        type=DirType(should_exist=True),
        help=(
            "The directory to output to. Must exist. Will overwrite any existing files"
            " with the same namesMagnetisation transfer ratio (MTR) maps and the"
            " accompanying JSON sidecarwill be saved with the same filename as the"
            " saturated NIFTI, with '_MTRmap'appended"
        ),
    )

    mtr_quantify_parser.set_defaults(func=mtr_quantify)

    # ADC Quantify
    adc_quantify_parser = subparsers.add_parser(
        name="adc-quantify",
        description=(
            "Calculates the apparent diffusion coefficient based on"
            "a set of diffusion weighted images."
        ),
    )

    adc_quantify_parser.add_argument(
        "dwi",
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help=(
            "The path to the 4D NIFTI file of diffusion weighted images."
            "There should be at least two volumes, one acquired with a b-value of 0."
            "It is assumed that there are also corresponding BIDS sidecar (*.json),"
            "bvec (*.bvec), and bval (*.bval ) files with the same base filenames"
            "as the NIFTI."
        ),
    )

    adc_quantify_parser.add_argument(
        "outputdir",
        type=DirType(should_exist=True),
        help=(
            "The directory to output to. Must exist. Will overwrite any existing files"
            " with the same namesApparent Diffusion Coefficient (ADC) maps and the"
            " accompanying JSON sidecarwill be saved with the same filename as the"
            " input DWI NIFTI, with '_ADCmap'appended"
        ),
    )

    adc_quantify_parser.set_defaults(func=adc_quantify)

    # Create HRGT
    create_hrgt_parser = subparsers.add_parser(
        name="create-hrgt",
        description="""Generates a HRGT based on input segmentation
        masks and values to be assigned for each quantity and region type""",
    )
    create_hrgt_parser.add_argument(
        "seg_mask_path",
        type=FileType(extensions=[".nii", ".nii.gz"], should_exist=True),
        help=(
            "The path to the segmentation mask image. Must be a NIFTI or gzipped NIFTI"
            " with extension .nii or .nii.gz. The image data can either be integer, or"
            " floatingpoint. For floating point data voxel values will be rounded to"
            " the nearest integer whendefining which region type is in a voxel."
        ),
    )
    create_hrgt_parser.add_argument(
        "hrgt_params_path",
        type=FileType(extensions=["json"], should_exist=True),
        help=(
            "The path to the parameter file containing values to assign to each region."
            " Mustbe a .json."
        ),
    )
    create_hrgt_parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help=(
            "The directory to output to. Will create 'hrgt.nii.gz' and 'hrgt.json'"
            " files.Must exist. Will overwrite any existing files with the same names."
        ),
    )
    create_hrgt_parser.set_defaults(func=create_hrgt)

    generate_parser = subparsers.add_parser(
        name="generate",
        description="Generate a Digital Reference Object (DRO)",
    )

    generate_parser.add_argument(
        "modality",
        type=Modality,
        help=(
            "The DRO modality. Current types are 'asl' for an Arterial Spin Labelling"
            " (ASL) DRO, and 'dwi' for a Diffusion Weighted Image (DWI) DRO"
        ),
    )

    generate_parser.add_argument(
        "--params",
        type=FileType(extensions=["json"], should_exist=True),
        help=(
            "A path to a JSON file containing the input parameters, otherwise the"
            " defaults (white paper) are used"
        ),
    )
    generate_parser.add_argument(
        "output",
        type=FileType(extensions=["zip", "tar.gz"]),
        help=(
            "The output filename (optionally with path). Must be an archive type"
            " (zip/tar.gz). Will overwrite an existing file."
        ),
    )
    generate_parser.set_defaults(func=generate)
    args = parser.parse_args()
    args.func(args)  # call the default function


if __name__ == "__main__":
    main()
