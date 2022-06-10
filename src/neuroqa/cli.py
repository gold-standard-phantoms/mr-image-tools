"""Command Line Interface"""

import argparse
import logging
import sys

from asldro.cli import DirType, FileType

from neuroqa.pipelines.adc_pipeline import adc_pipeline
from neuroqa.pipelines.mtr_pipeline import mtr_pipeline

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s %(message)s", level=logging.INFO
)  # Set the log level to INFO
logger = logging.getLogger(__name__)


def mtr_quantify(args):
    """Parses the 'mtr-quantify' subcommand. Must have a:
    * 'nosat_nifti_path', which is the path to the image without
      bound pool saturation.
    * 'sat_nifti_path', which is the path to the image with
      bound pool saturation
    * 'output_dir', which is the path to a directory to save the output
      files to.
    """

    mtr_pipeline(args.sat, args.nosat, args.outputdir)


def adc_quantify(args):
    """Parses the 'adc-quantify' subcommand. Must have a:
    * 'dwi_image_path', which is the path to the dwi image.
    * 'output_dir', which is the path to a directory to save the output
      files to.
    """
    adc_pipeline(args.dwi, args.outputdir)


def main():
    """Main function for the Command Line Interface"""

    parser = argparse.ArgumentParser(
        description="""A set of tools for performing the NeuroQA analysis.
        For help using the commands, use the -h flag, for example:
        neuroqa mtr-quantify -h""",
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
        description="Calculates the magnetisation transfer ratio based on"
        "two images, one with and one without bound pool saturation.",
    )
    mtr_quantify_parser.add_argument(
        "sat",
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help="The path to the input image with bound pool saturation, or a"
        "mutli-volume image with both the saturated and unsaturated images. This"
        "should be accompanied by a corresponding *.json file in BIDS format",
    )

    mtr_quantify_parser.add_argument(
        "--nosat",
        required=False,
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help="The path to the input image without bound pool saturation. If not"
        "supplied then it is assumed the sat nifti is multi-volume containing"
        "both. This should be accompanied by a corresponding *.json file in"
        "BIDS format",
    )

    mtr_quantify_parser.add_argument(
        "outputdir",
        type=DirType(should_exist=True),
        help="The directory to output to. "
        "Must exist. Will overwrite any existing files with the same names"
        "Magnetisation transfer ratio (MTR) maps and the accompanying JSON sidecar"
        "will be saved with the same filename as the saturated NIFTI, with '_MTRmap'"
        "appended",
    )

    mtr_quantify_parser.set_defaults(func=mtr_quantify)

    # ADC Quantify
    adc_quantify_parser = subparsers.add_parser(
        name="adc-quantify",
        description="Calculates the apparent diffusion coefficient based on"
        "a set of diffusion weighted images.",
    )

    adc_quantify_parser.add_argument(
        "dwi",
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help="The path to the 4D NIFTI file of diffusion weighted images."
        "There should be at least two volumes, one acquired with a b-value of 0."
        "It is assumed that there are also corresponding BIDS sidecar (*.json),"
        "bvec (*.bvec), and bval (*.bval ) files with the same base filenames"
        "as the NIFTI.",
    )

    adc_quantify_parser.add_argument(
        "outputdir",
        type=DirType(should_exist=True),
        help="The directory to output to. "
        "Must exist. Will overwrite any existing files with the same names"
        "Apparent Diffusion Coefficient (ADC) maps and the accompanying JSON sidecar"
        "will be saved with the same filename as the input DWI NIFTI, with '_ADCmap'"
        "appended",
    )

    adc_quantify_parser.set_defaults(func=adc_quantify)

    args = parser.parse_args()
    args.func(args)  # call the default function


if __name__ == "__main__":
    main()
