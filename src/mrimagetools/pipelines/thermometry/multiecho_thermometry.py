"""Multi-echo MR thermometry command line interface.

This module provides a Typer-based CLI command that estimates temperature from
multi-echo magnitude NIfTI images using the dual-resonance model implemented in
`mrimagetools.filters.multiecho_thermometry_filter`.

At a high level the command:

- Loads one or more 4D multi-echo magnitude images (echo dimension is the last axis).
- Loads a 3D segmentation/label map (same XYZ shape/affine as the images).
- Loads echo times for each input image (text files, in seconds), concatenates and sorts
  all echoes by TE.
- Determines B0 in Tesla from an optional JSON sidecar (`ImagingFrequency` or
  `MagneticFieldStrength`).
- Runs voxelwise or regionwise fitting, writes a temperature map NIfTI and a JSON report.
"""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional, Tuple, cast

import nibabel as nib
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mrimagetools.filters.multiecho_thermometry_filter import (
    GAMMA_H,
    MultiEchoThermometryParameters,
    ThermometryResults,
    multiecho_thermometry_filter,
)
from mrimagetools.v2.containers.image import NiftiImageContainer


class AnalysisMethod(str, Enum):
    """The analysis method to use."""

    REGIONWISE = "regionwise"
    REGIONWISE_BOOTSTRAP = "regionwise_bootstrap"
    VOXELWISE = "voxelwise"


@dataclass
class ThermometryReportData:
    """Dataclass to hold the report data for the multiecho thermometry analysis."""

    input_files: List[Path]
    segmentation_file: Path
    output_file: Path
    magnetic_field_tesla: float
    analysis_method: AnalysisMethod
    n_bootstrap: Optional[int]
    echo_times: List[float]
    report: List[ThermometryResults]
    acquisition_date_time: List[str]
    processing_date: str
    processing_time_seconds: float

    def to_json(self) -> dict:
        """Return a JSON serializable dictionary."""
        return {
            "input_files": [str(f) for f in self.input_files],
            "segmentation_file": str(self.segmentation_file),
            "output_file": str(self.output_file),
            "acquisition_date_time": self.acquisition_date_time,
            "processing_date": self.processing_date,
            "processing_time_seconds": self.processing_time_seconds,
            "magnetic_field_tesla": self.magnetic_field_tesla,
            "analysis_method": self.analysis_method.value,
            "n_bootstrap": self.n_bootstrap,
            "echo_times": self.echo_times,
            "report": [r.to_json() for r in self.report],
        }


console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


def load_echo_times(echo_times_file: Path) -> np.ndarray:
    """Load echo times from a text file.

    The file must contain a 1D list/array of echo times **in seconds**. Common
    extensions (e.g. `.txt`, `.tsv`, `.csv`) are accepted by the CLI, but the
    contents must still parse as numeric values.

    Args:
        echo_times_file: Path to a text file containing echo times in seconds.

    Returns:
        A 1D NumPy array of echo times in seconds.

    Raises:
        ValueError: If the parsed data is not 1D.
        Exception: Propagates any I/O or parsing errors from NumPy.
    """
    try:
        echo_times = np.loadtxt(echo_times_file)
        logger.info(f"Loaded {len(echo_times)} echo times from {echo_times_file}.")
        if echo_times.ndim != 1:
            raise ValueError("Echo times file must contain a 1D array.")
        return echo_times  #
    except Exception as e:
        logger.error(f"Error loading echo times from {echo_times_file}: {e}")
        raise


def remove_suffix(filename: Path, suffix: str) -> Path:
    """Remove a single suffix from a filename.

    Args:
        filename: The original filename.
        suffix: The suffix to remove (e.g. `.gz`).

    Returns:
        The filename without the suffix if it matches; otherwise the original filename.
    """
    if filename.suffix == suffix:
        return filename.with_suffix("")
    return filename


@app.command()
def multiecho_thermometry(
    segmentation_nifti_file: Annotated[
        Path,
        typer.Option(
            "--segmentation",
            exists=True,
            file_okay=True,
            readable=True,
            help="Input segmentation (NIfTI) filename.",
        ),
    ],
    multiecho_nifti_files: Annotated[
        List[Path],
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Input Multiecho (NIfTI) filenames.",
        ),
    ],
    echo_times_files: Annotated[
        List[Path],
        typer.Option(
            "--echotimes",
            exists=True,
            file_okay=True,
            readable=True,
            help="Input list of echo times (text file, in seconds), one file per multiecho image.",
        ),
    ],
    method: Annotated[
        AnalysisMethod,
        typer.Option(
            "--method",
            help="Analysis method. Options are: voxelwise, regionwise, regionwise_bootstrap.",
        ),
    ] = AnalysisMethod.REGIONWISE,
    n_bootstrap: Annotated[
        int,
        typer.Option(
            "--nb",
            help="Number of bootstrap iterations.",
        ),
    ] = 100,
    output_prefix: Annotated[
        Optional[str],
        typer.Option(
            "--output-prefix",
            help="Output filename prefix.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Output directory.",
        ),
    ] = None,
) -> Tuple[NiftiImageContainer, ThermometryReportData]:
    """Estimate temperature from multi-echo magnitude images and a segmentation.

    This is the **user-facing** entrypoint used by the `mrimagetools` CLI.

    You provide:

    - One or more 4D multi-echo magnitude NIfTI files (echo dimension must be the last axis).
    - A 3D segmentation/label-map NIfTI file that is co-located with the multi-echo data
      (same XYZ shape and affine). Label value 0 is treated as background.
    - A list of echo-time text files (one per multi-echo image), containing echo times
      **in seconds**. The number of echo times in each file must match the number of
      echoes (4th dimension) in its corresponding image.

    The command concatenates all echoes across all provided multi-echo images, sorts them
    by echo time, and then runs one of the analysis methods:

    - `regionwise`: Fit the dual-resonance model to the **mean** signal within each
      non-zero segmentation label. The fitted temperature is assigned to all voxels
      in that region.
    - `voxelwise`: Fit the model independently per voxel within each region; the output
      map is voxelwise, and region summaries are computed from voxel estimates.
    - `regionwise_bootstrap`: Like `regionwise`, but uses bootstrapping within each
      region to estimate uncertainty (controlled by `--nb/--n-bootstrap`).

    Magnetic field strength B0 (Tesla) is determined from an optional JSON sidecar
    for the input images. The first sidecar containing either `ImagingFrequency` (MHz) or
    `MagneticFieldStrength` (Tesla) is used. If no suitable metadata is found, the command
    exits with an error.

    Outputs (written into `--output-dir` or the input directory by default):

    - `<output_prefix>_temperature_map.nii.gz`: Temperature map in °C.
    - `<output_prefix>_report.json`: Summary report including per-region results and
      timing/metadata.

    Args:
        segmentation_nifti_file: Segmentation/label-map NIfTI (3D).
        multiecho_nifti_files: One or more multi-echo magnitude NIfTI files (4D).
        echo_times_files: Echo-time text files (seconds), one per multi-echo image.
        method: Analysis method to use.
        n_bootstrap: Number of bootstrap iterations (only used for `regionwise_bootstrap`).
        output_prefix: Prefix for output filenames. Defaults to the first input image stem.
        output_dir: Output directory. Defaults to the directory of the first input image.

    Returns:
        A tuple of `(temperature_map, report_data)` where:

        - `temperature_map` is a `NiftiImageContainer` containing the output temperature map.
        - `report_data` is a `ThermometryReportData` instance suitable for JSON serialization.

    Raises:
        typer.Exit: For invalid inputs (missing files, mismatched shapes/affines, echo-time
            length mismatches, or missing B0 metadata).
    """
    # Start timing
    tic = time.perf_counter()

    console.print("[bold]Multi-Echo Thermometry[/bold]")
    # pdb.set_trace()

    # validate the input multiecho files
    for filename in multiecho_nifti_files:
        if not filename.exists():
            console.print(f"[red]Error: File {filename} does not exist.[/red]")
            raise typer.Exit(code=1)
        if not filename.is_file():
            console.print(f"[red]Error: {filename} is not a file.[/red]")
            raise typer.Exit(code=1)
        if filename.suffix not in [".nii", ".gz"]:
            console.print(f"[red]Error: {filename} is not a NIfTI file.[/red]")
            raise typer.Exit(code=1)

    # validate the input echo times files
    for filename in echo_times_files:
        if not filename.exists():
            console.print(f"[red]Error: File {filename} does not exist.[/red]")
            raise typer.Exit(code=1)
        if not filename.is_file():
            console.print(f"[red]Error: {filename} is not a file.[/red]")
            raise typer.Exit(code=1)
        if filename.suffix not in [".txt", ".tsv", ".csv"]:
            console.print(f"[red]Error: {filename} is not a text file.[/red]")
            raise typer.Exit(code=1)

    # Validate echo times filenames, number of files must equal the number of multiecho images
    if len(multiecho_nifti_files) != len(echo_times_files):
        console.print(
            f"[red]Error: Number of Multiecho images ({len(multiecho_nifti_files)}) "
            f"does not match number of echo times files ({len(echo_times_files)})[/red]"
        )
        raise typer.Exit(code=1)

    # Load echo times
    echo_times = [
        load_echo_times(echo_times_file) for echo_times_file in echo_times_files
    ]

    # Load multi echo data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Multiecho data", total=None)

        multiecho_images = [
            cast(nib.Nifti1Image, nib.load(filename))
            for filename in multiecho_nifti_files
        ]

        # if present, also load in the json sidecar files for each multiecho image
        json_sidecars = []
        for filename in multiecho_nifti_files:
            json_filename = remove_suffix(filename, ".gz").with_suffix(".json")
            if json_filename.exists():
                with open(json_filename, "r") as f:
                    json_sidecar = json.load(f)
                json_sidecars.append(json_sidecar)
            else:
                json_sidecars.append(None)

        progress.update(
            task, description=f"Loaded {len(multiecho_images)} Multiecho Images"
        )

    # Validate multiecho dimensions
    # the first three dimensions of all multiecho images must be the same
    # all multiecho images must have the same affine
    if not all(
        (image.shape == multiecho_images[0].shape)
        and (image.affine == multiecho_images[0].affine).all()
        and (image.ndim == 4)
        for image in multiecho_images
    ):
        console.print(
            "[red]Error: Multiecho images must be 4 dimensional, have the same shape, "
            "and affine[/red]"
        )
        raise typer.Exit(code=1)

    # validate that the number of echoes in each multiecho image matches the number of echo times provided
    if not all(
        image.shape[-1] == len(echo_times[i])
        for i, image in enumerate(multiecho_images)
    ):
        console.print(
            "[red]Error: Number of echoes in each Multiecho image must match the "
            "number of echo times provided[/red]"
        )
        raise typer.Exit(code=1)

    # Load segmentation data
    segmentation_image = cast(nib.Nifti1Image, nib.load(segmentation_nifti_file))

    # Validate the segmentation image
    if not (
        segmentation_image.ndim == 3
        and segmentation_image.shape == multiecho_images[0].shape[:-1]
    ):
        console.print(
            "[red]Error: Segmentation image must be a 3D image with the same shape "
            "as the Multiecho images[/red]"
        )
        raise typer.Exit(code=1)

    # extract the image data arrays from the multiecho data, convert to np.float64 for processing
    multiecho_data = np.concatenate(
        [image.get_fdata(dtype=np.float64) for image in multiecho_images], axis=3
    )
    all_echo_times = np.concatenate(echo_times)
    sorted_indices = np.argsort(all_echo_times)
    multiecho_data_sorted = multiecho_data[:, :, :, sorted_indices]
    sorted_echo_times = all_echo_times[sorted_indices]

    # get the ImagingFrequency from the first json sidecar that has it, otherwise MagneticFieldStrength
    magnetic_field_tesla = None
    acquisition_date_time = []
    for json_sidecar in json_sidecars:
        if json_sidecar is not None and "ImagingFrequency" in json_sidecar:
            imaging_frequency_mhz = json_sidecar["ImagingFrequency"]
            magnetic_field_tesla = imaging_frequency_mhz / (GAMMA_H / 1e6)

        elif json_sidecar is not None and "MagneticFieldStrength" in json_sidecar:
            magnetic_field_tesla = json_sidecar["MagneticFieldStrength"]

        if (
            json_sidecar is not None and "AcquisitionDateTime" in json_sidecar
        ):  # use AcquisitionDateTime if available
            acquisition_date_time.append(json_sidecar["AcquisitionDateTime"])
        elif (
            json_sidecar is not None and "AcquisitionTime" in json_sidecar
        ):  # fallback to AcquisitionTime
            acquisition_date_time.append(json_sidecar["AcquisitionTime"])
        else:
            acquisition_date_time.append("Unknown")

    if magnetic_field_tesla is None:
        console.print(
            "[red]Error: Could not find MagneticFieldStrength or ImagingFrequency in any of the json sidecars[/red]"
        )
        raise typer.Exit(code=1)

    # create image containers for multiecho_data_sorted and segmentation data
    multiecho_input_image = NiftiImageContainer(
        nifti_img=nib.Nifti1Image(multiecho_data_sorted, multiecho_images[0].affine)
    )
    segmentation_input_image = NiftiImageContainer(nifti_img=segmentation_image)

    # Run analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running Thermometry Analysis", total=None)

        report, temperature_map = multiecho_thermometry_filter(
            parameters=MultiEchoThermometryParameters(
                image_multiecho=multiecho_input_image,
                image_segmentation=segmentation_input_image,
                echo_times=sorted_echo_times.tolist(),
                analysis_method=method.value,
                n_bootstrap=n_bootstrap,
                magnetic_field_tesla=magnetic_field_tesla,
            )
        )
        temperature_map = cast(NiftiImageContainer, temperature_map)

        progress.update(task, description="Thermometry Analysis Complete")

    # Save output
    # Prepare output paths
    if output_dir is None:
        output_dir = multiecho_nifti_files[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = remove_suffix(multiecho_nifti_files[0], ".gz").stem

    temperature_map_filename = output_dir / f"{output_prefix}_temperature_map.nii.gz"
    report_filename = output_dir / f"{output_prefix}_report.json"
    nib.save(temperature_map.nifti_image, temperature_map_filename)
    console.print(f"Saved temperature map to [bold]{temperature_map_filename}[/bold]")

    report_data = ThermometryReportData(
        input_files=[f.relative_to(output_dir) for f in multiecho_nifti_files],
        segmentation_file=segmentation_nifti_file.relative_to(output_dir),
        output_file=temperature_map_filename.relative_to(output_dir),
        magnetic_field_tesla=magnetic_field_tesla,
        analysis_method=method,
        n_bootstrap=(
            n_bootstrap if method is AnalysisMethod.REGIONWISE_BOOTSTRAP else None
        ),
        echo_times=sorted_echo_times.tolist(),
        report=report,
        acquisition_date_time=acquisition_date_time,
        processing_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        processing_time_seconds=time.perf_counter() - tic,
    )
    with open(report_filename, "w") as f:
        json.dump(report_data.to_json(), f, indent=2)
    console.print(f"Saved report to [bold]{report_filename}[/bold]")
    return temperature_map, report_data


def main() -> None:
    """Main entry point for CLI"""
    app()


if __name__ == "__main__":
    main()
