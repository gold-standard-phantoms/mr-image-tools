"""Command line interface for theremometry from multi-echo data."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, List, Optional, cast, Tuple
import pdb

import nibabel as nib
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mrimagetools.filters.multiecho_thermometry_filter import (
    GAMMA_H,
    MultiEchoThermometryParameters,
    ThermometryResults,
    multiecho_thermometry_filter,
)
from mrimagetools.v2.containers.image import NiftiImageContainer


@dataclass
class ThermometryReportData:
    """Dataclass to hold the report data for the multiecho thermometry analysis."""

    input_files: List[Path]
    segmentation_file: Path
    output_file: Path
    magnetic_field_tesla: float
    analysis_method: str
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
            "analysis_method": self.analysis_method,
            "n_bootstrap": self.n_bootstrap,
            "echo_times": self.echo_times,
            "report": [r.to_json() for r in self.report],
        }


console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)

VALID_ANALYSIS_METHODS = ["regionwise", "voxelwise", "regionwise_bootstrap"]


def load_echo_times(echo_times_file: Path) -> np.ndarray:
    """Load echo times from a text file.

    Args:
        echo_times_file (Path): Path to the text file containing echo times in seconds.

    Returns:
        np.ndarray: Array of echo times in seconds.
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
    """Remove a suffix from a filename.

    Args:
        filename (Path): The original filename.
        suffix (str): The suffix to remove.

    Returns:
        Path: The filename without the suffix.
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
        str,
        typer.Option(
            "--method",
            help="Analysis method. Options are: voxelwise, regionwise, regionwise_bootstrap.",
        ),
    ] = "regionwise",
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
    """Perform thermometry from multi-echo data.

    Args:
        segmentation_nifti_file (Path): Input segmentation (NIfTI) filename.
        multiecho_nifti_files (List[Path]): Input Multiecho (NIfTI) filenames.
        echo_times_files (List[Path]): Input list of echo times (text file, in seconds),
            one file per multiecho image.
        method (str, optional): Analysis method. Options are: voxelwise, regionwise,
            regionwise_bootstrap. Defaults to "regionwise".
        n_bootstrap (int, optional): Number of bootstrap iterations. Defaults to 100.
        output_prefix (Optional[str], optional): Output filename prefix. Defaults to None.
        output_dir (Optional[Path], optional): Output directory. Defaults to None.

    Returns:
        Tuple[NiftiImageContainer, dict]: A tuple containing:
            - temperature_map (NiftiImageContainer): The calculated temperature map.
            - report_data (dict): A dictionary containing the report of the analysis.
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
        if not filename.suffix in [".nii", ".gz"]:
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
        if not filename.suffix in [".txt", ".tsv", ".csv"]:
            console.print(f"[red]Error: {filename} is not a text file.[/red]")
            raise typer.Exit(code=1)

    # Validate method
    if method not in VALID_ANALYSIS_METHODS:
        console.print(
            f"[red]Error: Invalid analysis method '{method}'. Valid methods are:"
            f"{', '.join(VALID_ANALYSIS_METHODS)}[/red]"
        )
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
            cast(nib.nifti1.Nifti1Image, nib.load(filename))  # type: ignore
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
            f"[red]Error: Multiecho images must be 4 dimensional, have the same shape, and affine[/red]"
        )
        raise typer.Exit(code=1)

    # validate that the number of echoes in each multiecho image matches the number of echo times provided
    if not all(
        image.shape[-1] == len(echo_times[i])
        for i, image in enumerate(multiecho_images)
    ):
        console.print(
            f"[red]Error: Number of echoes in each Multiecho image must match the number of echo times provided[/red]"
        )
        raise typer.Exit(code=1)

    # Load segmentation data
    segmentation_image = cast(nib.nifti1.Nifti1Image, nib.load(segmentation_nifti_file))  # type: ignore

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
        nifti_img=nib.nifti1.Nifti1Image(
            multiecho_data_sorted, multiecho_images[0].affine
        )
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
                analysis_method=method,  # type: ignore
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
    nib.nifti1.save(temperature_map.nifti_image, temperature_map_filename)
    console.print(f"Saved temperature map to [bold]{temperature_map_filename}[/bold]")

    report_data = ThermometryReportData(
        input_files=multiecho_nifti_files,
        segmentation_file=segmentation_nifti_file,
        output_file=temperature_map_filename,
        magnetic_field_tesla=magnetic_field_tesla,
        analysis_method=method,
        n_bootstrap=n_bootstrap if method == "regionwise_bootstrap" else None,
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
