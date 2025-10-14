"""Command line interface for theremometry from multi-echo data."""

import json
import logging
import time
from pathlib import Path
from typing import Annotated, List, Optional, cast

import nibabel as nib
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mrimagetools.filters.multiecho_thermometry_filter import (
    GAMMA_H,
    MultiEchoThermometryParameters,
    multiecho_thermometry_filter,
)
from mrimagetools.v2.containers.image import NiftiImageContainer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer()

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


@app.command()
def multiecho_thermometry(
    multiecho_nifti_filenames: Annotated[
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
    segmentation_nifti_filename: Annotated[
        Path,
        typer.Option(
            "--segmentation",
            exists=True,
            file_okay=True,
            readable=True,
            help="Input segmentation (NIfTI) filename.",
        ),
    ],
    echo_times_files: Annotated[
        List[Path],
        typer.Option(
            "--echo-times",
            exists=True,
            file_okay=True,
            readable=True,
            help="Input list of echo times (text file, in seconds).",
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
            "--n-bootstrap",
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
) -> None:
    """Command line interface for thermometry from multi-echo data.
    Loads:
    - 1 or more Multiecho images (4D NIfTI, last dimension = echoes)
    - 1 segmentation image (3D NIfTI)
    - 1 or more lists of echo times (text file, in seconds). Number corresponds to the
    number of Multiecho images.

    Requires parameters for
    - Analysis method
    - number of boostrap iterations
    - output filename prefix
    - path to the output directory


    Saves:
    - Temperature map (3D NIfTI)
    - Report (json format)

    """
    # Start timing
    tic = time.perf_counter()

    console.print("[bold]Multi-Echo Thermometry[/bold]")

    # Validate method
    if method not in VALID_ANALYSIS_METHODS:
        console.print(
            f"[red]Error: Invalid analysis method '{method}'. Valid methods are:"
            f"{', '.join(VALID_ANALYSIS_METHODS)}[/red]"
        )
        raise typer.Exit(code=1)

    # Validate echo times filenames, number of files must equal the number of multiecho images
    if len(multiecho_nifti_filenames) != len(echo_times_files):
        console.print(
            f"[red]Error: Number of Multiecho images ({len(multiecho_nifti_filenames)}) "
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
            for filename in multiecho_nifti_filenames
        ]

        # if present, also load in the json sidecar files for each multiecho image
        json_sidecars = []
        for filename in multiecho_nifti_filenames:
            json_filename = filename.with_suffix(".json")
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
        and (image.affine == multiecho_images[0].affine)
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
    segmentation_image = cast(
        nib.nifti1.Nifti1Image, nib.load(segmentation_nifti_filename)
    )

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
    multiecho_data = [image.get_fdata(dtype=np.float64) for image in multiecho_images]
    all_echo_times = np.concatenate(echo_times)
    sorted_indices = np.argsort(all_echo_times)
    multiecho_data_sorted = np.stack(
        [data[..., sorted_indices] for data in multiecho_data], axis=-1
    )
    sorted_echo_times = all_echo_times[sorted_indices]

    # get the ImagingFrequency from the first json sidecar that has it, otherwise MagneticFieldStrength
    magnetic_field_tesla = None
    for json_sidecar in json_sidecars:
        if json_sidecar is not None and "ImagingFrequency" in json_sidecar:
            imaging_frequency_mhz = json_sidecar["ImagingFrequency"]
            magnetic_field_tesla = imaging_frequency_mhz / GAMMA_H
            break
        elif json_sidecar is not None and "MagneticFieldStrength" in json_sidecar:
            magnetic_field_tesla = json_sidecar["MagneticFieldStrength"]
            break

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

        progress.update(task, description="Thermometry Analysis Complete")

    # Save output
    # Prepare output paths
    if output_dir is None:
        output_dir = multiecho_nifti_filenames[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = multiecho_nifti_filenames[0].stem

    temperature_map_filename = output_dir / f"{output_prefix}_temperature_map.nii.gz"
    report_filename = output_dir / f"{output_prefix}_thermometry_report.json"
    nib.save(
        cast(NiftiImageContainer, temperature_map).nifti_image, temperature_map_filename
    )
    console.print(f"Saved temperature map to [bold]{temperature_map_filename}[/bold]")

    report_data = {
        "input_files": [str(f) for f in multiecho_nifti_filenames],
        "segmentation_file": str(segmentation_nifti_filename),
        "output_file": str(temperature_map_filename),
        "magnetic_field_tesla": magnetic_field_tesla,
        "analysis_method": method,
        "n_bootstrap": n_bootstrap if method == "regionwise_bootstrap" else None,
        "echo_times": sorted_echo_times.tolist(),
        "report": report,
        "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "processing_time_seconds": time.perf_counter() - tic,
    }
    with open(report_filename, "w") as f:
        json.dump(report_data, f, indent=2)
    console.print(f"Saved report to [bold]{report_filename}[/bold]")
