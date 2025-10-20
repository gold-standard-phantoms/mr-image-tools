"""Command line interface for ADC mapping from DWI data"""

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Optional, cast

import nibabel as nib
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mrimagetools.filters.adc_quantification_filter import (
    adc_quantification_simple,
    process_dwi_volume,
)
from mrimagetools.v2.containers.image import NiftiImageContainer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer()


def load_bvalues(
    bval_file: Optional[Path] = None,
    json_file: Optional[Path] = None,
    b_values_str: Optional[str] = None,
) -> np.ndarray:
    """Load b-values from various sources.

    :param bval_file: Path to .bval file (FSL format)
    :param json_file: Path to JSON sidecar
    :param b_values_str: Comma-separated b-values string
    :return: Array of b-values
    :raises ValueError: If no valid b-values source provided
    """
    if bval_file is not None:
        # Load FSL format bval file (space or tab separated)
        b_values = np.loadtxt(bval_file)
        logger.info(f"Loaded {len(b_values)} b-values from {bval_file}")
        return b_values

    if json_file is not None:
        with open(json_file) as f:
            json_data = json.load(f)
        # Look for common b-value keys in JSON
        for key in ["DiffusionBValue", "bvalue", "b_value", "bval"]:
            if key in json_data:
                b_values = np.array(json_data[key])
                logger.info(f"Loaded {len(b_values)} b-values from JSON key '{key}'")
                return b_values
        raise ValueError(f"No b-value information found in {json_file}")

    if b_values_str is not None:
        # Parse comma-separated string
        b_values = np.array([float(b) for b in b_values_str.split(",")])
        logger.info(f"Parsed {len(b_values)} b-values from command line")
        return b_values

    raise ValueError(
        "No b-values source provided. Use --bval-file, --json-sidecar, or --b-values"
    )


def save_nifti_with_header(
    data: np.ndarray,
    reference_nifti: nib.Nifti1Image,
    output_path: Path,
    description: str = "",
) -> None:
    """Save NIfTI file preserving spatial information from reference.

    :param data: Data array to save
    :param reference_nifti: Reference NIfTI for header/affine
    :param output_path: Output file path
    :param description: Description for logging
    """
    # Create new image with proper data type (float32 for floating point data)
    if data.dtype in [np.float32, np.float64]:
        # Ensure float data is saved as float32 for efficiency
        data = data.astype(np.float32)

    # Create new header to avoid inheriting inappropriate data type
    output_img = nib.Nifti1Image(data, reference_nifti.affine)
    nib.save(output_img, output_path)
    logger.info(f"Saved {description} to {output_path}")


VALID_METHODS = ["simple", "lls", "wlls2", "iwlls"]


@app.command()
def adc_mapping(
    input_nifti_filename: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Input DWI (NIfTI) filename.",
        ),
    ],
    bval_file: Annotated[
        Optional[Path],
        typer.Option(
            "--bval-file",
            exists=True,
            file_okay=True,
            readable=True,
            help="B-values file in FSL format (.bval)",
        ),
    ] = None,
    json_sidecar: Annotated[
        Optional[Path],
        typer.Option(
            "--json-sidecar",
            exists=True,
            file_okay=True,
            readable=True,
            help="JSON sidecar containing b-values",
        ),
    ] = None,
    b_values: Annotated[
        Optional[str],
        typer.Option(
            "--b-values",
            help="Comma-separated b-values (e.g., '0,500,1000')",
        ),
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="ADC fitting method to use (simple, lls, wlls2, iwlls)",
        ),
    ] = "wlls2",
    mask: Annotated[
        Optional[Path],
        typer.Option(
            "--mask",
            exists=True,
            file_okay=True,
            readable=True,
            help="Binary mask NIfTI file",
        ),
    ] = None,
    output_prefix: Annotated[
        Optional[str],
        typer.Option(
            "--output-prefix",
            help="Output filename prefix",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Output directory",
        ),
    ] = None,
    save_s0: Annotated[
        bool,
        typer.Option(
            "--save-s0",
            help="Save S0 (baseline) map",
        ),
    ] = False,
    save_r2: Annotated[
        bool,
        typer.Option(
            "--save-r2",
            help="Save R-squared quality map",
        ),
    ] = False,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            help="Maximum iterations for IWLLS method",
        ),
    ] = 5,
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            help="Convergence tolerance for IWLLS method",
        ),
    ] = 1e-6,
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            help="Generate JSON report with statistics",
        ),
    ] = False,
) -> None:
    """Calculate ADC maps from DWI data using various fitting methods."""

    # Start timing
    tic = time.perf_counter()

    console.print("[bold]ADC Mapping[/bold]")
    console.print(f"Input: {input_nifti_filename}")
    console.print(f"Method: {method}")

    # Validate method
    if method not in VALID_METHODS:
        console.print(
            f"[red]Error: Invalid method '{method}'. Must be one of:"
            f" {', '.join(VALID_METHODS)}[/red]"
        )
        raise typer.Exit(code=1)

    # Load b-values
    try:
        b_vals = load_bvalues(bval_file, json_sidecar, b_values)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

    # Load DWI data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading DWI data...", total=None)

        dwi_img = cast(nib.Nifti1Image, nib.load(input_nifti_filename))
        dwi_data = dwi_img.get_fdata()

        progress.update(task, description=f"Loaded DWI data: shape {dwi_data.shape}")

    # Validate dimensions
    if dwi_data.ndim == 3:
        # Single volume - check if this makes sense
        if len(b_vals) != 1:
            console.print(
                f"[red]Error: 3D data but {len(b_vals)} b-values provided[/red]"
            )
            raise typer.Exit(code=1)
        dwi_data = dwi_data[..., np.newaxis]
    elif dwi_data.ndim == 4:
        if dwi_data.shape[-1] != len(b_vals):
            console.print(
                f"[red]Error: Number of b-values ({len(b_vals)}) doesn't match "
                f"number of DWI volumes ({dwi_data.shape[-1]})[/red]"
            )
            raise typer.Exit(code=1)
    else:
        console.print(
            f"[red]Error: DWI data must be 3D or 4D, got shape {dwi_data.shape}[/red]"
        )
        raise typer.Exit(code=1)

    # Check for b=0
    if 0 not in b_vals and np.min(b_vals) > 100:
        console.print(
            "[yellow]Warning: No b=0 image found. Using lowest b-value as"
            " reference.[/yellow]"
        )

    # Load mask if provided
    mask_data = None
    if mask is not None:
        mask_img = cast(nib.Nifti1Image, nib.load(mask))
        mask_data = mask_img.get_fdata().astype(bool)
        console.print(f"Loaded mask: {np.sum(mask_data)} voxels")

    # Convert data to float64 for processing
    dwi_data = dwi_data.astype(np.float64)
    b_vals_array = b_vals.astype(np.float64)

    # Process based on method
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if method == "simple":
            task = progress.add_task("Calculating ADC (simple method)...", total=None)

            # Simple method requires b=0
            b_vals_list = b_vals.tolist()
            if 0 not in b_vals_list:
                progress.stop()
                console.print(
                    "[red]Error: Simple method requires a b=0 image, but b-values are"
                    f" {b_vals_list}[/red]"
                )
                console.print(
                    "[yellow]Consider using 'lls', 'wlls2', or 'iwlls' methods"
                    " instead.[/yellow]"
                )
                raise typer.Exit(code=1)

            # Simple method needs BaseImageContainer wrapper
            dwi_container = NiftiImageContainer(
                nib.Nifti1Image(dwi_data, dwi_img.affine)
            )
            adc_data = adc_quantification_simple(dwi_container, b_vals_list)

            # Convert to clinical units (× 10⁻³ mm²/s)
            adc_data = adc_data * 1000

            # Create placeholder S0 and R² maps
            s0_data = dwi_data[..., np.argmin(b_vals)]  # Use b=0 or lowest b-value
            r2_data = np.ones(adc_data.shape[:-1])  # Placeholder R²

        else:
            # Use advanced fitting methods
            task = progress.add_task(
                f"Calculating ADC ({method} method)...", total=None
            )

            if method in ["lls", "wlls2", "iwlls"]:
                result = process_dwi_volume(
                    dwi_data,
                    b_vals_array,
                    mask=mask_data,
                    method=method,  # type: ignore
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
                adc_data = result.adc_map
                s0_data = result.s0_map
                r2_data = result.r2_map
            else:
                console.print(f"[red]Error: Unknown method {method}[/red]")
                raise typer.Exit(code=1)

        progress.update(task, description="Processing complete")

    # Prepare output paths
    if output_dir is None:
        output_dir = input_nifti_filename.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = input_nifti_filename.stem

    # Save ADC map (always)
    adc_output = output_dir / f"{output_prefix}_ADCmap_{method}.nii.gz"

    # Handle different output shapes
    if adc_data.ndim == 4:
        # Simple method returns 4D, take mean across non-b0 volumes
        adc_save = np.mean(adc_data, axis=-1)
    else:
        adc_save = adc_data

    save_nifti_with_header(adc_save, dwi_img, adc_output, "ADC map")

    # Save S0 map if requested
    if save_s0:
        s0_output = output_dir / f"{output_prefix}_S0map_{method}.nii.gz"
        save_nifti_with_header(s0_data, dwi_img, s0_output, "S0 map")

    # Save R² map if requested
    if save_r2:
        r2_output = output_dir / f"{output_prefix}_R2map_{method}.nii.gz"
        save_nifti_with_header(r2_data, dwi_img, r2_output, "R² map")

    # Calculate statistics
    if mask_data is not None:
        adc_values = adc_save[mask_data]
    else:
        adc_values = adc_save[adc_save > 0]  # Exclude background

    if len(adc_values) > 0:
        stats = {
            "mean_adc": float(np.mean(adc_values)),
            "median_adc": float(np.median(adc_values)),
            "std_adc": float(np.std(adc_values)),
            "min_adc": float(np.min(adc_values)),
            "max_adc": float(np.max(adc_values)),
            "voxels_processed": int(len(adc_values)),
        }
    else:
        stats = {
            "mean_adc": 0.0,
            "median_adc": 0.0,
            "std_adc": 0.0,
            "min_adc": 0.0,
            "max_adc": 0.0,
            "voxels_processed": 0,
        }

    # Save report if requested
    if report:
        report_data = {
            "input_file": str(input_nifti_filename),
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "b_values": b_vals.tolist(),
            "statistics": stats,
            "processing_time_seconds": time.perf_counter() - tic,
            "output_files": {
                "adc_map": str(adc_output),
            },
        }

        if save_s0:
            report_data["output_files"]["s0_map"] = str(s0_output)
        if save_r2:
            report_data["output_files"]["r2_map"] = str(r2_output)

        report_file = output_dir / f"{output_prefix}_ADC_report_{method}.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"Report saved to {report_file}")

    # Display results table
    table = Table(title="ADC Mapping Results")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Method", method)
    table.add_row("B-values", f"{b_vals.tolist()} s/mm²")
    table.add_row("Voxels processed", f"{stats['voxels_processed']:,}")
    table.add_row("Mean ADC", f"{stats['mean_adc']:.3f} × 10⁻³ mm²/s")
    table.add_row("Median ADC", f"{stats['median_adc']:.3f} × 10⁻³ mm²/s")
    table.add_row("Std ADC", f"{stats['std_adc']:.3f} × 10⁻³ mm²/s")
    table.add_row("Processing time", f"{time.perf_counter() - tic:.2f} seconds")

    console.print(table)
    console.print(f"[green]✓[/green] ADC map saved to {adc_output}")


def main() -> None:
    """Main entry point for CLI"""
    app()


if __name__ == "__main__":
    main()
