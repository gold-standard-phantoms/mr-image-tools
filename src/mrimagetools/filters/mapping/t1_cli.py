"""Command line interface for T1 mapping"""

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from mrimagetools.filters.mapping.t1 import T1Model, t1_mapping_from_files

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def map(
    input_t1w_filenames: Annotated[
        list[Path],
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Input T1w (NIfTI) filenames. Should have associated JSON files "
            "with the same name and the extension '.json'.",
        ),
    ],
    output_t1_filename: Annotated[
        Path,
        typer.Option(
            ...,
            help="Output T1 map (NIfTI) filename",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ],
    output_s0_filename: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            help=(
                "Output S0 map (NIfTI) filename. Is only relevant for inversion"
                " recovery fitting."
            ),
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ] = None,
    output_inv_eff_filename: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            help=(
                "Output inversion efficiency map (NIfTI) filename. Is only"
                " relevant for inversion recovery fitting."
            ),
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ] = None,
    mask_filename: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Mask (NIfTI) filename. Used to mask the input T1w images.",
        ),
    ] = None,
    model: Annotated[
        T1Model,
        typer.Option(
            help="Model to use for T1 mapping."
            " The 'general' model is the default and uses the general equation"
            " `S = S_0 (1 - 2 inv_eff exp(-TI/T1) + exp(-TR/T1))`"
            " for inversion recovery."
            " The 'classical' model uses the classical equation"
            " `S = S_0 (1 - 2 inv_eff exp(-TI/T1))` for inversion recovery."
            " The 'vtr' (variable TR) model uses the equation:"
            " `S = M (1 - exp(-TR/T1))`"
        ),
    ] = T1Model.GENERAL,
) -> None:
    """T1 mapping"""
    typer.echo(
        f"Input T1w filenames: {', '.join([str(path) for path in input_t1w_filenames])}"
    )
    typer.echo(f"Output T1 filename: {output_t1_filename}")
    t1_mapping_from_files(
        filepaths=input_t1w_filenames,
        model=model,
        output_t1_file=output_t1_filename,
        output_s0_file=output_s0_filename,
        output_inv_eff_file=output_inv_eff_filename,
        mask_file=mask_filename,
    )


def main() -> None:
    """Main function"""
    app()


if __name__ == "__main__":
    main()
