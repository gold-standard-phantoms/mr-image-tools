"""Command line interface for T2 mapping"""

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from mrimagetools.filters.mapping.t2 import T2Model, t2_mapping_from_files

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def map(
    input_t2w_filenames: Annotated[
        list[Path],
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Input t2w (NIfTI) filenames. Should have associated JSON files "
            "with the same name and the extension '.json'.",
        ),
    ],
    output_t2_filename: Annotated[
        Path,
        typer.Option(
            ...,
            help="Output t2 map (NIfTI) filename",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ],
    mask_filename: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Mask (NIfTI) filename. Used to mask the input t2w images.",
        ),
    ] = None,
    model: Annotated[
        T2Model,
        typer.Option(
            help="The model to use for the Mono-Exponential Fitting in T2-Relaxometry."
            " The full model is"
            " `S(TE)=k.S_o.exp(-TE/T_2)+offset`. The reduced model is"
            " `S(TE)=k.S_o.exp(-TE/T_2)`. In both cases, `k.S_0`"
            " is combined into a single parameter A."
        ),
    ] = T2Model.FULL,
    skip_echos: Annotated[
        int,
        typer.Option(
            help=(
                "Number of echo times to skip."
                " Discarding the first echo is a fast and easy method to minimize the"
                " error in T2 fitting and is the default option."
                " Default: 1"
            )
        ),
    ] = 1,
) -> None:
    """t2 mapping"""
    typer.echo(
        f"Input t2w filenames: {', '.join([str(path) for path in input_t2w_filenames])}"
    )
    typer.echo(f"Output t2 filename: {output_t2_filename}")
    t2_mapping_from_files(
        filepaths=input_t2w_filenames,
        output_t2_file=output_t2_filename,
        model=model,
        mask_file=mask_filename,
        skip_echos=skip_echos,
    )


def main() -> None:
    """Main function"""
    app()


if __name__ == "__main__":
    main()
