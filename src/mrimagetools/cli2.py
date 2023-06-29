"""Command line interface (using Typer)"""

import logging

import typer

from mrimagetools.filters.mapping import t1_cli, t2_cli

from . import __version__

logging.basicConfig(level=logging.INFO)
app = typer.Typer()
app.add_typer(t1_cli.app, name="t1")
app.add_typer(t2_cli.app, name="t2")

if __name__ == "__main__":
    app()


@app.command()
def version() -> None:
    """Print version"""
    typer.echo(__version__)


def main() -> None:
    """Main function"""
    app()


if __name__ == "__main__":
    main()
