"""The CLI argument parser for the pipelines"""


from argparse import ArgumentParser

from mrimagetools.v2.pipelines import dwi_pipeline


def add_cli_arguments_to(parser: ArgumentParser) -> None:
    """Add all of the pipeline arguments (including subparsers) to the supplied
    argument parser, allowing for argument nesting"""

    # Add the pipeline_type parser
    pipeline_type_subparser = parser.add_subparsers(
        title="pipeline_type", description="The type of pipeline to run"
    )

    # Add the DWI pipeline parser
    dwi_parser = pipeline_type_subparser.add_parser(
        name="dwi", description="DWI pipeline"
    )
    dwi_pipeline.add_cli_arguments_to(dwi_parser)
