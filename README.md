# MRImageTools

## Overview

MRImageTools was developed to address the need to test MR image processing pipelines.
A strong emphasis has been placed on ensuring traceability of the developed code, in
particular with respect to testing. The library uses a “pipe and filter” architecture
with “filters” performing data processing, which provides a common interface between
processing blocks.

## How to cite

If you use MRImageTools in your work, please include the following citation:


## How to contribute

Got a great idea for something to implement in MRImageTools, or maybe you have just
found a bug? Create an issue at
`https://github.com/gold-standard-phantoms/mr-image-tools/issues` to get in touch with
the development team and we’ll take it from there.

## Installation

MRImageTools can be installed from PyPI.

### Python version

MRImageTools supports Python 3.9 and newer.

### Install (recommended)

```sh
pip install mrimagetools
```

### Install from source (development)

```sh
pip install -e .
```

## Virtual environments (`.venv`)

Use a virtual environment to manage dependencies for your project, both in
development and in production.

From the repository root, create a `.venv` folder:

```sh
python -m venv .venv
```

On Windows:

```bat
py -m venv .venv
```

Activate the environment:

```sh
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows (cmd):

```bat
.\.venv\Scripts\activate.bat
```

## Command-line tools

MRImageTools installs two console scripts:

- **`mrimagetools2`**: newer Typer-based CLI (actively developed).
- **`mrimagetools`**: legacy argparse-based CLI (older commands; will be merged in the future).

Use `--help` to discover commands and options:

```sh
mrimagetools2 --help
mrimagetools2 <command> --help
mrimagetools --help
mrimagetools <command> --help
```

### `mrimagetools2` (new)

Top-level command groups include:

- **`version`**: print the installed package version.
- **`t1`**: T1 mapping tools.
- **`t2`**: T2 mapping tools.
- **`adc`**: diffusion ADC mapping tools.
- **`multiecho-thermometry`**: temperature estimation from multi-echo magnitude data.

Examples:

```sh
mrimagetools2 version
mrimagetools2 multiecho-thermometry --help
```

Multi-echo thermometry example:

```sh
mrimagetools2 multiecho-thermometry \
  --segmentation segmentation.nii.gz \
  --echotimes echo_times_1.txt \
  --echotimes echo_times_2.txt \
  --method regionwise \
  --output-dir outputs \
  --output-prefix subject01 \
  multiecho_1.nii.gz multiecho_2.nii.gz
```

Notes:

- Echo-time files must contain echo times **in seconds**, one per echo/volume.
- Thermometry requires B0 metadata from an input JSON sidecar (`ImagingFrequency` or `MagneticFieldStrength`).
- Outputs are written as `<prefix>_temperature_map.nii.gz` and `<prefix>_report.json`.

### `mrimagetools` (legacy)

This CLI contains older commands. Exact availability can change by version, so
use `mrimagetools --help` as the source of truth. Common commands include:

- **`mtr-quantify`**: magnetisation transfer ratio mapping.
- **`adc-quantify`**: apparent diffusion coefficient mapping.
- **`create-hrgt`**: generate a high-resolution ground truth image.
- **`generate`**: generate Digital Reference Objects (DROs) for supported modalities.
- **`pipeline`**: run v2 pipelines via a nested subcommand structure.

## Web UI

The newer commands found in `mrimagetools2` can be accessed through a web browser.

Install the dependencies for the web UI support as follows:

```sh
pip install mrimagetools[web]
```

Launch the web server by calling the `mrimagetools.web2` module as follows:

```sh
python -m mrimagetools.web2
```

## Development

Development of this project must comply with the styling/quality rules and processes below:

- Before pushing code, ensure `CHANGELOG.md` is updated (see instructions in that file).
  Run `tox` to ensure tests and code-quality checks pass.
- Keep a good level of test coverage. To view coverage locally, run `coverage report`.
- Use [pre-commit](https://pre-commit.com/) (see `requirements/dev.txt`) and run
  `pre-commit install` when first cloning this repo. Hooks enforce minimum
  code-quality and formatting before commits. mypy, Pylint, black and isort will be
  run automatically, and results displayed after a git commit and on CI.
- Run [mypy](https://github.com/python/mypy) on source code to find static typing errors.
- Use [Pylint](https://pylint.org/) as the linter (config in `.pylintrc`). Keep lint
  errors low and the score high (close to 10).
- Use [black](https://black.readthedocs.io/en/stable/) for auto-formatting Python files.
- Use [isort](https://isort.readthedocs.io/en/latest/) to keep imports consistent.
