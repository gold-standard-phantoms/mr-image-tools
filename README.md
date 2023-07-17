# Overview

MRImageTools was developed to address the need to test MR image processing pipelines.
A strong emphasis has been placed on ensuring traceability of the developed
code, in particular with respect to testing. The library uses a ‘pipe and filter’ architecture
with ‘filters’ performing data processing, which provides a common interface between processing
blocks.

## How To Cite

If you use MRImageTools in your work, please include the following citation

## How To Contribute

Got a great idea for something to implement in MRImageTools, or maybe you have just
found a bug? Create an issue at
[https://github.com/gold-standard-phantoms/mrimagetools/issues](https://github.com/gold-standard-phantoms/mrimagetools/issues) to get in touch with
the development team and we’ll take it from there.

# Installation

MRImageTools can be installed as a module directly from the python package index.
For more information how to use a python package in this
way please see [https://docs.python.org/3/installing/index.html](https://docs.python.org/3/installing/index.html)

## Python Version

We recommend using the latest version of Python. MRImageTools supports Python
3.9 and newer.

## Dependencies

These distributions will be installed automatically when installing MRImageTools.


* [nibabel](https://nipy.org/nibabel/) provides read / write access to some common neuroimaging file formats


* [numpy](https://numpy.org/) provides efficient calculations with arrays and matrices


* [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/) provides an implementation of JSON Schema validation for Python


* [nilearn](https://nipy.org/packages/nilearn/index.html) provides image manipulation tools and statistical learning for neuroimaging data

## Virtual environments

Use a virtual environment to manage the dependencies for your project, both in
development and in production.

What problem does a virtual environment solve? The more Python projects you
have, the more likely it is that you need to work with different versions of
Python libraries, or even Python itself. Newer versions of libraries for one
project can break compatibility in another project.

Virtual environments are independent groups of Python libraries, one for each
project. Packages installed for one project will not affect other projects or
the operating system’s packages.

Python comes bundled with the `venv` module to create virtual
environments.

### Create an environment

Create a project folder and a `venv` folder within:

```sh
$ mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

On Windows:

```bat
$ py -3 -m venv venv
```

### Activate the environment

Before you work on your project, activate the corresponding environment:

```sh
$ . venv/bin/activate
```

On Windows:

```bat
> venv\Scripts\activate
```

Your shell prompt will change to show the name of the activated
environment.

## Install MRImageTools

Within the activated environment, use the following command to install
MRImageTools:

```sh
$ pip install mrimagetools
```

MRImageTools is now installed. Go to the [Documentation Overview](../index.md).

# Command-line tools

There are some WIP command-line tools. After installing the `mrimagetools`
package, you can find these by running `mrimagetools --help`, or
mrimagetools2 –help. This will list the commands that are available with a short
description. To get more information about a command, run
`mrimagetools <command> --help` or `mrimagetools2 <command> --help`.

The newer commands can be found in `mrimagetools2`, and the older ones in
`mrimagetools`. These will be merged in the future.

# Web UI

The newer commands found in `mrimagetools2` can be accessed through the web browser.

Install the dependencies for the web UI support as follows.

```sh
$ pip install mrimagetools[web]
```

Launch the web server by calling the `mrimagetools.web2` module as follows:

```sh
$ python -m mrimagetools.web2
```

# Development

Development of this software project must comply with a few code styling/quality rules and processes:


* Before pushing any code, make sure the CHANGELOG.md is updated as per the instructions in the CHANGELOG.md file. tox should also be run to ensure that tests and code-quality checks pass.


* Ensure that a good level of test coverage is kept. The test reports will be committed to the CI system when testing is run, and these will be made available during code review. If you wish to view test coverage locally, run coverage report.


* To ensure these code quality rules are kept to, [pre-commit]([https://pre-commit.com/](https://pre-commit.com/)) should be installed (see the requirements/dev.txt), and pre-commit install run when first cloning this repo. This will install some pre-commit hooks that will ensure any committed code meets the minimum code-quality and is formatted correctly *before* being committed to Git. mypy, Pylint, black and isort will be run automatically, and results displayed after a git commit. These will also be checked on the GitLab CI system after code is pushed. The tools should also be included in any IDEs/editors used, where possible.


* [mypy]([https://github.com/python/mypy](https://github.com/python/mypy)) should be run on all source code to find any static typing errors.


* [Pylint]([https://pylint.org/](https://pylint.org/)) must be used as the linter for all source code. A linting configuration can be found in .pylintrc. There should be as few linting errors as possible when checking in code. Code score should be kept high (close to 10), and if linting error are occurring frequently where they aren’t expected, relevant changes should be make to the .pylintrc file to reflect these (and then code-reviewed).


* Before committing any files, [black]([https://black.readthedocs.io/en/stable/](https://black.readthedocs.io/en/stable/)) must be run with the default settings in order perform autoformatting on any python files.


* [isort]([https://isort.readthedocs.io/en/latest/](https://isort.readthedocs.io/en/latest/)) should be run on all files before they are committed.
