# Overview

MRImageTools was developed to address the need to test MR image processing pipelines.
A strong emphasis has been placed on ensuring traceability of the developed
code, in particular with respect to testing. The library uses a ‘pipe and filter’ architecture
with ‘filters’ performing data processing, which provides a common interface between processing
blocks.

## How To Contribute

Got a great idea for something to implement in MRImageTools, or maybe you have just
found a bug? Create an issue at
[https://github.com/gold-standard-phantoms/mrimagetools/issues](https://github.com/gold-standard-phantoms/mrimagetools/issues)
to get in touch with the development team and we’ll take it from there.

# Installation

MrImageTools can be installed as a module directly from the python package index. For more information how to use a python package in this
way please see [https://docs.python.org/3/installing/index.html](https://docs.python.org/3/installing/index.html)

## Python Version

We recommend using the latest version of Python. MRImageTools supports Python 3.8 and newer.

## Dependencies

These distributions will be installed automatically when installing MRImageTools.

- [nibabel](https://nipy.org/nibabel/) provides read / write access to some common neuroimaging file formats

- [numpy](https://numpy.org/) provides efficient calculations with arrays and matrices

- [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/) provides an implementation of JSON Schema validation for Python

- [nilearn](https://nipy.org/packages/nilearn/index.html) provides image manipulation tools and statistical learning for neuroimaging data

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

```
$ mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

On Windows:

```
$ py -3 -m venv venv
```

### Activate the environment

Before you work on your project, activate the corresponding environment:

```
$ . venv/bin/activate
```

On Windows:

```
> venv\Scripts\activate
```

Your shell prompt will change to show the name of the activated
environment.

## Install MRImageTools

Within the activated environment, use the following command to install
MRImageTools:

```
$ pip install mrimagetools
```

MRImageTools is now installed. Check out the Quickstart or go to the
Documentation Overview.

# Quickstart

## Getting started

# Development

Development of this software project must comply with a few code styling/quality rules and processes:

- Pylint must be used as the linter for all source code. A linting configuration can be found in `.pylintrc`. There should be no linting errors when checking in code.

- Before committing any files, [black](https://black.readthedocs.io/en/stable/) must be run with the default settings in order perform autoformatting on the python
  files, and [prettier](https://prettier.io/) must be run (.prettierrc in project root) to autoformat JSON files.

- Before pushing any code, make sure the CHANGELOG.md is updated as per the instructions in the CHANGELOG.md file.

- The project’s software development processes must be used ([found here](https://confluence.goldstandardphantoms.com/display/AD/Software+development+processes)).
