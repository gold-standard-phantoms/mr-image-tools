# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Not Versioned

### Changed

- Deverged from ASLDRO as new MRImageTools library
- Merge neuroqa library

### Added

- DWI pipeline
- DWI signal filter
- Nifti data type field validation
- Unit field validation
- Pydantic-based validation/data sanitising
- Generic ground truth loader filter with input validation.
- General maths utility function and filter.
- Filter that assigns values to regions specified in a segmentation mask.
- Filter validation utility function.
- Numeric utility functions for creating 3D gaussian maps.
- Filter that combines fuzzy masks.
- CLI for creating ground truths.
- CLI for combining masks into a segmentation mask.
- Pipeline for creating a QASPER ground truth
- QASPER ground truth
- Filter that simulates background suppression
- Filter for loading in ASL BIDS files
- Filter for performing ASL quantification using both the single subtraction
  model as well as least squares fit to the full GKM.
- "Whitepaper" mode for generating the perfusion signal.
- Filter to calculate the magnetisation transfer ratio
- Filter to load in BIDS files
- Filter to split an image into multiple images
- Pipeline to calculate a magnetisation transfer ratio image
- Filter to calculate the apparent diffusion coefficient
- Pipeline to calculate the apparent diffusion coefficient

### Fixed

- Default parameter output has hrgt string instead of paths.
- Long, unecessary tests removed.
- Resampling corrected to use the input voxel size as part of the
  scaling calculation.

### Changed

- BIDS output now matches 1.5.0 specification
- Changed to readthedocs theme.
- Blood brain partition coefficient can be supplied as a ground truth
  image in addition to a single parameter.
- GkmFilter accepts the blood brain partition coefficient as an image.
- Default interpolation is now linear (was continuous) as this has more
  consistent results
- Unique version number for output DRO data based on git commit hash.

## [2.2.0] - 2020-12-03

### Fixed

- Bug that caused an error if the SNR was set to 0.

### Changed

- Default 'image_type' attribute of image containers is now 'REAL_IMAGE_TYPE'
- PhaseMagnitudeFilter now accepts additional 'image_type'
- Supplied ground truths (hrgt_icbm_2009a_nls_3t and hrgt_icbm_2009a_nls_31.5t)
  have had their transit times changed to 0.8s for Grey Matter and 1.2s for
  White Matter.

## [2.1.0] - 2020-11-27

### Added

- Allow using an input (external) HRGT file.
- Modulation of ground truth quantities by a scale and offset value

### Fixed

- Bug that caused ground truth not to be resampled before being output

### Changed

- JSON loader filter can take any JSON schema (or none)

## [2.0.0] - 2020-11-19

### Added

- Inversion recovery MRI model
- Image container metadata
- Acquire MRI Image Filter (combines MRI Signal, transform/resampling, noise)
- Append metadata filter
- Combine time-series images filter
- Phase-magnitude filter
- CLI tool to output the HRGT files
- BIDS output
- Output of structural and ground truth data
- 1.5T and 3T HRGT files
- Allow overriding the ground truth values in the input parameter file

### Changed

- New input parameter file format (allows multi-series output)
- GE MRI signal model now allows arbitrary flip-angle

## [1.0.0] - 2020-10-16

### Added

- Project repository set up with Continuous integration,
  Continuous testing (pytest, tox), Repository structure,
  Versioning information, Setup configuration,
  Deployment configuration
- Autoformatting and linting rules set up
- Pipe-and-filter architecture
- Ground truth data based on the ICBM 2009a Nonlinear Symmetric brain atlas
- NiftiImageContainer and NumpyImageContainer
- Ground truth JSON validation schema
- Filters for fft and ifft
- Complex noise filter
- Filter blocks (chains of filters)
- Parameter validation module
- GKM filter (PASL, CASL, pCASL)
- MRI signal filter
- Sphinx documentation generation (and documentation)
- Command line interface
- Affine matrix transformation filter (output sampling)
- README generator
- PyPI deployment
- CLI tool to output example model parameter file
