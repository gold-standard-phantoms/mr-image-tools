"""ImageMetadata class"""
from collections.abc import Sequence
from copy import copy
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Extra

from mrimagetools.containers.bids_metadata import BidsMetadata
from mrimagetools.utils.general import (
    SnakeCamelConvertType,
    camel_to_snake_case_keys_converter,
)
from mrimagetools.validators.fields import NiftiDataTypeField, UnitField
from mrimagetools.validators.parameter_model import ParameterModel

AslSingleContext = Literal["m0scan", "control", "label", "cbf"]
AslContext = Union[AslSingleContext, Sequence[AslSingleContext]]
MrAcqType = Literal["1D", "2D", "3D"]
AcqContrastType = Literal["ge", "ir", "se"]

ImageFlavorType = Literal["PERFUSION", "DIFFUSION", "OTHER"]

ImageType = tuple[
    Literal["ORIGINAL", "DERIVED"], Literal["PRIMARY"], str, Literal["NONE", "RCBF"]
]
ComplexImageComponent = Literal["REAL", "IMAGINARY"]

PostLabelDelay = Union[float, None]

Number = Union[float, int]

# Some key conversions between BIDs and ImageMetadata
# It is unneccessary to specify a conversion if only CamelCase to
# snake_case is to be carried out
_BIDS_KEY_CONVERSION: dict[str, str] = {
    "LabelingEfficiency": "label_efficiency",
    "LabelingDuration": "label_duration",
    "FlipAngle": "excitation_flip_angle",
    "ArterialSpinLabelingType": "label_type",
    "AcquisitionVoxelSize": "voxel_size",
    "PostLabelingDelay": "post_label_delay",
    "MRAcquisitionType": "mr_acquisition_type",
    "QuantificationModel": "gkm_model",
    "BloodBrainPartitionCoefficient": "lambda_blood_brain",
}


class Converter(BaseModel):
    """Defines functions to convert metadata from image to BIDS and vice-versa"""

    image_to_bids: Callable
    bids_to_image: Callable


_BIDS_VALUE_CONVERSION: dict[str, Converter] = {
    "ArterialSpinLabelingType": Converter(
        bids_to_image=str.lower, image_to_bids=str.upper
    )
}


class ImageMetadata(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )
    """Image Metadata

    To initialise:

    `metadata = ImageMetadata(series_number=1, series_description="foo")`

    To output a corresponding dict (excluding anything set to None
    (Optional fields are None by default)):

    `metadata.dict(exclude_none=True)`

    To output a corresponding dict (excluding anything not explicitly set). NOTE, if
    a field is explicitly set to None, or a filter changes it and sets it back to None,
    the field will be output with this method:

    `metadata.dict(exclude_unset=True)`"""

    echo_time: Optional[Number] = None
    """Time in ms between the middle of the excitation pulse and the peak of the echo
    produced (kx=0). In the case of segmented k-space, the TE(eff) is the time between
    the middle of the excitation pulse to the peak of the echo that is used to cover
    the center of k-space (i.e.-kx=0, ky=0)."""

    repetition_time: Optional[Union[Number, Sequence[Number]]] = None
    """The period of time in msec between the beginning of a pulse sequence and the
    beginning of the succeeding (essentially identical) pulse sequence."""

    excitation_flip_angle: Optional[Number] = None
    """Steady state angle in degrees to which the magnetic vector is flipped from the
    magnetic vector of the primary field."""

    inversion_flip_angle: Optional[Number] = None
    """Inversion pulse flip angle in degrees. Only used when `acq_contrast`
    is `"ir"`."""

    inversion_time: Optional[Number] = None
    """Time in msec after the middle of inverting RF pulse to middle of excitation
    pulse to detect the amount of longitudinal magnetization."""

    mr_acquisition_type: Optional[MrAcqType] = None
    """Type of sequence readout. Corresponds to DICOM Tag 0018, 0023 MR Acquisition
    Type. Must be one of: "1D", "2D", or "3D"."""

    image_flavor: Optional[Literal["PERFUSION", "DIFFUSION", "OTHER"]] = None
    """A description of the type of image this corresponds with"""

    image_type: Optional[ImageType] = None
    """Image identification characteristics. Corresponds with DICOM Tag (0008,0008).
    Examples:
    ['ORIGINAL', 'PRIMARY', 'OTHER']
    ['ORIGINAL', 'PRIMARY', 'AXIAL']
    ['ORIGINAL', 'PRIMARY']"""

    b_vectors: Optional[Sequence[Sequence[Number]]] = None
    """List of b-vectors, one for each dwi volume. The number of vectors must be the
    same as the number of dwi volumes."""

    b_values: Optional[Sequence[Number]] = None
    """List of b-values, one for each dwi volume. One of these must be equal to 0,
    and the length of values should be the same as the number of dwi volumes."""

    acq_contrast: Optional[AcqContrastType] = None
    """Corresponds with:
    https://dicom.innolitics.com/ciods/enhanced-mr-image/enhanced-mr-image/00089209"""

    series_type: Optional[Literal["asl", "structural", "ground_truth"]] = None
    """An ASL series type"""

    quantity: Optional[str] = None
    """The quantity associated with this image. For example: t1, m0, mtr, adc"""

    modality: Optional[str] = None
    (
        "The image modality. e.g. t1w, t2w, ADCmap. Modality refers to the 'suffix'"
        "in the BIDS spec. See "
        "(https://bids-specification.readthedocs.io/en/stable/04-modality"
        "-specific-files/01-magnetic-resonance-imaging-data.html#anatomy-imaging-data)"
    )

    repetition_time_preparation: Optional[Union[Number, Sequence[Number]]] = None
    (
        """The interval, in seconds, that it takes a preparation pulse block to
        re-appear at the beginning of the succeeding (essentially identical) pulse
        sequence block. The data type number may apply to files from any MRI modality
        concerned with a single value for this field. The data type array provides a
        value for each volume in a 4D dataset and should only be used when the volume
        timing is critical for interpretation of the data, such as in ASL. """
        "Corresponds with"
        "https://bids-specification.readthedocs.io/en/stable/99-appendices/"
        "14-glossary.html#repetitiontimepreparation-metadata"
    )
    units: Optional[UnitField] = None
    """Measurement units. e.g. mm^2/s. Can be set to the empty string to explicitly
    indicate that this is unitless."""

    data_type: Optional[NiftiDataTypeField] = None
    """The data type. See :class:`NiftiDataTypeField` for more information."""

    series_number: Optional[int] = None
    """A number that identifies this Series. Corresponds with DICOM Tag (0020,0011)"""

    series_description: Optional[str] = None
    """User provided description of the Series. Corresponds with DICOM Tag
    (0008,103E)"""

    asl_context: Optional[AslContext] = None
    """In most cases, the ASL timeseries provided by the scanner consist of a series
    of control and label, and optionally m0scan volumes. In this case, the sequence of
    control, label, and m0scan volumes shall be described here"""

    label_type: Optional[Literal["pcasl", "casl", "pasl"]] = None
    """The asl label type"""

    label_duration: Optional[Number] = None
    """Total duration of the labeling pulse train, in seconds, corresponding to the
    temporal width of the labeling bolus for "PCASL" or "CASL". In case all
    control-label volumes (or deltam or CBF) have the same LabelingDuration, a scalar
    must be specified. In case the control-label volumes (or deltam or cbf) have a
    different "LabelingDuration", an array of numbers must be specified, for which any
    m0scan in the timeseries has a "LabelingDuration" of zero. In case an array of
    numbers is provided, its length should be equal to the number of volumes specified
    in *_aslcontext.tsv. Corresponds to DICOM Tag 0018, 9258 ASL Pulse Train
    Duration."""

    post_label_delay: Optional[Union[PostLabelDelay, Sequence[PostLabelDelay]]] = None
    """This is the postlabeling delay (PLD) time, in seconds, after the end of the
    labeling (for "CASL" or "PCASL") or middle of the labeling pulse (for "PASL")
    until the middle of the excitation pulse applied to the imaging slab
    (for 3D acquisition) or first slice (for 2D acquisition). Can be a number (for a
    single-PLD time series) or an array of numbers (for multi-PLD and Look-Locker). In
    the latter case, the array of numbers contains the PLD of each volume, namely each
    control and label, in the acquisition order. Any image within the time-series
    without a PLD, for example an m0scan, is indicated by a zero. Based on DICOM Tags
    0018, 9079 Inversion Times and 0018, 0082 InversionTime."""

    label_efficiency: Optional[Number] = None
    """Labeling efficiency, specified as a number between zero and one, only if
    obtained externally (for example phase-contrast based)."""

    lambda_blood_brain: Optional[Number] = None
    """The brain–blood partition coefficient."""

    t1_arterial_blood: Optional[Number] = None
    """The T1 of arterial blood"""

    voxel_size: Optional[Sequence[Number]] = None
    """An array of numbers with a length of 3, in millimeters. This parameter denotes
    the original acquisition voxel size, excluding any inter-slice gaps and before any
    interpolation or resampling within reconstruction or image processing. Any point
    spread function effects, for example due to T2-blurring, that would decrease the
    effective resolution are not considered here."""

    magnetic_field_strength: Optional[Number] = None
    """Nominal field strength of MR magnet in Tesla. Corresponds to DICOM Tag
    0018,0087 Magnetic Field Strength."""

    multiphase_index: Optional[Union[int, Sequence[int]]] = None
    """Array of the index in the multiphase loop when the volume was acquired. Only
    required if ``post_label_delay`` is a list and has length > 1"""

    m0: Optional[Union[float, int]] = None
    """A numerical (constant) value for M0"""

    bolus_cut_off_flag: Optional[bool] = None
    """Boolean indicating if a bolus cut-off technique is used. Corresponds to DICOM
    Tag 0018, 925C ASL Bolus Cut-off Flag."""

    bolus_cut_off_delay_time: Optional[Number] = None
    """Duration between the end of the labeling and the start of the bolus cut-off
    saturation pulse(s), in seconds. This can be a number or array of numbers, of which
    the values must be non-negative and monotonically increasing, depending on the
    number of bolus cut-off saturation pulses. For Q2TIPS, only the values for the
    first and last bolus cut-off saturation pulses are provided. Based on DICOM Tag
    0018, 925F ASL Bolus Cut-off Delay Time."""

    gkm_model: Optional[Literal["full", "whitepaper"]] = None
    """The model used to solve the general kinetic model (GKM). Either:
    `full`: full solutions to the GKM :cite:p:`Buxton1998`
    `whitepaper`: The simplified model, derived from the single subtraction
    quantification equations
    (see :class:`mrimagetools.filters.AslQuantificationFilter`)"""

    m0_type: Optional[Literal["Included", "Estimate", "Absent"]] = None
    (
        """Describes the presence of M0 information. "Separate" means that a separate
    image is present. "Included" means that an m0scan volume is contained within the
    current image. "Estimate" means that a single whole-brain M0 value is provided.
    "Absent" means that no specific M0 information is present. """
        "See: https://bids-specification.readthedocs.io/en/stable/99-appendices/"
        "14-glossary.html#m0type-metadata"
    )

    estimation_algorithm: Optional[str] = None
    (
        "The kind of algorithm used. Roughly corresponds with"
        "https://bids-specification.readthedocs.io/en/stable/99-appendices/"
        "11-qmri.html#metadata-requirements-for-qmri-maps"
    )

    segmentation: Optional[Union[str, dict[str, int]]] = None
    """Either the name of the segmentation e.g. "m0" or a dict which represents
    each of the labels. e.g.:

    ```
        segmentation={
            "csf": 3,
            "grey_matter": 1,
            "white_matter": 2,
        },
    ```
    """

    vascular_crushing: Optional[bool] = None
    """Boolean indicating if Vascular Crushing is used. Corresponds to DICOM Tag
    0018, 9259 ASL Crusher Flag."""

    complex_image_component: Optional[ComplexImageComponent] = None
    """Either "REAL" or "IMAGINARY" """

    background_suppression: Optional[bool] = None
    """Boolean indicating if background suppression is used."""

    background_suppression_inv_pulse_timing: Optional[Sequence[Number]] = None
    """The inversion pulse timings."""

    background_suppression_sat_pulse_timing: Optional[Number] = None
    """The time, in seconds between the saturation pulse and the imaging excitation
    pulse. Must be greater than 0."""

    background_suppression_num_pulses: Optional[Number] = None
    """The number of background suppression pulses used. Note that this excludes any
    effect of background suppression pulses applied before the labeling."""

    # Directly from BIDS (after conversion to snake_case)
    acquisition_date_time: Optional[str] = None
    """The date and time that the acquisition of data that resulted in this image
    started."""

    manufacturer: Optional[str] = None
    """Manufacturer of the equipment that produced the composite instances."""

    pulse_sequence_details: Optional[str] = None
    """Information beyond pulse sequence type that identifies the specific pulse
    sequence used (for example, "Standard Siemens Sequence distributed with the VB17
    software", "Siemens WIP ### version #.##," or "Sequence written by X using a
    version compiled on MM/DD/YYYY")."""

    scanning_sequence: Optional[str] = None
    """Description of the type of data acquired. Corresponds to DICOM Tag 0018, 0020
    Scanning Sequence."""

    @classmethod
    def from_bids(cls, bids: dict[str, Any]) -> "ImageMetadata":
        """Converts a BIDS-like dictionary to ImageMetadata"""
        semi_converted = copy(bids)

        # Run any value conversion
        for key, convert_func in _BIDS_VALUE_CONVERSION.items():
            if key in semi_converted:
                semi_converted[key] = convert_func.bids_to_image(semi_converted[key])

        # Convert keys from BIDS
        for old_key, new_key in _BIDS_KEY_CONVERSION.items():
            if old_key in semi_converted:
                semi_converted[new_key] = semi_converted.pop(old_key)
        return ImageMetadata(**camel_to_snake_case_keys_converter(semi_converted))

    def to_bids(self) -> BidsMetadata:
        """Converts ImageMetadata to a BIDS metadata dictionary"""
        to_convert = self.dict(exclude_none=True)
        converted = {}

        # Run any value conversion
        for key, convert_func in _BIDS_VALUE_CONVERSION.items():
            if key in to_convert:
                to_convert[key] = convert_func.image_to_bids(to_convert[key])

        # Convert keys to BIDS
        for new_key, old_key in _BIDS_KEY_CONVERSION.items():
            if old_key in to_convert:
                converted[new_key] = to_convert.pop(old_key)

        return BidsMetadata(
            **converted,
            **camel_to_snake_case_keys_converter(
                to_convert, SnakeCamelConvertType.SNAKE_TO_CAMEL
            )
        )
