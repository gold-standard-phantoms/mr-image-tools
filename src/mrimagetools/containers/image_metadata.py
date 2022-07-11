"""ImageMetadata class"""
from copy import copy
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

from mrimagetools.utils.general import camel_to_snake_case_keys
from mrimagetools.validators.fields import NiftiDataTypeField, UnitField
from mrimagetools.validators.parameter_model import ParameterModel

AslSingleContext = Literal["m0scan", "control", "label", "cbf"]
AslContext = Union[AslSingleContext, Sequence[AslSingleContext]]
MrAcqType = Literal["2D", "3D"]
AcqContrastType = Literal["ge", "ir", "se"]

ImageFlavorType = Literal["PERFUSION", "DIFFUSION", "OTHER"]

ImageType = Tuple[
    Literal["ORIGINAL", "DERIVED"], Literal["PRIMARY"], str, Literal["NONE", "RCBF"]
]
ComplexImageComponent = Literal["REAL"]

PostLabelDelay = Union[float, None]

Number = Union[float, int]


class ImageMetadata(ParameterModel):
    """Image Metadata
    To initialise:
    `metadata = ImageMetadata(series_number=1, series_description="foo")`
    To output a corresponding dict (excluding anything not explicitly set):
    `metadata.dict(exclude_unset=True)`
    To output a corresponding dict (excluding anything set to None
    (Optional field are None by default)):
    `metadata.dict(exclude_none=True)`"""

    echo_time: Optional[Number]
    repetition_time: Optional[Union[Number, Sequence[Number]]]
    excitation_flip_angle: Optional[Number]
    inversion_flip_angle: Optional[Number]
    inversion_time: Optional[Number]
    mr_acquisition_type: Optional[MrAcqType]
    image_flavor: Optional[str]
    image_type: Optional[ImageType]
    b_vectors: Optional[Sequence[Sequence[Number]]]
    b_values: Optional[Sequence[Number]]
    acq_contrast: Optional[AcqContrastType]
    series_type: Optional[Literal["asl", "structural", "ground_truth"]]
    quantity: Optional[str]
    repetition_time_preparation: Optional[Union[Number, Sequence[Number]]]
    units: Optional[UnitField]
    data_type: Optional[NiftiDataTypeField]
    modality: Optional[str]
    series_number: Optional[int]
    series_description: Optional[str]
    asl_context: Optional[AslContext]
    label_type: Optional[Literal["pcasl", "casl", "pasl"]]
    label_duration: Optional[Number]
    post_label_delay: Optional[Union[PostLabelDelay, Sequence[PostLabelDelay]]]
    label_efficiency: Optional[Number]
    lambda_blood_brain: Optional[Number]
    t1_arterial_blood: Optional[Number]
    voxel_size: Optional[Sequence[Number]]
    magnetic_field_strength: Optional[Number]
    multiphase_index: Optional[Sequence[int]]
    m0: Optional[Union[float, int]]
    bolus_cut_off_flag: Optional[bool]
    bolus_cut_off_delay_time: Optional[Number]
    gkm_model: Optional[Literal["full", "whitepaper"]]
    m0_type: Optional[Literal["Included", "Excluded"]]
    estimation_algorithm: Optional[str]

    segmentation: Optional[Union[str, Dict[str, int]]]
    vascular_crushing: Optional[bool]
    complex_image_component: Optional[ComplexImageComponent]

    background_suppression: Optional[bool]
    background_suppression_inv_pulse_timing: Optional[Sequence[Number]]
    background_suppression_sat_pulse_timing: Optional[Number]
    background_suppression_num_pulses: Optional[Number]

    # Directly from BIDS (after conversion to snake_case)
    acquisition_date_time: Optional[str]
    manufacturer: Optional[str]
    pulse_sequence_details: Optional[str]
    scanning_sequence: Optional[str]

    # Some key conversions between BIDs and ImageMetadata
    # It is unneccessary to specify a conversion if only CamelCase to
    # snake_case is to be carried out
    _BIDS_KEY_CONVERSION: Dict[str, str] = {
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
    _BIDS_VALUE_CONVERSION: Dict[str, Callable[[Any], Any]] = {
        "ArterialSpinLabelingType": lambda x: x.lower()
    }

    @classmethod
    def from_bids(cls, bids: Dict[str, Any]) -> "ImageMetadata":
        """Converts a BIDS-like dictionary to ImageMetadata"""
        semi_converted = copy(bids)
        for key, convert_func in cls._BIDS_VALUE_CONVERSION.items():
            if key in semi_converted:
                semi_converted[key] = convert_func(semi_converted[key])
        for old_key, new_key in cls._BIDS_KEY_CONVERSION.items():
            if old_key in semi_converted:
                bids_value = semi_converted.pop(old_key)
                semi_converted[new_key] = bids_value
        return ImageMetadata(**camel_to_snake_case_keys(semi_converted))
