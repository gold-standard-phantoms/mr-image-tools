"""
A user input validator. Used to initialise the model.
All of the validation rules are contained within this file.
The validator may be used with:
`d = USER_INPUT_VALIDATOR(some_input_dictionary)`
`d` will now contain the input dictionary with any defaults values added.
A ValidationError will be raised if any validation rules fail.
"""
import os
import typing
from copy import deepcopy
from typing import Final, Literal

import jsonschema

from mrimagetools.v2.data.filepaths import GROUND_TRUTH_DATA
from mrimagetools.v2.filters.gkm_filter import GkmFilter
from mrimagetools.v2.utils.general import generate_random_numbers, splitext
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    ValidationError,
    Validator,
    and_validator,
    for_each_validator,
    from_list_validator,
    greater_than_equal_to_validator,
    greater_than_validator,
    isinstance_validator,
    list_of_type_validator,
    non_empty_list_or_tuple_validator,
    of_length_validator,
    or_validator,
    range_inclusive_validator,
    reserved_string_list_validator,
)
from mrimagetools.v2.validators.schemas.index import load_schemas

# String constants
ASL_CONTEXT = "asl_context"
LABEL_TYPE = "label_type"
LABEL_DURATION = "label_duration"
SIGNAL_TIME = "signal_time"
LABEL_EFFICIENCY = "label_efficiency"
LAMBDA_BLOOD_BRAIN = "lambda_blood_brain"
T1_ARTERIAL_BLOOD = "t1_arterial_blood"
GKM_MODEL = "gkm_model"
M0 = "m0"
ECHO_TIME = "echo_time"
REPETITION_TIME = "repetition_time"
ROT_Z = "rot_z"
ROT_Y = "rot_y"
ROT_X = "rot_x"
TRANSL_X = "transl_x"
TRANSL_Y = "transl_y"
TRANSL_Z = "transl_z"
ACQ_MATRIX = "acq_matrix"
ACQ_CONTRAST = "acq_contrast"
DESIRED_SNR = "desired_snr"
RANDOM_SEED = "random_seed"
EXCITATION_FLIP_ANGLE = "excitation_flip_angle"
INVERSION_FLIP_ANGLE = "inversion_flip_angle"
INVERSION_TIME = "inversion_time"
OUTPUT_IMAGE_TYPE = "output_image_type"
MODALITY = "modality"
DEFAULT_GROUND_TRUTH = "hrgt_icbm_2009a_nls_3t"
INTERPOLATION = "interpolation"
BACKGROUND_SUPPRESSION = "background_suppression"
BS_SAT_PULSE_TIME = "sat_pulse_time"
BS_INV_PULSE_TIMES = "inv_pulse_times"
BS_PULSE_EFFICIENCY = "pulse_efficiency"
BS_T1_OPT = "t1_opt"
BS_SAT_PULSE_TIME_OPT = "sat_pulse_time_opt"
BS_NUM_INV_PULSES = "num_inv_pulses"
BS_APPLY_TO_ASL_CONTEXT = "apply_to_asl_context"

TRANSFORMATION_PARAMS = [ROT_X, ROT_Y, ROT_Z, TRANSL_X, TRANSL_Y, TRANSL_Z]
ARRAY_PARAMS = TRANSFORMATION_PARAMS + [ECHO_TIME, REPETITION_TIME]


# Creates a validator which checks a parameter is the same
# length as the number of entries in asl_context
asl_context_length_validator_generator = lambda other: Validator(
    func=lambda d: ASL_CONTEXT in d
    and other in d
    and len(d[ASL_CONTEXT].split()) == len(d[other]),
    criteria_message=(
        f"{other} must be present and have the same number of entries as {ASL_CONTEXT}"
    ),
)


# Supported image types
SupportedImageTypes = Literal["asl", "ground_truth", "structural"]
GROUND_TRUTH: SupportedImageTypes = "ground_truth"
ASL: SupportedImageTypes = "asl"
STRUCTURAL: SupportedImageTypes = "structural"
SUPPORTED_IMAGE_TYPES: Final[tuple] = typing.get_args(SupportedImageTypes)

# Supported asl contexts
M0SCAN: Final[str] = "m0scan"
CONTROL: Final[str] = "control"
LABEL: Final[str] = "label"
SUPPORTED_ASL_CONTEXTS = [M0SCAN, CONTROL, LABEL]

# Suported Interpolation types
LINEAR: Final[str] = "linear"
CONTINUOUS: Final[str] = "continuous"
NEAREST: Final[str] = "nearest"
SUPPORTED_INTERPOLATION_TYPES = [LINEAR, CONTINUOUS, NEAREST]

DEFAULT_ASL_MATRIX = [64, 64, 40]

DEFAULT_BS_PARAMS = {
    "sat_pulse_time": 4.0,
    "sat_pulse_time_opt": 3.98,
    "pulse_efficiency": "ideal",
    "num_inv_pulses": 4,
    "apply_to_asl_context": ["label", "control"],
}
DEFAULT_SUBJECT_LABEL = "001"

# supported structural image series modality labels
SUPPORTED_STRUCT_MODALITY_LABELS = [
    "T1w",
    "T2w",
    "FLAIR",
    "PDw",
    "T2starw",
    "inplaneT1",
    "inplaneT2",
    "PDT2",
    "UNIT1",
]

ASL_ECHO_TIME_DEFAULT = {
    M0SCAN: 0.01,
    CONTROL: 0.01,
    LABEL: 0.01,
}
ASL_REPETITION_TIME_DEFAULT = {
    M0SCAN: 10.0,
    CONTROL: 5.0,
    LABEL: 5.0,
}

ROT_RANGE_VALIDATOR = for_each_validator(range_inclusive_validator(-180, 180))
TRANSL_RANGE_VALIDATOR = for_each_validator(range_inclusive_validator(-1000, 1000))

# Input validator
IMAGE_TYPE_VALIDATOR = {
    GROUND_TRUTH: ParameterValidator(
        parameters={
            ROT_X: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Y: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Z: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            TRANSL_X: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Y: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Z: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            ACQ_MATRIX: Parameter(
                validators=[
                    list_of_type_validator(int),
                    of_length_validator(3),
                    for_each_validator(greater_than_validator(0)),
                ],
                default_value=DEFAULT_ASL_MATRIX,
            ),
            INTERPOLATION: Parameter(
                validators=for_each_validator(
                    from_list_validator(SUPPORTED_INTERPOLATION_TYPES)
                ),
                default_value=[LINEAR, NEAREST],
            ),
        }
    ),
    STRUCTURAL: ParameterValidator(
        parameters={
            ECHO_TIME: Parameter(
                validators=greater_than_validator(0), default_value=0.005
            ),
            REPETITION_TIME: Parameter(
                validators=greater_than_validator(0), default_value=0.3
            ),
            ROT_X: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Y: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Z: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            TRANSL_X: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Y: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Z: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            ACQ_MATRIX: Parameter(
                validators=[
                    list_of_type_validator(int),
                    of_length_validator(3),
                    for_each_validator(greater_than_validator(0)),
                ],
                default_value=[197, 233, 189],
            ),
            ACQ_CONTRAST: Parameter(
                validators=from_list_validator(
                    ["ge", "se", "ir"], case_insensitive=True
                ),
                default_value="se",
            ),
            EXCITATION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=90.0
            ),
            INVERSION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=180.0
            ),
            INVERSION_TIME: Parameter(
                validators=greater_than_equal_to_validator(0.0), default_value=1.0
            ),
            DESIRED_SNR: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=100.0
            ),
            RANDOM_SEED: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=0
            ),
            OUTPUT_IMAGE_TYPE: Parameter(
                validators=from_list_validator(["complex", "magnitude"]),
                default_value="magnitude",
            ),
            MODALITY: Parameter(
                validators=from_list_validator(SUPPORTED_STRUCT_MODALITY_LABELS),
                default_value="T1w",
            ),
            INTERPOLATION: Parameter(
                validators=from_list_validator(SUPPORTED_INTERPOLATION_TYPES),
                default_value=LINEAR,
            ),
        }
    ),
    ASL: ParameterValidator(
        parameters={
            ROT_X: Parameter(
                validators=or_validator(
                    [ROT_RANGE_VALIDATOR, isinstance_validator(dict)]
                ),
                default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
            ),
            ROT_Y: Parameter(
                validators=or_validator(
                    [ROT_RANGE_VALIDATOR, isinstance_validator(dict)]
                ),
                default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
            ),
            ROT_Z: Parameter(
                validators=or_validator(
                    [ROT_RANGE_VALIDATOR, isinstance_validator(dict)]
                ),
                default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
            ),
            TRANSL_X: Parameter(
                validators=or_validator(
                    [TRANSL_RANGE_VALIDATOR, isinstance_validator(dict)]
                ),
                default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
            ),
            TRANSL_Y: Parameter(
                validators=or_validator(
                    [TRANSL_RANGE_VALIDATOR, isinstance_validator(dict)]
                ),
                default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
            ),
            TRANSL_Z: Parameter(
                validators=or_validator(
                    [TRANSL_RANGE_VALIDATOR, isinstance_validator(dict)]
                ),
                default_value={"distribution": "gaussian", "mean": 0.0, "sd": 0.0},
            ),
            ACQ_MATRIX: Parameter(
                validators=[
                    list_of_type_validator(int),
                    of_length_validator(3),
                    for_each_validator(greater_than_validator(0)),
                ],
                default_value=DEFAULT_ASL_MATRIX,
            ),
            LABEL_TYPE: Parameter(
                validators=from_list_validator(
                    ["CASL", "PCASL", "PASL"], case_insensitive=True
                ),
                default_value="pcasl",
            ),
            LABEL_DURATION: Parameter(
                validators=range_inclusive_validator(0, 100), default_value=1.8
            ),
            SIGNAL_TIME: Parameter(
                validators=or_validator(
                    [
                        range_inclusive_validator(0, 100),
                        for_each_validator(range_inclusive_validator(0, 100)),
                    ]
                ),
                default_value=3.6,
            ),
            LABEL_EFFICIENCY: Parameter(
                validators=range_inclusive_validator(0, 1), default_value=0.85
            ),
            LAMBDA_BLOOD_BRAIN: Parameter(
                validators=range_inclusive_validator(0, 1), optional=True
            ),
            T1_ARTERIAL_BLOOD: Parameter(
                validators=range_inclusive_validator(0, 100), optional=True
            ),
            M0: Parameter(validators=greater_than_equal_to_validator(0), optional=True),
            ASL_CONTEXT: Parameter(
                validators=reserved_string_list_validator(
                    ["m0scan", "control", "label"], case_insensitive=True
                ),
                default_value="m0scan control label",
            ),
            ECHO_TIME: Parameter(
                validators=[
                    or_validator(
                        [
                            and_validator(
                                [
                                    list_of_type_validator((int, float)),
                                    non_empty_list_or_tuple_validator(),
                                ]
                            ),
                            isinstance_validator(dict),
                        ]
                    ),
                ],
                default_value=ASL_ECHO_TIME_DEFAULT,
            ),
            REPETITION_TIME: Parameter(
                validators=[
                    or_validator(
                        [
                            and_validator(
                                [
                                    list_of_type_validator((int, float)),
                                    non_empty_list_or_tuple_validator(),
                                ]
                            ),
                            isinstance_validator(dict),
                        ]
                    ),
                ],
                default_value=ASL_REPETITION_TIME_DEFAULT,
            ),
            ACQ_CONTRAST: Parameter(
                validators=from_list_validator(
                    ["ge", "se", "ir"], case_insensitive=True
                ),
                default_value="se",
            ),
            DESIRED_SNR: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=1000.0
            ),
            RANDOM_SEED: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=0
            ),
            EXCITATION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=90.0
            ),
            INVERSION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=180.0
            ),
            INVERSION_TIME: Parameter(
                validators=greater_than_equal_to_validator(0.0), default_value=1.0
            ),
            INTERPOLATION: Parameter(
                validators=from_list_validator(SUPPORTED_INTERPOLATION_TYPES),
                default_value=LINEAR,
            ),
            BACKGROUND_SUPPRESSION: Parameter(
                validators=isinstance_validator((dict, bool)),
                default_value=DEFAULT_BS_PARAMS,
            ),
            OUTPUT_IMAGE_TYPE: Parameter(
                validators=from_list_validator(["complex", "magnitude"]),
                default_value="magnitude",
            ),
            GKM_MODEL: Parameter(
                validators=from_list_validator(
                    [GkmFilter.MODEL_WP, GkmFilter.MODEL_FULL]
                ),
                default_value=GkmFilter.MODEL_FULL,
            ),
        },
    ),
}

ASL_POST_VALIDATOR = ParameterValidator(
    parameters={},
    post_validators=[
        Validator(
            func=lambda d: ASL_CONTEXT in d,
            criteria_message=f"{ASL_CONTEXT} must be supplied",
        )
    ]
    + [
        asl_context_length_validator_generator(param)
        for param in [ECHO_TIME, REPETITION_TIME] + TRANSFORMATION_PARAMS
    ],
)


BS_VALIDATOR = {
    "pulse_times_present": ParameterValidator(
        parameters={
            BS_SAT_PULSE_TIME: Parameter(
                validators=[greater_than_validator(0)], default_value=4.0
            ),
            BS_INV_PULSE_TIMES: Parameter(
                validators=[
                    list_of_type_validator(float),
                    for_each_validator(greater_than_validator(0)),
                ],
            ),
            BS_PULSE_EFFICIENCY: Parameter(
                validators=[
                    isinstance_validator((float, str)),
                    or_validator(
                        [
                            from_list_validator(["realistic", "ideal"]),
                            range_inclusive_validator(-1, 0),
                        ]
                    ),
                ],
                default_value="ideal",
            ),
            BS_APPLY_TO_ASL_CONTEXT: Parameter(
                validators=for_each_validator(
                    from_list_validator(SUPPORTED_ASL_CONTEXTS)
                ),
                default_value=[LABEL, CONTROL],
            ),
        }
    ),
    "pulse_times_omitted": ParameterValidator(
        parameters={
            BS_SAT_PULSE_TIME: Parameter(
                validators=[greater_than_validator(0)], default_value=4.0
            ),
            BS_PULSE_EFFICIENCY: Parameter(
                validators=isinstance_validator((float, str)), default_value="ideal"
            ),
            BS_T1_OPT: Parameter(
                validators=[
                    for_each_validator(isinstance_validator(float)),
                    for_each_validator(greater_than_validator(0)),
                ],
                optional=True,
            ),
            BS_SAT_PULSE_TIME_OPT: Parameter(
                validators=greater_than_validator(0), optional=True
            ),
            BS_NUM_INV_PULSES: Parameter(
                validators=[isinstance_validator(int), greater_than_validator(0)],
                default_value=4,
            ),
            BS_APPLY_TO_ASL_CONTEXT: Parameter(
                validators=for_each_validator(
                    from_list_validator(SUPPORTED_ASL_CONTEXTS)
                ),
                default_value=[LABEL, CONTROL],
            ),
        }
    ),
}

DISTRIBUTION_VALIDATOR = {
    "gaussian": ParameterValidator(
        parameters={
            "mean": Parameter(validators=isinstance_validator((int, float))),
            "sd": Parameter(validators=isinstance_validator((int, float))),
            "seed": Parameter(validators=isinstance_validator(int), default_value=0),
        }
    ),
    "uniform": ParameterValidator(
        parameters={
            "min": Parameter(validators=isinstance_validator((int, float))),
            "max": Parameter(validators=isinstance_validator((int, float))),
            "seed": Parameter(validators=isinstance_validator(int), default_value=0),
        }
    ),
}

VALUE_EACH_ASL_CONTEXT_VALIDATOR = {
    "echo_time": ParameterValidator(
        parameters={
            "m0scan": Parameter(validators=isinstance_validator((int, float))),
            "control": Parameter(validators=isinstance_validator((int, float))),
            "label": Parameter(validators=isinstance_validator((int, float))),
        }
    ),
    "repetition_time": ParameterValidator(
        parameters={
            "m0scan": Parameter(validators=isinstance_validator((int, float))),
            "control": Parameter(validators=isinstance_validator((int, float))),
            "label": Parameter(validators=isinstance_validator((int, float))),
        }
    ),
}


def validate_input_params(input_params: dict) -> dict:
    """
    Validate the input parameters
    :param input_params: The input parameters asa Python dict
    :returns: The parsed input parameter dictionary, with any defaults added
    :raises mrimagetools.v2.validators.parameters.ValidationError: if the input
        validation does not pass
    """
    # Check that the input parameters validate against the input parameter schema
    # This checks the general structure of the input, but does not validate the
    # series parameters
    try:
        jsonschema.validate(
            instance=input_params, schema=load_schemas()["input_params"]
        )
    except jsonschema.exceptions.ValidationError as ex:
        # Make the type of exception raised consistent
        raise ValidationError from ex

    validated_input_params = deepcopy(input_params)
    # For every image series
    for _, image_series in enumerate(validated_input_params["image_series"]):
        # Perform the parameter validation based on the 'series_type'
        # (and insert defaults)
        if "series_parameters" not in image_series:
            image_series["series_parameters"] = {}
        image_series["series_parameters"] = IMAGE_TYPE_VALIDATOR[
            image_series["series_type"]
        ].validate(image_series["series_parameters"])
        series_params = image_series["series_parameters"]

        # for image series ASL some extra validation/generation is required
        if image_series["series_type"] == ASL:
            # Check for any array parameters that need to be dynamically generated
            asl_context = series_params[ASL_CONTEXT].split()
            num_acquisitions = len(asl_context)

            # the transformation parameters should be generated as per the
            # prescribed distributions
            for param in TRANSFORMATION_PARAMS:
                series_params[param] = generate_parameter_distribution(
                    series_params[param], num_acquisitions
                )

            # echo_time and repetition_time should be generated based on the
            # values for each value in asl_context
            for param in [ECHO_TIME, REPETITION_TIME]:
                if isinstance(series_params[param], dict):
                    VALUE_EACH_ASL_CONTEXT_VALIDATOR[param].validate(
                        series_params[param]
                    )
                    series_params[param] = [
                        series_params[param][context] for context in asl_context
                    ]

            # run the post validator
            ASL_POST_VALIDATOR.validate(series_params)
            # if "background_suppression" is True then defaults required, so make a blank dict
            if image_series["series_parameters"].get(BACKGROUND_SUPPRESSION) is True:
                image_series["series_parameters"][
                    BACKGROUND_SUPPRESSION
                ] = DEFAULT_BS_PARAMS

            # if there's no "background_suppression" key then don't do anything
            if image_series["series_parameters"].get(BACKGROUND_SUPPRESSION) not in [
                None,
                False,
            ]:
                # otherwise check whether inversion pulse times are supplied or not
                if (
                    image_series["series_parameters"][BACKGROUND_SUPPRESSION].get(
                        BS_INV_PULSE_TIMES
                    )
                    is not None
                ):
                    # pulses present, validate "pulse_times_present" parameters
                    image_series["series_parameters"][
                        BACKGROUND_SUPPRESSION
                    ] = BS_VALIDATOR["pulse_times_present"].validate(
                        image_series["series_parameters"][BACKGROUND_SUPPRESSION]
                    )
                else:
                    # pulses omitted, validate "pulse_times_omitted" parameters
                    image_series["series_parameters"][
                        BACKGROUND_SUPPRESSION
                    ] = BS_VALIDATOR["pulse_times_omitted"].validate(
                        image_series["series_parameters"][BACKGROUND_SUPPRESSION]
                    )

    # Determine whether the ground truth is a valid filename (and exists)
    # or is a pre-existing dataset in the asldro data
    ground_truth_params = validated_input_params["global_configuration"]["ground_truth"]

    if isinstance(ground_truth_params, dict):
        # The input is already a dict with the filename included, so don't do anything
        pass
    elif ground_truth_params in GROUND_TRUTH_DATA:
        # The input is a string - use it to look up the relevant files from the
        # included datasets
        # Replace the 'ground_truth' with the paths to the nii.gz and json files
        validated_input_params["global_configuration"]["ground_truth"] = deepcopy(
            GROUND_TRUTH_DATA[ground_truth_params]
        )
    else:
        # Assume the ground_truth_str is a path to the nifti file, and there is an
        # associated json file
        if not ground_truth_params.endswith((".nii", ".nii.gz")):
            raise ValidationError(
                f"The ground truth {ground_truth_params} must be one of: "
                f"{'. '.join(GROUND_TRUTH_DATA.keys())} or be a .nii or .nii.gz file"
            )
        validated_input_params["global_configuration"]["ground_truth"] = {
            "nii_file": ground_truth_params,
            "json_file": splitext(ground_truth_params)[0] + ".json",
        }

    ground_truth_dict = validated_input_params["global_configuration"]["ground_truth"]
    for filetype in ["json_file", "nii_file"]:
        if not (
            os.path.exists(ground_truth_dict[filetype])
            and os.path.isfile(ground_truth_dict[filetype])
        ):
            raise ValidationError(
                f"Ground truth file {ground_truth_dict[filetype]} does not exist"
            )

    # Check the subject label, if it is empty set to the default
    if validated_input_params["global_configuration"].get("subject_label") is None:
        validated_input_params["global_configuration"][
            "subject_label"
        ] = DEFAULT_SUBJECT_LABEL

    return validated_input_params


def get_example_input_params() -> dict:
    """Generate and validate an example input parameter dictionary.
    Will contain one of each supported image type containing the
    default parameters for each.
    :return: the validated input parameter dictionary
    :raises mrimagetools.v2.validators.parameters.ValidationError: if the input
        validation does not pass
    """
    params = validate_input_params(
        {
            "global_configuration": {
                "ground_truth": DEFAULT_GROUND_TRUTH,
                "image_override": {},
                "parameter_override": {},
            },
            "image_series": [
                {
                    "series_type": IMAGE_TYPE,
                    "series_description": f"user description for {IMAGE_TYPE}",
                }
                for IMAGE_TYPE in SUPPORTED_IMAGE_TYPES
            ],
        }
    )
    # validate_input_params returns the ground truth as paths, we want the default params
    # to use the string, so change it back
    params["global_configuration"]["ground_truth"] = DEFAULT_GROUND_TRUTH

    return params


def generate_parameter_distribution(param: dict, length=1) -> list:
    """Generates a list of values based on the supplied distribution
    specification. If the argument param is not a dictionary, its value
    will be returned. Values will be returned rounded to 4 decimal places.

    :param param: Parameter distribution
    :type param: dict
    :param length: number of values to generate, defaults to 1
    :type length: int, optional
    :raises ValidationError: [description]
    :return: [description]
    :rtype: list
    """
    # check if the parameter value is a dictionary
    if isinstance(param, dict):
        if param.get("distribution") not in (
            "gaussian",
            "uniform",
        ):
            raise ValidationError(
                f"Parameter {param} must have key 'distribution' with value"
                f"'gaussian' or 'uniform'. Value is {param.get('distribution')}"
            )

        # validate the dictionary
        param = DISTRIBUTION_VALIDATOR[param["distribution"]].validate(
            param, ValidationError
        )
        # generate the values
        return (
            generate_random_numbers(
                param,
                (length,),
                param["seed"],
            )
            .round(decimals=4)
            .tolist()
        )
    return param
