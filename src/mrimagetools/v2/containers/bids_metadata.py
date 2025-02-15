# Generated using BIDS-pydantic v0.0.3 using BIDS schema v1.8.0
# from https://raw.githubusercontent.com/bids-standard/bids-specification/v1.8.0/src/schema/objects/metadata.yaml
# Uses datamodel-code-generator v0.21.1
from __future__ import annotations

from datetime import date, time
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import AnyUrl, BaseModel, Field, RootModel


class AcquisitionVoxelSizeItem(RootModel):
    root: Annotated[float, Field(gt=0.0)]


class AnatomicalLandmarkCoordinateUnits(Enum):
    """
    Units of the coordinates of `"AnatomicalLandmarkCoordinateSystem"`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class ArterialSpinLabelingType(Enum):
    """
    The arterial spin labeling type.

    """

    CASL = "CASL"
    PCASL = "PCASL"
    PASL = "PASL"


class BackgroundSuppressionPulseTimeItem(RootModel):
    root: Annotated[float, Field(ge=0.0)]


class BolusCutOffDelayTimeItem(RootModel):
    root: Annotated[float, Field(ge=0.0)]


class CASLType(Enum):
    """
    Describes if a separate coil is used for labeling.

    """

    single_coil = "single-coil"
    double_coil = "double-coil"


class ChunkTransformationMatrixItem(RootModel):
    root: Annotated[list[Any], Field(max_length=3, min_length=3)]


class ChunkTransformationMatrixItem1(RootModel):
    root: Annotated[list[Any], Field(max_length=4, min_length=4)]


class ContrastBolusIngredient(Enum):
    """
    Active ingredient of agent.
    Corresponds to DICOM Tag 0018, 1048 `Contrast/Bolus Ingredient`.

    """

    IODINE = "IODINE"
    GADOLINIUM = "GADOLINIUM"
    CARBON_DIOXIDE = "CARBON DIOXIDE"
    BARIUM = "BARIUM"
    XENON = "XENON"


class DatasetType(Enum):
    """
    The interpretation of the dataset.
    For backwards compatibility, the default value is `"raw"`.

    """

    raw = "raw"
    derivative = "derivative"


class DetectorTypeEnum(Enum):
    mixed = "mixed"


class DigitizedHeadPointsCoordinateUnits(Enum):
    """
    Units of the coordinates of `"DigitizedHeadPointsCoordinateSystem"`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class EEGCoordinateUnits(Enum):
    """
    Units of the coordinates of `EEGCoordinateSystem`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class EchoTimeItem(RootModel):
    root: Annotated[float, Field(gt=0.0)]


class FiducialsCoordinateUnits(Enum):
    """
    Units in which the coordinates that are  listed in the field
    `"FiducialsCoordinateSystem"` are represented.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class FlipAngleItem(RootModel):
    root: Annotated[float, Field(gt=0.0, le=360.0)]


class Container(BaseModel):
    Type: Optional[str] = None
    Tag: Optional[str] = None
    URI: Optional[AnyUrl] = None


class GeneratedByItem(BaseModel):
    Name: Optional[str] = None
    Version: Optional[str] = None
    Description: Optional[str] = None
    CodeURL: Optional[AnyUrl] = None
    Container: Optional[Container] = None


class Genetics(BaseModel):
    """
    An object containing information about the genetics descriptor.

    """

    Database: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\nof"
                " database where the dataset is hosted.\n"
            )
        ),
    ] = None
    Dataset: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\nwhere"
                " data can be retrieved.\n"
            )
        ),
    ] = None
    Descriptors: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "List of relevant descriptors (for example, journal articles) for"
                " dataset\nusing a"
                " valid\n[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\nwhen"
                " possible.\n"
            )
        ),
    ] = None


class HardwareFilter(Enum):
    n_a = "n/a"


class HeadCoilCoordinateUnits(Enum):
    """
    Units of the coordinates of `HeadCoilCoordinateSystem`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class InjectedMas(Enum):
    n_a = "n/a"


class InjectedMassUnit(Enum):
    n_a = "n/a"


class LabelingDurationItem(RootModel):
    root: Annotated[float, Field(ge=0.0)]


class M0Type(Enum):
    """
    Describes the presence of M0 information.
    `"Separate"` means that a separate `*_m0scan.nii[.gz]` is present.
    `"Included"` means that an m0scan volume is contained within the current
    `*_asl.nii[.gz]`.
    `"Estimate"` means that a single whole-brain M0 value is provided.
    `"Absent"` means that no specific M0 information is present.

    """

    Separate = "Separate"
    Included = "Included"
    Estimate = "Estimate"
    Absent = "Absent"


class MEGCoordinateUnits(Enum):
    """
    Units of the coordinates of `"MEGCoordinateSystem"`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class MRAcquisitionType(Enum):
    """
    Type of sequence readout.
    Corresponds to DICOM Tag 0018, 0023 `MR Acquisition Type`.

    """

    field_2D = "2D"
    field_3D = "3D"


class MTPulseShape(Enum):
    """
    Shape of the magnetization transfer RF pulse waveform.
    The value `"GAUSSHANN"` refers to a Gaussian pulse with a Hanning window.
    The value `"SINCHANN"` refers to a sinc pulse with a Hanning window.
    The value `"SINCGAUSS"` refers to a sinc pulse with a Gaussian window.

    """

    HARD = "HARD"
    GAUSSIAN = "GAUSSIAN"
    GAUSSHANN = "GAUSSHANN"
    SINC = "SINC"
    SINCHANN = "SINCHANN"
    SINCGAUSS = "SINCGAUSS"
    FERMI = "FERMI"


class MeasurementToolMetadata(BaseModel):
    """
    A description of the measurement tool as a whole.
    Contains two fields: `"Description"` and `"TermURL"`.
    `"Description"` is a free text description of the measurement tool.
    `"TermURL"` is a URL to an entity in an ontology corresponding to this tool.

    """

    TermURL: Optional[AnyUrl] = None
    Description: Optional[str] = None


class NIRSCoordinateUnits(Enum):
    """
    Units of the coordinates of `NIRSCoordinateSystem`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class PCASLType(Enum):
    """
    The type of gradient pulses used in the `control` condition.

    """

    balanced = "balanced"
    unbalanced = "unbalanced"


class PhaseEncodingDirection(Enum):
    """
    The letters `i`, `j`, `k` correspond to the first, second and third axis of
    the data in the NIFTI file.
    The polarity of the phase encoding is assumed to go from zero index to
    maximum index unless `-` sign is present
    (then the order is reversed - starting from the highest index instead of
    zero).
    `PhaseEncodingDirection` is defined as the direction along which phase is was
    modulated which may result in visible distortions.
    Note that this is not the same as the DICOM term
    `InPlanePhaseEncodingDirection` which can have `ROW` or `COL` values.

    """

    i = "i"
    j = "j"
    k = "k"
    i_ = "i-"
    j_ = "j-"
    k_ = "k-"


class PixelSizeItem(RootModel):
    root: Annotated[float, Field(ge=0.0)]


class PixelSizeUnits(Enum):
    """
    Unit format of the specified `"PixelSize"`. MUST be one of: `"mm"` (millimeter), `"um"`
    (micrometer) or `"nm"` (nanometer).

    """

    mm = "mm"
    um = "um"
    nm = "nm"


class PostLabelingDelayItem(RootModel):
    root: Annotated[float, Field(gt=0.0)]


class PowerLineFrequencyItem(RootModel):
    root: Annotated[float, Field(gt=0.0)]


class PowerLineFrequencyEnum(Enum):
    n_a = "n/a"


class RecordingType(Enum):
    """
    Defines whether the recording is `"continuous"`, `"discontinuous"`, or
    `"epoched"`, where `"epoched"` is limited to time windows about events of
    interest (for example, stimulus presentations or subject responses).

    """

    continuous = "continuous"
    epoched = "epoched"
    discontinuous = "discontinuous"


class RepetitionTimePreparationItem(RootModel):
    root: Annotated[float, Field(ge=0.0)]


class SampleEnvironment(Enum):
    """
    Environment in which the sample was imaged. MUST be one of: `"in vivo"`, `"ex vivo"`
    or `"in vitro"`.

    """

    in_vivo = "in vivo"
    ex_vivo = "ex vivo"
    in_vitro = "in vitro"


class SampleOrigin(Enum):
    """
    Describes from which tissue the genetic information was extracted.

    """

    blood = "blood"
    saliva = "saliva"
    brain = "brain"
    csf = "csf"
    breast_milk = "breast milk"
    bile = "bile"
    amniotic_fluid = "amniotic fluid"
    other_biospecimen = "other biospecimen"


class SamplingFrequencyNir(Enum):
    n_a = "n/a"


class ScatterFractionItem(RootModel):
    root: Annotated[float, Field(ge=0.0, le=100.0)]


class SliceEncodingDirection(Enum):
    """
    The axis of the NIfTI data along which slices were acquired,
    and the direction in which `"SliceTiming"` is defined with respect to.
    `i`, `j`, `k` identifiers correspond to the first, second and third axis of
    the data in the NIfTI file.
    A `-` sign indicates that the contents of `"SliceTiming"` are defined in
    reverse order - that is, the first entry corresponds to the slice with the
    largest index, and the final entry corresponds to slice index zero.
    When present, the axis defined by `"SliceEncodingDirection"` needs to be
    consistent with the `slice_dim` field in the NIfTI header.
    When absent, the entries in `"SliceTiming"` must be in the order of increasing
    slice index as defined by the NIfTI header.

    """

    i = "i"
    j = "j"
    k = "k"
    i_ = "i-"
    j_ = "j-"
    k_ = "k-"


class SliceTimingItem(RootModel):
    root: Annotated[float, Field(ge=0.0)]


class SoftwareFilter(Enum):
    n_a = "n/a"


class SourceDataset(BaseModel):
    URL: Optional[AnyUrl] = None
    DOI: Optional[str] = None
    Version: Optional[str] = None


class SpatialReferenceEnum(Enum):
    orig = "orig"


class SpecificRadioactivityEnum(Enum):
    n_a = "n/a"


class SpecificRadioactivityUnit(Enum):
    n_a = "n/a"


class SpoilingType(Enum):
    """
    Specifies which spoiling method(s) are used by a spoiled sequence.

    """

    RF = "RF"
    GRADIENT = "GRADIENT"
    COMBINED = "COMBINED"


class StimulusPresentation(BaseModel):
    """
    Object containing key-value pairs related to the software used to present
    the stimuli during the experiment, specifically:
    `"OperatingSystem"`, `"SoftwareName"`, `"SoftwareRRID"`, `"SoftwareVersion"` and
    `"Code"`.
    See table below for more information.

    """

    OperatingSystem: Optional[Any] = None
    SoftwareName: Optional[Any] = None
    SoftwareRRID: Optional[Any] = None
    SoftwareVersion: Optional[Any] = None
    Code: Optional[Any] = None


class TissueOrigin(Enum):
    """
    Describes the type of tissue analyzed for `"SampleOrigin"` `brain`.

    """

    gray_matter = "gray matter"
    white_matter = "white matter"
    csf = "csf"
    meninges = "meninges"
    macrovascular = "macrovascular"
    microvascular = "microvascular"


class Type(Enum):
    """
    Short identifier of the mask.
    The value `"Brain"` refers to a brain mask.
    The value `"Lesion"` refers to a lesion mask.
    The value `"Face"` refers to a face mask.
    The value `"ROI"` refers to a region of interest mask.

    """

    Brain = "Brain"
    Lesion = "Lesion"
    Face = "Face"
    ROI = "ROI"


class FieldCoordUnits(Enum):
    m = "m"
    mm = "mm"
    cm = "cm"
    n_a = "n/a"


class FieldEEGCoordSys(Enum):
    CapTrak = "CapTrak"
    EEGLAB = "EEGLAB"
    EEGLAB_HJ = "EEGLAB-HJ"
    Other = "Other"


class FieldGeneticLevelEnum(Enum):
    Genetic = "Genetic"
    Genomic = "Genomic"
    Epigenomic = "Epigenomic"
    Transcriptomic = "Transcriptomic"
    Metabolomic = "Metabolomic"
    Proteomic = "Proteomic"


class FieldMEGCoordSys(Enum):
    CTF = "CTF"
    ElektaNeuromag = "ElektaNeuromag"
    field_4DBti = "4DBti"
    KitYokogawa = "KitYokogawa"
    ChietiItab = "ChietiItab"
    Other = "Other"


class FieldStandardTemplateCoordSys(Enum):
    ICBM452AirSpace = "ICBM452AirSpace"
    ICBM452Warp5Space = "ICBM452Warp5Space"
    IXI549Space = "IXI549Space"
    fsaverage = "fsaverage"
    fsaverageSym = "fsaverageSym"
    fsLR = "fsLR"
    MNIColin27 = "MNIColin27"
    MNI152Lin = "MNI152Lin"
    MNI152NLin2009aSym = "MNI152NLin2009aSym"
    MNI152NLin2009bSym = "MNI152NLin2009bSym"
    MNI152NLin2009cSym = "MNI152NLin2009cSym"
    MNI152NLin2009aAsym = "MNI152NLin2009aAsym"
    MNI152NLin2009bAsym = "MNI152NLin2009bAsym"
    MNI152NLin2009cAsym = "MNI152NLin2009cAsym"
    MNI152NLin6Sym = "MNI152NLin6Sym"
    MNI152NLin6ASym = "MNI152NLin6ASym"
    MNI305 = "MNI305"
    NIHPD = "NIHPD"
    OASIS30AntsOASISAnts = "OASIS30AntsOASISAnts"
    OASIS30Atropos = "OASIS30Atropos"
    Talairach = "Talairach"
    UNCInfant = "UNCInfant"


class FieldStandardTemplateDeprecatedCoordSys(Enum):
    fsaverage3 = "fsaverage3"
    fsaverage4 = "fsaverage4"
    fsaverage5 = "fsaverage5"
    fsaverage6 = "fsaverage6"
    fsaveragesym = "fsaveragesym"
    UNCInfant0V21 = "UNCInfant0V21"
    UNCInfant1V21 = "UNCInfant1V21"
    UNCInfant2V21 = "UNCInfant2V21"
    UNCInfant0V22 = "UNCInfant0V22"
    UNCInfant1V22 = "UNCInfant1V22"
    UNCInfant2V22 = "UNCInfant2V22"
    UNCInfant0V23 = "UNCInfant0V23"
    UNCInfant1V23 = "UNCInfant1V23"
    UNCInfant2V23 = "UNCInfant2V23"


class FieldIEEGCoordSys(Enum):
    Pixels = "Pixels"
    ACPC = "ACPC"
    Other = "Other"


class IEEGCoordinateUnits(Enum):
    """
    Units of the `*_electrodes.tsv`.
    MUST be `"pixels"` if `iEEGCoordinateSystem` is `Pixels`.

    """

    m = "m"
    mm = "mm"
    cm = "cm"
    pixels = "pixels"
    n_a = "n/a"


class BidsMetadata(BaseModel):
    """
    This schema contains definitions for all metadata fields (fields which may  appear in sidecar JSON files) currently supported in BIDS.

    """

    ACCELChannelCount: Annotated[
        Optional[int], Field(description="Number of accelerometer channels.\n", ge=0)
    ] = None
    Acknowledgements: Annotated[
        Optional[str],
        Field(
            description=(
                "Text acknowledging contributions of individuals or institutions"
                " beyond\nthose listed in Authors or Funding.\n"
            )
        ),
    ] = None
    AcquisitionDuration: Annotated[
        Optional[float],
        Field(
            description=(
                "Duration (in seconds) of volume acquisition.\nCorresponds to DICOM Tag"
                " 0018, 9073 `Acquisition Duration`.\nThis field is mutually exclusive"
                ' with `"RepetitionTime"`.\n'
            ),
            gt=0.0,
        ),
    ] = None
    AcquisitionMode: Annotated[
        Optional[str],
        Field(
            description=(
                'Type of acquisition of the PET data (for example, `"list mode"`).\n'
            )
        ),
    ] = None
    AcquisitionVoxelSize: Annotated[
        Optional[list[AcquisitionVoxelSizeItem]],
        Field(
            description=(
                "An array of numbers with a length of 3, in millimeters.\nThis"
                " parameter denotes the original acquisition voxel size,\nexcluding any"
                " inter-slice gaps and before any interpolation or resampling\nwithin"
                " reconstruction or image processing.\nAny point spread function"
                " effects, for example due to T2-blurring,\nthat would decrease the"
                " effective resolution are not considered here.\n"
            ),
            max_length=3,
            min_length=3,
        ),
    ] = None
    Anaesthesia: Annotated[
        Optional[str], Field(description="Details of anaesthesia used, if any.\n")
    ] = None
    AnalyticalApproach: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Methodology or methodologies used to analyse the"
                ' `"GeneticLevel"`.\nValues MUST be taken from the\n[database of'
                " Genotypes and"
                " Phenotypes\n(dbGaP)](https://www.ncbi.nlm.nih.gov/gap/advanced)\nunder"
                " /Study/Molecular Data Type (for example, SNP Genotypes (Array)"
                " or\nMethylation (CpG).\n"
            )
        ),
    ] = None
    AnatomicalLandmarkCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the anatomical landmarks.\nSee the"
                " [Coordinate Systems"
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                ' in\n`"AnatomicalLandmarkCoordinateSystemDescription"`.\n'
            )
        ),
    ] = None
    AnatomicalLandmarkCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    AnatomicalLandmarkCoordinateUnits: Annotated[
        Optional[AnatomicalLandmarkCoordinateUnits],
        Field(
            description=(
                'Units of the coordinates of `"AnatomicalLandmarkCoordinateSystem"`.\n'
            )
        ),
    ] = None
    AnatomicalLandmarkCoordinates: Annotated[
        Optional[dict[str, list[float]]],
        Field(
            description=(
                "Key-value pairs of the labels and 3-D digitized locations of"
                " anatomical landmarks,\ninterpreted following the"
                ' `"AnatomicalLandmarkCoordinateSystem"`\n(for example, `{"NAS":'
                ' [12.7,21.3,13.9], "LPA": [5.2,11.3,9.6],\n"RPA":'
                " [20.2,11.3,9.1]}`.\nEach array MUST contain three numeric values"
                " corresponding to x, y, and z\naxis of the coordinate system in that"
                " exact order.\n"
            )
        ),
    ] = None
    AnatomicalLandmarkCoordinates__mri: Annotated[
        Optional[dict[str, list[float]]],
        Field(
            description=(
                "Key-value pairs of any number of additional anatomical landmarks and"
                " their\ncoordinates in voxel units (where first voxel has index"
                " 0,0,0)\nrelative to the associated anatomical MRI\n(for example,"
                ' `{"AC": [127,119,149], "PC": [128,93,141],\n"IH": [131,114,206]}`, or'
                ' `{"NAS": [127,213,139], "LPA": [52,113,96],\n"RPA":'
                " [202,113,91]}`).\nEach array MUST contain three numeric values"
                " corresponding to x, y, and z\naxis of the coordinate system in that"
                " exact order.\n"
            )
        ),
    ] = None
    ArterialSpinLabelingType: Annotated[
        Optional[ArterialSpinLabelingType],
        Field(description="The arterial spin labeling type.\n"),
    ] = None
    AssociatedEmptyRoom: Annotated[
        Optional[Union[list[str], str]],
        Field(
            description=(
                "One or more [BIDS"
                " URIs](SPEC_ROOT/02-common-principles.md#bids-uri)\npointing to"
                " empty-room file(s) associated with the subject's MEG"
                " recording.\nUsing forward-slash separated paths relative to the"
                " dataset root"
                " is\n[DEPRECATED](SPEC_ROOT/02-common-principles.md#definitions).\n"
            )
        ),
    ] = None
    Atlas: Annotated[
        Optional[str],
        Field(description="Which atlas (if any) was used to generate the mask.\n"),
    ] = None
    AttenuationCorrection: Annotated[
        Optional[str],
        Field(
            description="Short description of the attenuation correction method used.\n"
        ),
    ] = None
    AttenuationCorrectionMethodReference: Annotated[
        Optional[str],
        Field(
            description="Reference paper for the attenuation correction method used.\n"
        ),
    ] = None
    Authors: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "List of individuals who contributed to the creation/curation of the"
                " dataset.\n"
            )
        ),
    ] = None
    B0FieldIdentifier: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "The presence of this key states that this particular 3D or 4D image"
                " MAY be\nused for fieldmap estimation purposes.\nEach"
                ' `"B0FieldIdentifier"` MUST be a unique string within one'
                " participant's tree,\nshared only by the images meant to be used as"
                " inputs for the estimation of a\nparticular instance of the"
                " *B<sub>0</sub> field* estimation.\nIt is RECOMMENDED to derive this"
                " identifier from DICOM Tags, for example,\nDICOM tag 0018, 1030"
                " `Protocol Name`, or DICOM tag 0018, 0024 `Sequence Name`\nwhen the"
                " former is not defined (for example, in GE devices.)\n"
            )
        ),
    ] = None
    B0FieldSource: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                'At least one existing `"B0FieldIdentifier"` defined by images in'
                " the\nparticipant's tree.\nThis field states the *B<sub>0</sub>"
                ' field* estimation designated by the\n`"B0FieldIdentifier"` that may'
                " be used to correct the dataset for distortions\ncaused by"
                ' B<sub>0</sub> inhomogeneities.\n`"B0FieldSource"` and'
                ' `"B0FieldIdentifier"` MAY both be present for images that\nare used'
                " to estimate their own B<sub>0</sub> field, for example, in"
                ' "pepolar"\nacquisitions.\n'
            )
        ),
    ] = None
    BIDSVersion: Annotated[
        Optional[str],
        Field(description="The version of the BIDS standard that was used.\n"),
    ] = None
    BackgroundSuppression: Annotated[
        Optional[bool],
        Field(description="Boolean indicating if background suppression is used.\n"),
    ] = None
    BackgroundSuppressionNumberPulses: Annotated[
        Optional[float],
        Field(
            description=(
                "The number of background suppression pulses used.\nNote that this"
                " excludes any effect of background suppression pulses applied\nbefore"
                " the labeling.\n"
            ),
            ge=0.0,
        ),
    ] = None
    BackgroundSuppressionPulseTime: Annotated[
        Optional[list[BackgroundSuppressionPulseTimeItem]],
        Field(
            description=(
                "Array of numbers containing timing, in seconds,\nof the background"
                " suppression pulses with respect to the start of the\nlabeling.\nIn"
                " case of multi-PLD with different background suppression pulse"
                " times,\nonly the pulse time of the first PLD should be defined.\n"
            )
        ),
    ] = None
    BasedOn: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "List of files in a file collection to generate the map.\nFieldmaps are"
                " also listed, if involved in the processing.\nThis field is"
                " DEPRECATED, and this metadata SHOULD be recorded in the\n`Sources`"
                " field using [BIDS"
                " URIs](SPEC_ROOT/02-common-principles.md#bids-uri)\nto distinguish"
                " sources from different datasets.\n"
            )
        ),
    ] = None
    BloodDensity: Annotated[
        Optional[float],
        Field(
            description=(
                'Measured blood density. Unit of blood density should be in `"g/mL"`.\n'
            )
        ),
    ] = None
    BodyPart: Annotated[
        Optional[str],
        Field(description="Body part of the organ / body region scanned.\n"),
    ] = None
    BodyPartDetails: Annotated[
        Optional[str],
        Field(
            description=(
                'Additional details about body part or location (for example: `"corpus'
                ' callosum"`).\n'
            )
        ),
    ] = None
    BodyPartDetailsOntology: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator) of"
                " ontology used for\nBodyPartDetails (for example:"
                ' `"https://www.ebi.ac.uk/ols/ontologies/uberon"`).\n'
            )
        ),
    ] = None
    BolusCutOffDelayTime: Annotated[
        Optional[Union[BolusCutOffDelayTimeItem, list[BolusCutOffDelayTimeItem]]],
        Field(
            description=(
                "Duration between the end of the labeling and the start of the bolus"
                " cut-off\nsaturation pulse(s), in seconds.\nThis can be a number or"
                " array of numbers, of which the values must be\nnon-negative and"
                " monotonically increasing, depending on the number of bolus\ncut-off"
                " saturation pulses.\nFor Q2TIPS, only the values for the first and"
                " last bolus cut-off saturation\npulses are provided.\nBased on DICOM"
                " Tag 0018, 925F `ASL Bolus Cut-off Delay Time`.\n"
            )
        ),
    ] = None
    BolusCutOffFlag: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean indicating if a bolus cut-off technique is used.\nCorresponds"
                " to DICOM Tag 0018, 925C `ASL Bolus Cut-off Flag`.\n"
            )
        ),
    ] = None
    BolusCutOffTechnique: Annotated[
        Optional[str],
        Field(
            description=(
                'Name of the technique used, for example `"Q2TIPS"`, `"QUIPSS"`,'
                ' `"QUIPSSII"`.\nCorresponds to DICOM Tag 0018, 925E `ASL Bolus Cut-off'
                " Technique`.\n"
            )
        ),
    ] = None
    BrainLocation: Annotated[
        Optional[str],
        Field(
            description=(
                'Refers to the location in space of the `"TissueOrigin"`.\nValues may'
                " be an MNI coordinate,\na label taken from the\n[Allen Brain"
                " Atlas](https://atlas.brain-map.org/atlas?atlas=265297125&plate=\\\n112360888&structure=4392&x=40348.15104166667&y=46928.75&zoom=-7&resolution=\\\n206.60&z=3),\nor"
                " layer to refer to layer-specific gene expression,\nwhich can also tie"
                " up with laminar fMRI.\n"
            )
        ),
    ] = None
    CASLType: Annotated[
        Optional[CASLType],
        Field(description="Describes if a separate coil is used for labeling.\n"),
    ] = None
    CapManufacturer: Annotated[
        Optional[str],
        Field(description='Name of the cap manufacturer (for example, `"EasyCap"`).\n'),
    ] = None
    CapManufacturersModelName: Annotated[
        Optional[str],
        Field(
            description=(
                "Manufacturer's designation of the cap model\n(for example, `\"actiCAP"
                ' 64 Ch Standard-2"`).\n'
            )
        ),
    ] = None
    CellType: Annotated[
        Optional[str],
        Field(
            description=(
                "Describes the type of cell analyzed.\nValues SHOULD come from"
                " the\n[cell ontology](http://obofoundry.org/ontology/cl.html).\n"
            )
        ),
    ] = None
    ChunkTransformationMatrix: Annotated[
        Optional[Union[ChunkTransformationMatrixItem, ChunkTransformationMatrixItem1]],
        Field(
            description=(
                "3x3 or 4x4 affine transformation matrix describing spatial chunk"
                " transformation,\nfor 2D and 3D respectively (for examples: `[[2, 0,"
                " 0], [0, 3, 0], [0, 0, 1]]`\nin 2D for 2x and 3x scaling along the"
                " first and second axis respectively; or\n`[[1, 0, 0, 0], [0, 2, 0, 0],"
                " [0, 0, 3, 0], [0, 0, 0, 1]]` in 3D for 2x and 3x\nscaling along the"
                " second and third axis respectively).\nNote that non-spatial"
                " dimensions like time and channel are not included in"
                " the\ntransformation matrix.\n"
            )
        ),
    ] = None
    ChunkTransformationMatrixAxis: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "Describe the axis of the ChunkTransformationMatrix\n(for examples:"
                ' `["X", "Y"]` or `["Z", "Y", "X"]`).\n'
            ),
            max_length=3,
            min_length=2,
        ),
    ] = None
    Code: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\nof"
                " the code used to present the stimuli.\nPersistent identifiers such as"
                " DOIs are preferred.\nIf multiple versions of code may be hosted at"
                " the same location,\nrevision-specific URIs are recommended.\n"
            )
        ),
    ] = None
    CogAtlasID: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\nof"
                " the corresponding [Cognitive"
                " Atlas](https://www.cognitiveatlas.org/)\nTask term.\n"
            )
        ),
    ] = None
    CogPOID: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\nof"
                " the corresponding [CogPO](http://www.cogpo.org/) term.\n"
            )
        ),
    ] = None
    CoilCombinationMethod: Annotated[
        Optional[str],
        Field(
            description=(
                "Almost all fMRI studies using phased-array coils use"
                " root-sum-of-squares\n(rSOS) combination, but other methods"
                " exist.\nThe image reconstruction is changed by the coil combination"
                " method\n(as for the matrix coil mode above),\nso anything"
                " non-standard should be reported.\n"
            )
        ),
    ] = None
    Columns: Annotated[
        Optional[list[str]], Field(description="Names of columns in file.\n")
    ] = None
    ContinuousHeadLocalization: Annotated[
        Optional[bool],
        Field(
            description=(
                "`true` or `false` value indicating whether continuous head"
                " localisation\nwas performed.\n"
            )
        ),
    ] = None
    ContrastBolusIngredient: Annotated[
        Optional[ContrastBolusIngredient],
        Field(
            description=(
                "Active ingredient of agent.\nCorresponds to DICOM Tag 0018, 1048"
                " `Contrast/Bolus Ingredient`.\n"
            )
        ),
    ] = None
    DCOffsetCorrection: Annotated[
        Optional[str],
        Field(
            description=(
                "A description of the method (if any) used to correct for a DC"
                " offset.\nIf the method used was subtracting the mean value for each"
                ' channel,\nuse "mean".\n'
            )
        ),
    ] = None
    DatasetDOI: Annotated[
        Optional[AnyUrl],
        Field(
            description=(
                "The Digital Object Identifier of the dataset (not the corresponding"
                " paper).\nDOIs SHOULD be expressed as a"
                " valid\n[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator);\nbare"
                " DOIs such as `10.0.2.3/dfjj.10`"
                " are\n[DEPRECATED](SPEC_ROOT/02-common-principles.md#definitions).\n"
            )
        ),
    ] = None
    DatasetLinks: Annotated[
        Optional[dict[str, AnyUrl]],
        Field(
            description=(
                "Used to map a given `<dataset-name>` from a [BIDS"
                " URI](SPEC_ROOT/02-common-principles.md#bids-uri)\nof the form"
                " `bids:<dataset-name>:path/within/dataset` to a local or remote"
                ' location.\nThe `<dataset-name>`: `""` (an empty string) is a reserved'
                " keyword that MUST NOT be a key in\n`DatasetLinks` (example:"
                " `bids::path/within/dataset`).\n"
            )
        ),
    ] = None
    DatasetType: Annotated[
        Optional[DatasetType],
        Field(
            description=(
                "The interpretation of the dataset.\nFor backwards compatibility, the"
                ' default value is `"raw"`.\n'
            )
        ),
    ] = None
    DecayCorrectionFactor: Annotated[
        Optional[list[float]],
        Field(description="Decay correction factor for each frame.\n"),
    ] = None
    DelayAfterTrigger: Annotated[
        Optional[float],
        Field(
            description=(
                "Duration (in seconds) from trigger delivery to scan onset.\nThis delay"
                " is commonly caused by adjustments and loading times.\nThis"
                " specification is entirely independent"
                ' of\n`"NumberOfVolumesDiscardedByScanner"` or'
                ' `"NumberOfVolumesDiscardedByUser"`,\nas the delay precedes the'
                " acquisition.\n"
            )
        ),
    ] = None
    DelayTime: Annotated[
        Optional[float],
        Field(
            description=(
                "User specified time (in seconds) to delay the acquisition of data for"
                " the\nfollowing volume.\nIf the field is not present it is assumed to"
                " be set to zero.\nCorresponds to Siemens CSA header field"
                " `lDelayTimeInTR`.\nThis field is REQUIRED for sparse sequences using"
                ' the `"RepetitionTime"` field\nthat do not have the `"SliceTiming"`'
                ' field set to allowed for accurate\ncalculation of "acquisition'
                ' time".\nThis field is mutually exclusive with `"VolumeTiming"`.\n'
            )
        ),
    ] = None
    Density: Annotated[
        Optional[Union[str, dict[str, str]]],
        Field(
            description=(
                "Specifies the interpretation of the density keyword.\nIf an object is"
                " used, then the keys should be values for the `den` entity\nand values"
                " should be descriptions of those `den` values.\n"
            )
        ),
    ] = None
    Derivative: Annotated[
        Optional[bool],
        Field(
            description=(
                "Indicates that values in the corresponding column are transformations"
                " of values\nfrom other columns (for example a summary score based on a"
                " subset of items in a\nquestionnaire).\n"
            )
        ),
    ] = None
    Description: Annotated[
        Optional[str], Field(description="Free-form natural language description.\n")
    ] = None
    DetectorType: Annotated[
        Optional[Union[str, DetectorTypeEnum]],
        Field(
            description=(
                "Type of detector. This is a free form description with the following"
                ' suggested terms:\n`"SiPD"`, `"APD"`. Preferably a specific model/part'
                " number is supplied.\nIf individual channels have different"
                " `DetectorType`,\nthen the field here should be specified as"
                ' `"mixed"`\nand this column should be included in `optodes.tsv`.\n'
            )
        ),
    ] = None
    DeviceSerialNumber: Annotated[
        Optional[str],
        Field(
            description=(
                "The serial number of the equipment that produced the measurements.\nA"
                " pseudonym can also be used to prevent the equipment from"
                " being\nidentifiable, so long as each pseudonym is unique within the"
                " dataset.\n"
            )
        ),
    ] = None
    DewarPosition: Annotated[
        Optional[str],
        Field(
            description=(
                'Position of the dewar during the MEG scan:\n`"upright"`, `"supine"` or'
                ' `"degrees"` of angle from vertical:\nfor example on CTF systems,'
                ' `"upright=15°, supine=90°"`.\n'
            )
        ),
    ] = None
    DigitizedHeadPoints: Annotated[
        Optional[bool],
        Field(
            description=(
                "`true` or `false` value indicating whether head points outlining"
                " the\nscalp/face surface are contained within this recording.\n"
            )
        ),
    ] = None
    DigitizedHeadPoints__coordsystem: Annotated[
        Optional[str],
        Field(
            description=(
                "Relative path to the file containing the locations of digitized head"
                " points\ncollected during the session (for example,"
                ' `"sub-01_headshape.pos"`).\nRECOMMENDED for all MEG systems,'
                " especially for CTF and BTi/4D.\nFor Elekta/Neuromag the head points"
                " will be stored in the fif file.\n"
            )
        ),
    ] = None
    DigitizedHeadPointsCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the digitized head points.\nSee"
                " the\n[Coordinate Systems"
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                ' in\n`"DigitizedHeadPointsCoordinateSystemDescription"`.\n'
            )
        ),
    ] = None
    DigitizedHeadPointsCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    DigitizedHeadPointsCoordinateUnits: Annotated[
        Optional[DigitizedHeadPointsCoordinateUnits],
        Field(
            description=(
                'Units of the coordinates of `"DigitizedHeadPointsCoordinateSystem"`.\n'
            )
        ),
    ] = None
    DigitizedLandmarks: Annotated[
        Optional[bool],
        Field(
            description=(
                "`true` or `false` value indicating whether anatomical landmark"
                " points\n(fiducials) are contained within this recording.\n"
            )
        ),
    ] = None
    DispersionConstant: Annotated[
        Optional[float],
        Field(
            description=(
                "External dispersion time constant resulting from tubing in default"
                " unit\nseconds.\n"
            )
        ),
    ] = None
    DispersionCorrected: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean flag specifying whether the blood data have been"
                " dispersion-corrected.\nNOTE: not customary for manual samples, and"
                " hence should be set to `false`.\n"
            )
        ),
    ] = None
    DoseCalibrationFactor: Annotated[
        Optional[float],
        Field(
            description=(
                "Multiplication factor used to transform raw data (in counts/sec) to"
                " meaningful unit (Bq/ml).\nCorresponds to DICOM Tag 0054, 1322 `Dose"
                " Calibration Factor`.\n"
            )
        ),
    ] = None
    DwellTime: Annotated[
        Optional[float],
        Field(
            description=(
                "Actual dwell time (in seconds) of the receiver per point in the"
                " readout\ndirection, including any oversampling.\nFor Siemens, this"
                " corresponds to DICOM field 0019, 1018 (in ns).\nThis value is"
                " necessary for the optional readout distortion correction"
                " of\nanatomicals in the HCP Pipelines.\nIt also usefully provides a"
                " handle on the readout bandwidth,\nwhich isn't captured in the other"
                ' metadata tags.\nNot to be confused with `"EffectiveEchoSpacing"`, and'
                " the frequent mislabeling\nof echo spacing (which is spacing in the"
                ' phase encoding direction) as\n"dwell time" (which is spacing in the'
                " readout direction).\n"
            )
        ),
    ] = None
    ECGChannelCount: Annotated[
        Optional[int], Field(description="Number of ECG channels.\n", ge=0)
    ] = None
    ECOGChannelCount: Annotated[
        Optional[int], Field(description="Number of ECoG channels.\n", ge=0)
    ] = None
    EEGChannelCount: Annotated[
        Optional[int],
        Field(
            description=(
                "Number of EEG channels recorded simultaneously (for example, `21`).\n"
            ),
            ge=0,
        ),
    ] = None
    EEGCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the EEG sensors.\n\nSee"
                " the\n[Coordinate Systems"
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                " in\n`EEGCoordinateSystemDescription`.\n"
            )
        ),
    ] = None
    EEGCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    EEGCoordinateUnits: Annotated[
        Optional[EEGCoordinateUnits],
        Field(description="Units of the coordinates of `EEGCoordinateSystem`.\n"),
    ] = None
    EEGGround: Annotated[
        Optional[str],
        Field(
            description=(
                "Description of the location of the ground electrode\n(for example,"
                ' `"placed on right mastoid (M2)"`).\n'
            )
        ),
    ] = None
    EEGPlacementScheme: Annotated[
        Optional[str],
        Field(
            description=(
                "Placement scheme of EEG electrodes.\nEither the name of a standardized"
                ' placement system (for example, `"10-20"`)\nor a list of standardized'
                ' electrode names (for example, `["Cz", "Pz"]`).\n'
            )
        ),
    ] = None
    EEGReference: Annotated[
        Optional[str],
        Field(
            description=(
                "General description of the reference scheme used and (when applicable)"
                " of\nlocation of the reference electrode in the raw recordings\n(for"
                ' example, `"left mastoid"`, `"Cz"`, `"CMS"`).\nIf different channels'
                " have a different reference,\nthis field should have a general"
                " description and the channel specific\nreference should be defined in"
                " the `channels.tsv` file.\n"
            )
        ),
    ] = None
    EMGChannelCount: Annotated[
        Optional[int], Field(description="Number of EMG channels.\n", ge=0)
    ] = None
    EOGChannelCount: Annotated[
        Optional[int], Field(description="Number of EOG channels.\n", ge=0)
    ] = None
    EchoTime: Annotated[
        Optional[Union[EchoTimeItem, list[EchoTimeItem]]],
        Field(
            description=(
                "The echo time (TE) for the acquisition, specified in"
                " seconds.\nCorresponds to DICOM Tag 0018, 0081 `Echo Time`\n(please"
                " note that the DICOM term is in milliseconds not seconds).\nThe data"
                " type number may apply to files from any MRI modality concerned"
                " with\na single value for this field, or to the files in a\n[file"
                " collection](SPEC_ROOT/appendices/file-collections.md)\nwhere the"
                " value of this field is iterated using the\n[`echo`"
                " entity](SPEC_ROOT/appendices/entities.md#echo).\nThe data type array"
                " provides a value for each volume in a 4D dataset and\nshould only be"
                " used when the volume timing is critical for interpretation\nof the"
                " data, such as"
                " in\n[ASL](SPEC_ROOT/04-modality-specific-files/01-magnetic-resonance-imaging-data.md#\\\narterial-spin-labeling-perfusion-data)\nor"
                " variable echo time fMRI sequences.\n"
            )
        ),
    ] = None
    EchoTime1: Annotated[
        Optional[float],
        Field(
            description="The time (in seconds) when the first (shorter) echo occurs.\n",
            gt=0.0,
        ),
    ] = None
    EchoTime2: Annotated[
        Optional[float],
        Field(
            description="The time (in seconds) when the second (longer) echo occurs.\n",
            gt=0.0,
        ),
    ] = None
    EchoTime__fmap: Annotated[
        Optional[float],
        Field(
            description=(
                "The time (in seconds) when the echo corresponding to this map was"
                " acquired.\n"
            ),
            gt=0.0,
        ),
    ] = None
    EffectiveEchoSpacing: Annotated[
        Optional[float],
        Field(
            description=(
                'The "effective" sampling interval, specified in seconds,\nbetween'
                " lines in the phase-encoding direction,\ndefined based on the size of"
                " the reconstructed image in the phase direction.\nIt is frequently,"
                ' but incorrectly, referred to as "dwell time"\n(see the `"DwellTime"`'
                " parameter for actual dwell time).\nIt is required for unwarping"
                " distortions using field maps.\nNote that beyond just in-plane"
                " acceleration,\na variety of other manipulations to the phase encoding"
                " need to be accounted\nfor properly, including partial fourier, phase"
                " oversampling,\nphase resolution, phase field-of-view and"
                " interpolation.\n"
            ),
            gt=0.0,
        ),
    ] = None
    ElectricalStimulation: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean field to specify if electrical stimulation was done during"
                " the\nrecording (options are `true` or `false`). Parameters for"
                " event-like\nstimulation should be specified in the `events.tsv`"
                " file.\n"
            )
        ),
    ] = None
    ElectricalStimulationParameters: Annotated[
        Optional[str],
        Field(
            description=(
                "Free form description of stimulation parameters, such as frequency or"
                " shape.\nSpecific onsets can be specified in the events.tsv"
                " file.\nSpecific shapes can be described here in freeform text.\n"
            )
        ),
    ] = None
    ElectrodeManufacturer: Annotated[
        Optional[str],
        Field(
            description=(
                "Can be used if all electrodes are of the same manufacturer\n(for"
                ' example, `"AD-TECH"`, `"DIXI"`).\nIf electrodes of different'
                " manufacturers are used,\nplease use the corresponding table in the"
                " `_electrodes.tsv` file.\n"
            )
        ),
    ] = None
    ElectrodeManufacturersModelName: Annotated[
        Optional[str],
        Field(
            description=(
                "If different electrode types are used,\nplease use the corresponding"
                " table in the `_electrodes.tsv` file.\n"
            )
        ),
    ] = None
    EpochLength: Annotated[
        Optional[float],
        Field(
            description=(
                "Duration of individual epochs in seconds (for example, `1`)\nin case"
                " of epoched data.\nIf recording was continuous or discontinuous, leave"
                " out the field.\n"
            ),
            ge=0.0,
        ),
    ] = None
    EstimationAlgorithm: Annotated[
        Optional[str],
        Field(
            description=(
                'Type of algorithm used to perform fitting\n(for example, `"linear"`,'
                ' `"non-linear"`, `"LM"` and such).\n'
            )
        ),
    ] = None
    EstimationReference: Annotated[
        Optional[str],
        Field(
            description=(
                "Reference to the study/studies on which the implementation is based.\n"
            )
        ),
    ] = None
    EthicsApprovals: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "List of ethics committee approvals of the research protocols"
                " and/or\nprotocol identifiers.\n"
            )
        ),
    ] = None
    FiducialsCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the fiducials.\nPreferably the same"
                ' as the `"EEGCoordinateSystem"`.\nSee the\n[Coordinate Systems'
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                ' in\n`"FiducialsCoordinateSystemDescription"`.\n'
            )
        ),
    ] = None
    FiducialsCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    FiducialsCoordinateUnits: Annotated[
        Optional[FiducialsCoordinateUnits],
        Field(
            description=(
                "Units in which the coordinates that are  listed in the"
                ' field\n`"FiducialsCoordinateSystem"` are represented.\n'
            )
        ),
    ] = None
    FiducialsCoordinates: Annotated[
        Optional[dict[str, list[float]]],
        Field(
            description=(
                "Key-value pairs of the labels and 3-D digitized position of"
                " anatomical\nlandmarks, interpreted following the"
                ' `"FiducialsCoordinateSystem"`\n(for example, `{"NAS":'
                ' [12.7,21.3,13.9], "LPA": [5.2,11.3,9.6],\n"RPA":'
                " [20.2,11.3,9.1]}`).\nEach array MUST contain three numeric values"
                " corresponding to x, y, and z\naxis of the coordinate system in that"
                " exact order.\n"
            )
        ),
    ] = None
    FiducialsDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of how the fiducials such as vitamin-E"
                " capsules\nwere placed relative to anatomical landmarks,\nand how the"
                ' position of the fiducials were measured\n(for example, `"both with'
                ' Polhemus and with T1w MRI"`).\n'
            )
        ),
    ] = None
    FlipAngle: Annotated[
        Optional[Union[FlipAngleItem, list[FlipAngleItem]]],
        Field(
            description=(
                "Flip angle (FA) for the acquisition, specified in"
                " degrees.\nCorresponds to: DICOM Tag 0018, 1314 `Flip Angle`.\nThe"
                " data type number may apply to files from any MRI modality concerned"
                " with\na single value for this field, or to the files in a\n[file"
                " collection](SPEC_ROOT/appendices/file-collections.md)\nwhere the"
                " value of this field is iterated using the\n[`flip`"
                " entity](SPEC_ROOT/appendices/entities.md#flip).\nThe data type array"
                " provides a value for each volume in a 4D dataset and\nshould only be"
                " used when the volume timing is critical for interpretation of\nthe"
                " data, such as"
                " in\n[ASL](SPEC_ROOT/04-modality-specific-files/01-magnetic-resonance-imaging-data.md#\\\narterial-spin-labeling-perfusion-data)\nor"
                " variable flip angle fMRI sequences.\n"
            )
        ),
    ] = None
    FrameDuration: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "Time duration of each frame in default unit seconds.\nThis corresponds"
                " to DICOM Tag 0018, 1242 `Actual Frame Duration` converted\nto"
                " seconds.\n"
            )
        ),
    ] = None
    FrameTimesStart: Annotated[
        Optional[list[float]],
        Field(
            description=(
                'Start times for all frames relative to `"TimeZero"` in default unit'
                " seconds.\n"
            )
        ),
    ] = None
    Funding: Annotated[
        Optional[list[str]],
        Field(description="List of sources of funding (grant numbers).\n"),
    ] = None
    GeneratedBy: Annotated[
        Optional[list[GeneratedByItem]],
        Field(description="Used to specify provenance of the dataset.\n", min_length=1),
    ] = None
    GeneticLevel: Annotated[
        Optional[Union[Any, list[Any]]],
        Field(
            description=(
                'Describes the level of analysis.\nValues MUST be one of `"Genetic"`,'
                ' `"Genomic"`, `"Epigenomic"`,\n`"Transcriptomic"`, `"Metabolomic"`, or'
                ' `"Proteomic"`.\n'
            )
        ),
    ] = None
    Genetics: Annotated[
        Optional[Genetics],
        Field(
            description=(
                "An object containing information about the genetics descriptor.\n"
            )
        ),
    ] = None
    GradientSetType: Annotated[
        Optional[str],
        Field(
            description=(
                "It should be possible to infer the gradient coil from the scanner"
                " model.\nIf not, for example because of a custom upgrade or use of a"
                " gradient\ninsert set, then the specifications of the actual gradient"
                " coil should be\nreported independently.\n"
            )
        ),
    ] = None
    GYROChannelCount: Annotated[
        Optional[int], Field(description="Number of gyrometer channels.\n", ge=0)
    ] = None
    HED: Annotated[
        Optional[Union[str, dict[str, str]]],
        Field(
            description=(
                "Hierarchical Event Descriptor (HED) information,\nsee the [HED"
                " Appendix](SPEC_ROOT/appendices/hed.md) for details.\n"
            )
        ),
    ] = None
    HEDVersion: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "If HED tags are used:\nThe version of the HED schema used to validate"
                " HED tags for study.\nMay include a single schema or a base schema and"
                " one or more library schema.\n"
            )
        ),
    ] = None
    Haematocrit: Annotated[
        Optional[float],
        Field(
            description=(
                "Measured haematocrit, meaning the volume of erythrocytes divided by"
                " the\nvolume of whole blood.\n"
            )
        ),
    ] = None
    HardcopyDeviceSoftwareVersion: Annotated[
        Optional[str],
        Field(
            description=(
                "Manufacturer's designation of the software of the device that created"
                " this\nHardcopy Image (the printer).\nCorresponds to DICOM Tag 0018,"
                " 101A `Hardcopy Device Software Version`.\n"
            )
        ),
    ] = None
    HardwareFilters: Annotated[
        Optional[Union[dict[str, dict[str, Any]], HardwareFilter]],
        Field(
            description=(
                'Object of temporal hardware filters applied, or `"n/a"` if the data is'
                " not\navailable. Each key-value pair in the JSON object is a name of"
                " the filter and\nan object in which its parameters are defined as"
                ' key-value pairs.\nFor example, `{"Highpass RC filter": {"Half'
                ' amplitude cutoff (Hz)":\n0.0159, "Roll-off": "6dB/Octave"}}`.\n'
            )
        ),
    ] = None
    HeadCircumference: Annotated[
        Optional[float],
        Field(
            description=(
                "Circumference of the participant's head, expressed in cm (for example,"
                " `58`).\n"
            ),
            gt=0.0,
        ),
    ] = None
    HeadCoilCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the head coils.\nSee"
                " the\n[Coordinate Systems"
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                " in\n`HeadCoilCoordinateSystemDescription`.\n"
            )
        ),
    ] = None
    HeadCoilCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the system in"
                " greater detail.\n"
            )
        ),
    ] = None
    HeadCoilCoordinateUnits: Annotated[
        Optional[HeadCoilCoordinateUnits],
        Field(description="Units of the coordinates of `HeadCoilCoordinateSystem`.\n"),
    ] = None
    HeadCoilCoordinates: Annotated[
        Optional[dict[str, list[float]]],
        Field(
            description=(
                "Key-value pairs describing head localization coil labels and"
                " their\ncoordinates, interpreted following the"
                ' `HeadCoilCoordinateSystem`\n(for example, `{"NAS": [12.7,21.3,13.9],'
                ' "LPA": [5.2,11.3,9.6],\n"RPA": [20.2,11.3,9.1]}`).\nNote that coils'
                " are not always placed at locations that have a known\nanatomical name"
                " (for example, for Elekta, Yokogawa systems); in that case\ngeneric"
                ' labels can be used\n(for example, `{"coil1": [12.2,21.3,12.3],'
                ' "coil2": [6.7,12.3,8.6],\n"coil3": [21.9,11.0,8.1]}`).\nEach array'
                " MUST contain three numeric values corresponding to x, y, and z\naxis"
                " of the coordinate system in that exact order.\n"
            )
        ),
    ] = None
    HeadCoilFrequency: Annotated[
        Optional[Union[float, list[float]]],
        Field(
            description=(
                "List of frequencies (in Hz) used by the head localisation"
                " coils\n('HLC' in CTF systems, 'HPI' in Elekta, 'COH' in BTi/4D)\nthat"
                " track the subject's head position in the MEG helmet\n(for example,"
                " `[293, 307, 314, 321]`).\n"
            )
        ),
    ] = None
    HowToAcknowledge: Annotated[
        Optional[str],
        Field(
            description=(
                "Text containing instructions on how researchers using this dataset"
                " should\nacknowledge the original authors.\nThis field can also be"
                " used to define a publication that should be cited in\npublications"
                " that use the dataset.\n"
            )
        ),
    ] = None
    ImageAcquisitionProtocol: Annotated[
        Optional[str],
        Field(
            description=(
                "Description of the image acquisition protocol"
                " or\n[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\n(for"
                " example from [protocols.io](https://www.protocols.io/)).\n"
            )
        ),
    ] = None
    ImageDecayCorrected: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean flag specifying whether the image data have been"
                " decay-corrected.\n"
            )
        ),
    ] = None
    ImageDecayCorrectionTime: Annotated[
        Optional[float],
        Field(
            description=(
                "Point in time from which the decay correction was applied with respect"
                ' to\n`"TimeZero"` in the default unit seconds.\n'
            )
        ),
    ] = None
    Immersion: Annotated[
        Optional[str],
        Field(
            description=(
                "Lens immersion medium. If the file format is OME-TIFF, the value MUST"
                " be consistent\nwith the `Immersion` OME metadata field.\n"
            )
        ),
    ] = None
    InfusionRadioactivity: Annotated[
        Optional[float],
        Field(
            description=(
                "Amount of radioactivity infused into the patient.\nThis value must be"
                " less than or equal to the total injected"
                ' radioactivity\n(`"InjectedRadioactivity"`).\nUnits should be the same'
                ' as `"InjectedRadioactivityUnits"`.\n'
            )
        ),
    ] = None
    InfusionSpeed: Annotated[
        Optional[float], Field(description="If given, infusion speed.\n")
    ] = None
    InfusionSpeedUnits: Annotated[
        Optional[str],
        Field(description='Unit of infusion speed (for example, `"mL/s"`).\n'),
    ] = None
    InfusionStart: Annotated[
        Optional[float],
        Field(
            description=(
                'Time of start of infusion with respect to `"TimeZero"` in the default'
                " unit\nseconds.\n"
            )
        ),
    ] = None
    InjectedMass: Annotated[
        Optional[Union[float, InjectedMas]],
        Field(
            description=(
                "Total mass of radiolabeled compound injected into subject (for"
                " example, `10`).\nThis can be derived as the ratio of the"
                ' `"InjectedRadioactivity"` and\n`"MolarRadioactivity"`.\n**For those'
                " tracers in which injected mass is not available (for example"
                ' FDG)\ncan be set to `"n/a"`)**.\n'
            )
        ),
    ] = None
    InjectedMassPerWeight: Annotated[
        Optional[float], Field(description="Injected mass per kilogram bodyweight.\n")
    ] = None
    InjectedMassPerWeightUnits: Annotated[
        Optional[str],
        Field(
            description=(
                "Unit format of the injected mass per kilogram bodyweight\n(for"
                ' example, `"ug/kg"`).\n'
            )
        ),
    ] = None
    InjectedMassUnits: Annotated[
        Optional[Union[str, InjectedMassUnit]],
        Field(
            description=(
                'Unit format of the mass of compound injected (for example, `"ug"`'
                ' or\n`"umol"`).\n**Note this is not required for an FDG acquisition,'
                ' since it is not available,\nand SHOULD be set to `"n/a"`**.\n'
            )
        ),
    ] = None
    InjectedRadioactivity: Annotated[
        Optional[float],
        Field(
            description=(
                "Total amount of radioactivity injected into the patient (for example,"
                " `400`).\nFor bolus-infusion experiments, this value should be the sum"
                " of all injected\nradioactivity originating from both bolus and"
                " infusion.\nCorresponds to DICOM Tag 0018, 1074 `Radionuclide Total"
                " Dose`.\n"
            )
        ),
    ] = None
    InjectedRadioactivityUnits: Annotated[
        Optional[str],
        Field(
            description=(
                "Unit format of the specified injected radioactivity (for example,"
                ' `"MBq"`).\n'
            )
        ),
    ] = None
    InjectedVolume: Annotated[
        Optional[float],
        Field(description='Injected volume of the radiotracer in the unit `"mL"`.\n'),
    ] = None
    InjectionEnd: Annotated[
        Optional[float],
        Field(
            description=(
                'Time of end of injection with respect to `"TimeZero"` in the default'
                " unit\nseconds.\n"
            )
        ),
    ] = None
    InjectionStart: Annotated[
        Optional[float],
        Field(
            description=(
                'Time of start of injection with respect to `"TimeZero"` in the default'
                " unit\nseconds.\nThis corresponds to DICOM Tag 0018, 1072"
                " `Contrast/Bolus Start Time`\nconverted to seconds relative to"
                ' `"TimeZero"`.\n'
            )
        ),
    ] = None
    InstitutionAddress: Annotated[
        Optional[str],
        Field(
            description=(
                "The address of the institution in charge of the equipment that"
                " produced the\nmeasurements.\n"
            )
        ),
    ] = None
    InstitutionName: Annotated[
        Optional[str],
        Field(
            description=(
                "The name of the institution in charge of the equipment that produced"
                " the\nmeasurements.\n"
            )
        ),
    ] = None
    InstitutionalDepartmentName: Annotated[
        Optional[str],
        Field(
            description=(
                "The department in the institution in charge of the equipment that"
                " produced\nthe measurements.\n"
            )
        ),
    ] = None
    Instructions: Annotated[
        Optional[str],
        Field(
            description=(
                "Text of the instructions given to participants before the recording.\n"
            )
        ),
    ] = None
    IntendedFor: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "The paths to files for which the associated file is intended to be"
                " used.\nContains one or more [BIDS"
                " URIs](SPEC_ROOT/02-common-principles.md#bids-uri).\nUsing"
                " forward-slash separated paths relative to the participant"
                " subdirectory"
                " is\n[DEPRECATED](SPEC_ROOT/02-common-principles.md#definitions).\n"
            )
        ),
    ] = None
    IntendedFor__ds_relative: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "The paths to files for which the associated file is intended to be"
                " used.\nContains one or more [BIDS"
                " URIs](SPEC_ROOT/02-common-principles.md#bids-uri).\nUsing"
                " forward-slash separated paths relative to the dataset root"
                " is\n[DEPRECATED](SPEC_ROOT/02-common-principles.md#definitions).\n"
            )
        ),
    ] = None
    InversionTime: Annotated[
        Optional[float],
        Field(
            description=(
                "The inversion time (TI) for the acquisition, specified in"
                " seconds.\nInversion time is the time after the middle of inverting RF"
                " pulse to middle\nof excitation pulse to detect the amount of"
                " longitudinal magnetization.\nCorresponds to DICOM Tag 0018, 0082"
                " `Inversion Time`\n(please note that the DICOM term is in milliseconds"
                " not seconds).\n"
            ),
            gt=0.0,
        ),
    ] = None
    LabelingDistance: Annotated[
        Optional[float],
        Field(
            description=(
                "Distance from the center of the imaging slab to the center of the"
                " labeling\nplane (`(P)CASL`) or the leading edge of the labeling slab"
                " (`PASL`),\nin millimeters.\nIf the labeling is performed inferior to"
                " the isocenter,\nthis number should be negative.\nBased on DICOM macro"
                " C.8.13.5.14.\n"
            )
        ),
    ] = None
    LabelingDuration: Annotated[
        Optional[Union[LabelingDurationItem, list[LabelingDurationItem]]],
        Field(
            description=(
                "Total duration of the labeling pulse train, in seconds,\ncorresponding"
                ' to the temporal width of the labeling bolus for\n`"PCASL"` or'
                ' `"CASL"`.\nIn case all control-label volumes (or deltam or CBF) have'
                " the same\n`LabelingDuration`, a scalar must be specified.\nIn case"
                " the control-label volumes (or deltam or cbf) have a"
                ' different\n`"LabelingDuration"`, an array of numbers must be'
                " specified,\nfor which any `m0scan` in the timeseries has a"
                ' `"LabelingDuration"` of zero.\nIn case an array of numbers is'
                " provided,\nits length should be equal to the number of volumes"
                " specified in\n`*_aslcontext.tsv`.\nCorresponds to DICOM Tag 0018,"
                " 9258 `ASL Pulse Train Duration`.\n"
            )
        ),
    ] = None
    LabelingEfficiency: Annotated[
        Optional[float],
        Field(
            description=(
                "Labeling efficiency, specified as a number between zero and one,\nonly"
                " if obtained externally (for example phase-contrast based).\n"
            ),
            gt=0.0,
        ),
    ] = None
    LabelingLocationDescription: Annotated[
        Optional[str],
        Field(
            description=(
                'Description of the location of the labeling plane (`"CASL"` or'
                ' `"PCASL"`) or\nthe labeling slab (`"PASL"`) that cannot be captured'
                " by fields\n`LabelingOrientation` or `LabelingDistance`.\nMay include"
                " a link to an anonymized screenshot of the planning of the\nlabeling"
                " slab/plane with respect to the imaging slab or"
                " slices\n`*_asllabeling.jpg`.\nBased on DICOM macro C.8.13.5.14.\n"
            )
        ),
    ] = None
    LabelingOrientation: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "Orientation of the labeling plane (`(P)CASL`) or slab (`PASL`).\nThe"
                " direction cosines of a normal vector perpendicular to the ASL"
                " labeling\nslab or plane with respect to the patient.\nCorresponds to"
                " DICOM Tag 0018, 9255 `ASL Slab Orientation`.\n"
            )
        ),
    ] = None
    LabelingPulseAverageB1: Annotated[
        Optional[float],
        Field(
            description=(
                "The average B1-field strength of the RF labeling pulses, in"
                ' microteslas.\nAs an alternative, `"LabelingPulseFlipAngle"` can be'
                " provided.\n"
            ),
            gt=0.0,
        ),
    ] = None
    LabelingPulseAverageGradient: Annotated[
        Optional[float],
        Field(
            description="The average labeling gradient, in milliteslas per meter.\n",
            gt=0.0,
        ),
    ] = None
    LabelingPulseDuration: Annotated[
        Optional[float],
        Field(
            description=(
                "Duration of the individual labeling pulses, in milliseconds.\n"
            ),
            gt=0.0,
        ),
    ] = None
    LabelingPulseFlipAngle: Annotated[
        Optional[float],
        Field(
            description=(
                "The flip angle of a single labeling pulse, in degrees,\nwhich can be"
                ' given as an alternative to `"LabelingPulseAverageB1"`.\n'
            ),
            gt=0.0,
            le=360.0,
        ),
    ] = None
    LabelingPulseInterval: Annotated[
        Optional[float],
        Field(
            description=(
                "Delay between the peaks of the individual labeling pulses, in"
                " milliseconds.\n"
            ),
            gt=0.0,
        ),
    ] = None
    LabelingPulseMaximumGradient: Annotated[
        Optional[float],
        Field(
            description=(
                "The maximum amplitude of the gradient switched on during the"
                " application of\nthe labeling RF pulse(s), in milliteslas per meter.\n"
            ),
            gt=0.0,
        ),
    ] = None
    LabelingSlabThickness: Annotated[
        Optional[float],
        Field(
            description=(
                "Thickness of the labeling slab in millimeters.\nFor non-selective FAIR"
                " a zero is entered.\nCorresponds to DICOM Tag 0018, 9254 `ASL Slab"
                " Thickness`.\n"
            ),
            gt=0.0,
        ),
    ] = None
    Levels: Annotated[
        Optional[dict[str, str]],
        Field(
            description=(
                "For categorical variables: An object of possible values (keys) and"
                " their\ndescriptions (values).\n"
            )
        ),
    ] = None
    License: Annotated[
        Optional[str],
        Field(
            description=(
                "The license for the dataset.\nThe use of license name abbreviations is"
                " RECOMMENDED for specifying a license\n(see"
                " [Licenses](SPEC_ROOT/appendices/licenses.md)).\nThe corresponding"
                " full license text MAY be specified in an additional\n`LICENSE`"
                " file.\n"
            )
        ),
    ] = None
    LongName: Annotated[
        Optional[str], Field(description="Long (unabbreviated) name of the column.\n")
    ] = None
    LookLocker: Annotated[
        Optional[bool],
        Field(description="Boolean indicating if a Look-Locker readout is used.\n"),
    ] = None
    M0Estimate: Annotated[
        Optional[float],
        Field(
            description=(
                "A single numerical whole-brain M0 value (referring to the M0 of"
                " blood),\nonly if obtained externally\n(for example retrieved from CSF"
                " in a separate measurement).\n"
            ),
            gt=0.0,
        ),
    ] = None
    M0Type: Annotated[
        Optional[M0Type],
        Field(
            description=(
                'Describes the presence of M0 information.\n`"Separate"` means that a'
                ' separate `*_m0scan.nii[.gz]` is present.\n`"Included"` means that an'
                " m0scan volume is contained within the"
                ' current\n`*_asl.nii[.gz]`.\n`"Estimate"` means that a single'
                ' whole-brain M0 value is provided.\n`"Absent"` means that no specific'
                " M0 information is present.\n"
            )
        ),
    ] = None
    MAGNChannelCount: Annotated[
        Optional[int], Field(description="Number of magnetometer channels.\n", ge=0)
    ] = None
    MEGChannelCount: Annotated[
        Optional[int],
        Field(description="Number of MEG channels (for example, `275`).\n", ge=0),
    ] = None
    MEGCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the MEG sensors.\nSee"
                " the\n[Coordinate Systems"
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                ' in\n`"MEGCoordinateSystemDescription"`.\n'
            )
        ),
    ] = None
    MEGCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    MEGCoordinateUnits: Annotated[
        Optional[MEGCoordinateUnits],
        Field(description='Units of the coordinates of `"MEGCoordinateSystem"`.\n'),
    ] = None
    MEGREFChannelCount: Annotated[
        Optional[int],
        Field(
            description=(
                "Number of MEG reference channels (for example, `23`).\nFor systems"
                " without such channels (for example, Neuromag"
                " Vectorview),\n`MEGREFChannelCount` should be set to `0`.\n"
            ),
            ge=0,
        ),
    ] = None
    MRAcquisitionType: Annotated[
        Optional[MRAcquisitionType],
        Field(
            description=(
                "Type of sequence readout.\nCorresponds to DICOM Tag 0018, 0023 `MR"
                " Acquisition Type`.\n"
            )
        ),
    ] = None
    MRTransmitCoilSequence: Annotated[
        Optional[str],
        Field(
            description=(
                "This is a relevant field if a non-standard transmit coil is"
                " used.\nCorresponds to DICOM Tag 0018, 9049 `MR Transmit Coil"
                " Sequence`.\n"
            )
        ),
    ] = None
    MTNumberOfPulses: Annotated[
        Optional[float],
        Field(
            description=(
                "The number of magnetization transfer RF pulses applied before the"
                " readout.\n"
            )
        ),
    ] = None
    MTOffsetFrequency: Annotated[
        Optional[float],
        Field(
            description=(
                "The frequency offset of the magnetization transfer pulse with respect"
                " to the\ncentral H1 Larmor frequency in Hertz (Hz).\n"
            )
        ),
    ] = None
    MTPulseBandwidth: Annotated[
        Optional[float],
        Field(
            description=(
                "The excitation bandwidth of the magnetization transfer pulse in Hertz"
                " (Hz).\n"
            )
        ),
    ] = None
    MTPulseDuration: Annotated[
        Optional[float],
        Field(
            description="Duration of the magnetization transfer RF pulse in seconds.\n"
        ),
    ] = None
    MTPulseShape: Annotated[
        Optional[MTPulseShape],
        Field(
            description=(
                "Shape of the magnetization transfer RF pulse waveform.\nThe value"
                ' `"GAUSSHANN"` refers to a Gaussian pulse with a Hanning window.\nThe'
                ' value `"SINCHANN"` refers to a sinc pulse with a Hanning window.\nThe'
                ' value `"SINCGAUSS"` refers to a sinc pulse with a Gaussian window.\n'
            )
        ),
    ] = None
    MTState: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean stating whether the magnetization transfer pulse is"
                " applied.\nCorresponds to DICOM Tag 0018, 9020 `Magnetization"
                " Transfer`.\n"
            )
        ),
    ] = None
    MagneticFieldStrength: Annotated[
        Optional[float],
        Field(
            description=(
                "Nominal field strength of MR magnet in Tesla.\nCorresponds to DICOM"
                " Tag 0018, 0087 `Magnetic Field Strength`.\n"
            )
        ),
    ] = None
    Magnification: Annotated[
        Optional[float],
        Field(
            description=(
                "Lens magnification (for example: `40`). If the file format is"
                " OME-TIFF,\nthe value MUST be consistent with the"
                ' `"NominalMagnification"` OME metadata field.\n'
            ),
            gt=0.0,
        ),
    ] = None
    Manual: Annotated[
        Optional[bool],
        Field(
            description=(
                "Indicates if the segmentation was performed manually or via an"
                " automated\nprocess.\n"
            )
        ),
    ] = None
    Manufacturer: Annotated[
        Optional[str],
        Field(
            description=(
                "Manufacturer of the equipment that produced the measurements.\n"
            )
        ),
    ] = None
    ManufacturersModelName: Annotated[
        Optional[str],
        Field(
            description=(
                "Manufacturer's model name of the equipment that produced the"
                " measurements.\n"
            )
        ),
    ] = None
    MatrixCoilMode: Annotated[
        Optional[str],
        Field(
            description=(
                "(If used)\nA method for reducing the number of independent channels by"
                " combining in\nanalog the signals from multiple coil elements.\nThere"
                " are typically different default modes when using un-accelerated"
                ' or\naccelerated (for example, `"GRAPPA"`, `"SENSE"`) imaging.\n'
            )
        ),
    ] = None
    MaxMovement: Annotated[
        Optional[float],
        Field(
            description=(
                "Maximum head movement (in mm) detected during the recording,\nas"
                " measured by the head localisation coils (for example, `4.8`).\n"
            )
        ),
    ] = None
    MeasurementToolMetadata: Annotated[
        Optional[MeasurementToolMetadata],
        Field(
            description=(
                "A description of the measurement tool as a whole.\nContains two"
                ' fields: `"Description"` and `"TermURL"`.\n`"Description"` is a free'
                ' text description of the measurement tool.\n`"TermURL"` is a URL to an'
                " entity in an ontology corresponding to this tool.\n"
            )
        ),
    ] = None
    MetaboliteAvail: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean that specifies if metabolite measurements are available.\nIf"
                " `true`, the `metabolite_parent_fraction` column MUST be present in"
                " the\ncorresponding `*_blood.tsv` file.\n"
            )
        ),
    ] = None
    MetaboliteMethod: Annotated[
        Optional[str], Field(description="Method used to measure metabolites.\n")
    ] = None
    MetaboliteRecoveryCorrectionApplied: Annotated[
        Optional[bool],
        Field(
            description=(
                "Metabolite recovery correction from the HPLC, for tracers where it"
                " changes\nwith time postinjection.\nIf `true`, the"
                " `hplc_recovery_fractions` column MUST be present in"
                " the\ncorresponding `*_blood.tsv` file.\n"
            )
        ),
    ] = None
    MiscChannelCount: Annotated[
        Optional[int],
        Field(
            description=(
                "Number of miscellaneous analog channels for auxiliary signals.\n"
            ),
            ge=0,
        ),
    ] = None
    MixingTime: Annotated[
        Optional[float],
        Field(
            description=(
                "In the context of a stimulated- and spin-echo 3D EPI sequence for B1+"
                " mapping,\ncorresponds to the interval between spin- and"
                " stimulated-echo pulses.\nIn the context of a diffusion-weighted"
                " double spin-echo sequence,\ncorresponds to the interval between two"
                " successive diffusion sensitizing\ngradients, specified in seconds.\n"
            )
        ),
    ] = None
    ModeOfAdministration: Annotated[
        Optional[str],
        Field(
            description=(
                'Mode of administration of the injection\n(for example, `"bolus"`,'
                ' `"infusion"`, or `"bolus-infusion"`).\n'
            )
        ),
    ] = None
    MolarActivity: Annotated[
        Optional[float],
        Field(
            description=(
                "Molar activity of compound injected.\nCorresponds to DICOM Tag 0018,"
                " 1077 `Radiopharmaceutical Specific Activity`.\n"
            )
        ),
    ] = None
    MolarActivityMeasTime: Annotated[
        Optional[time],
        Field(
            description=(
                "Time to which molar radioactivity measurement above applies in the"
                ' default\nunit `"hh:mm:ss"`.\n'
            )
        ),
    ] = None
    MolarActivityUnits: Annotated[
        Optional[str],
        Field(
            description=(
                "Unit of the specified molar radioactivity (for example,"
                ' `"GBq/umol"`).\n'
            )
        ),
    ] = None
    MultibandAccelerationFactor: Annotated[
        Optional[float],
        Field(description="The multiband factor, for multiband acquisitions.\n"),
    ] = None
    MultipartID: Annotated[
        Optional[str],
        Field(
            description=(
                "A unique (per participant) label tagging DWI runs that are part of"
                " a\nmultipart scan.\n"
            )
        ),
    ] = None
    Name: Annotated[Optional[str], Field(description="Name of the dataset.\n")] = None
    NegativeContrast: Annotated[
        Optional[bool],
        Field(
            description=(
                "`true` or `false` value specifying whether increasing voxel"
                " intensity\n(within sample voxels) denotes a decreased value with"
                " respect to the\ncontrast suffix.\nThis is commonly the case when"
                " Cerebral Blood Volume is estimated via\nusage of a contrast agent in"
                " conjunction with a T2\\* weighted acquisition\nprotocol.\n"
            )
        ),
    ] = None
    NIRSChannelCount: Annotated[
        Optional[int],
        Field(
            description=(
                "Total number of NIRS channels, including short channels.\nCorresponds"
                " to the number of rows in `channels.tsv` with any NIRS type.\n"
            ),
            ge=0,
        ),
    ] = None
    NIRSSourceOptodeCount: Annotated[
        Optional[int],
        Field(
            description=(
                "Number of NIRS sources.\nCorresponds to the number of rows in"
                ' `optodes.tsv` with type `"source"`.\n'
            ),
            ge=1,
        ),
    ] = None
    NIRSDetectorOptodeCount: Annotated[
        Optional[int],
        Field(
            description=(
                "Number of NIRS detectors.\nCorresponds to the number of rows in"
                ' `optodes.tsv` with type `"detector"`.\n'
            ),
            ge=1,
        ),
    ] = None
    NIRSPlacementScheme: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Placement scheme of NIRS optodes.\nEither the name of a standardized"
                ' placement system (for example, `"10-20"`)\nor an array of'
                ' standardized position names (for example, `["Cz", "Pz"]`).\nThis'
                " field should only be used if a cap was not used.\nIf a standard cap"
                " was used, then it should be specified in `CapManufacturer`\nand"
                ' `CapManufacturersModelName` and this field should be set to `"n/a"`\n'
            )
        ),
    ] = None
    NIRSCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system in which the optode positions are"
                " expressed.\n\nSee\n[Appendix"
                " VIII](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, a'
                " definition of the coordinate system MUST be\nprovided in"
                " `NIRSCoordinateSystemDescription`.\n"
            )
        ),
    ] = None
    NIRSCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    NIRSCoordinateUnits: Annotated[
        Optional[NIRSCoordinateUnits],
        Field(description="Units of the coordinates of `NIRSCoordinateSystem`.\n"),
    ] = None
    NIRSCoordinateProcessingDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Has any post-processing (such as projection) been done on the"
                ' optode\npositions (for example, `"surface_projection"`, `"n/a"`).\n'
            )
        ),
    ] = None
    NonlinearGradientCorrection: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean stating if the image saved has been corrected for"
                " gradient\nnonlinearities by the scanner sequence.\n"
            )
        ),
    ] = None
    NumberOfVolumesDiscardedByScanner: Annotated[
        Optional[int],
        Field(
            description=(
                'Number of volumes ("dummy scans") discarded by the scanner\n(as'
                " opposed to those discarded by the user post hoc)\nbefore saving the"
                " imaging file.\nFor example, a sequence that automatically discards"
                " the first 4 volumes\nbefore saving would have this field as 4.\nA"
                " sequence that does not discard dummy scans would have this set to"
                " 0.\nPlease note that the onsets recorded in the `events.tsv` file"
                " should always\nrefer to the beginning of the acquisition of the first"
                " volume in the\ncorresponding imaging file - independent of the value"
                ' of\n`"NumberOfVolumesDiscardedByScanner"` field.\n'
            ),
            ge=0,
        ),
    ] = None
    NumberOfVolumesDiscardedByUser: Annotated[
        Optional[int],
        Field(
            description=(
                'Number of volumes ("dummy scans") discarded by the user before'
                " including the\nfile in the dataset.\nIf possible, including all of"
                " the volumes is strongly recommended.\nPlease note that the onsets"
                " recorded in the `events.tsv` file should always\nrefer to the"
                " beginning of the acquisition of the first volume in"
                " the\ncorresponding imaging file - independent of the value"
                ' of\n`"NumberOfVolumesDiscardedByUser"` field.\n'
            ),
            ge=0,
        ),
    ] = None
    NumberShots: Annotated[
        Optional[Union[float, list[float]]],
        Field(
            description=(
                "The number of RF excitations needed to reconstruct a slice or"
                " volume\n(may be referred to as partition).\nPlease mind that this is"
                " not the same as Echo Train Length which denotes the\nnumber of"
                " k-space lines collected after excitation in a multi-echo"
                " readout.\nThe data type array is applicable for specifying this"
                " parameter before and\nafter the k-space center is sampled.\nPlease"
                ' see\n[`"NumberShots"` metadata'
                " field]\\\n(SPEC_ROOT/appendices/qmri.md#numbershots-metadata-field)\nin"
                " the qMRI appendix for corresponding calculations.\n"
            )
        ),
    ] = None
    NumericalAperture: Annotated[
        Optional[float],
        Field(
            description=(
                "Lens numerical aperture (for example: `1.4`). If the file format is"
                " OME-TIFF,\nthe value MUST be consistent with the `LensNA` OME"
                " metadata field.\n"
            ),
            gt=0.0,
        ),
    ] = None
    OperatingSystem: Annotated[
        Optional[str],
        Field(
            description=(
                "Operating system used to run the stimuli presentation software\n(for"
                " formatting recommendations, see examples below this table).\n"
            )
        ),
    ] = None
    OtherAcquisitionParameters: Annotated[
        Optional[str],
        Field(
            description="Description of other relevant image acquisition parameters.\n"
        ),
    ] = None
    PASLType: Annotated[
        Optional[str],
        Field(
            description=(
                "Type of the labeling pulse of the `PASL` labeling,\nfor example"
                ' `"FAIR"`, `"EPISTAR"`, or `"PICORE"`.\n'
            )
        ),
    ] = None
    PCASLType: Annotated[
        Optional[PCASLType],
        Field(
            description="The type of gradient pulses used in the `control` condition.\n"
        ),
    ] = None
    ParallelAcquisitionTechnique: Annotated[
        Optional[str],
        Field(
            description=(
                'The type of parallel imaging used (for example `"GRAPPA"`,'
                ' `"SENSE"`).\nCorresponds to DICOM Tag 0018, 9078 `Parallel'
                " Acquisition Technique`.\n"
            )
        ),
    ] = None
    ParallelReductionFactorInPlane: Annotated[
        Optional[float],
        Field(
            description=(
                "The parallel imaging (for instance, GRAPPA) factor.\nUse the"
                " denominator of the fraction of k-space encoded for each slice.\nFor"
                " example, 2 means half of k-space is encoded.\nCorresponds to DICOM"
                " Tag 0018, 9069 `Parallel Reduction Factor In-plane`.\n"
            )
        ),
    ] = None
    PartialFourier: Annotated[
        Optional[float],
        Field(
            description=(
                "The fraction of partial Fourier information collected.\nCorresponds to"
                " DICOM Tag 0018, 9081 `Partial Fourier`.\n"
            )
        ),
    ] = None
    PartialFourierDirection: Annotated[
        Optional[str],
        Field(
            description=(
                "The direction where only partial Fourier information was"
                " collected.\nCorresponds to DICOM Tag 0018, 9036 `Partial Fourier"
                " Direction`.\n"
            )
        ),
    ] = None
    PharmaceuticalDoseAmount: Annotated[
        Optional[Union[float, list[float]]],
        Field(
            description="Dose amount of pharmaceutical coadministered with tracer.\n"
        ),
    ] = None
    PharmaceuticalDoseRegimen: Annotated[
        Optional[str],
        Field(
            description=(
                "Details of the pharmaceutical dose regimen.\nEither adequate"
                " description or short-code relating to regimen documented\nelsewhere"
                ' (for example, `"single oral bolus"`).\n'
            )
        ),
    ] = None
    PharmaceuticalDoseTime: Annotated[
        Optional[Union[float, list[float]]],
        Field(
            description=(
                "Time of administration of pharmaceutical dose, relative to time"
                " zero.\nFor an infusion, this should be a vector with two elements"
                " specifying the\nstart and end of the infusion period. For more"
                " complex dose regimens,\nthe regimen description should be complete"
                " enough to enable unambiguous\ninterpretation of"
                ' `"PharmaceuticalDoseTime"`.\nUnit format of the specified'
                " pharmaceutical dose time MUST be seconds.\n"
            )
        ),
    ] = None
    PharmaceuticalDoseUnits: Annotated[
        Optional[str],
        Field(
            description=(
                'Unit format relating to pharmaceutical dose\n(for example, `"mg"` or'
                ' `"mg/kg"`).\n'
            )
        ),
    ] = None
    PharmaceuticalName: Annotated[
        Optional[str],
        Field(description="Name of pharmaceutical coadministered with tracer.\n"),
    ] = None
    PhaseEncodingDirection: Annotated[
        Optional[PhaseEncodingDirection],
        Field(
            description=(
                "The letters `i`, `j`, `k` correspond to the first, second and third"
                " axis of\nthe data in the NIFTI file.\nThe polarity of the phase"
                " encoding is assumed to go from zero index to\nmaximum index unless"
                " `-` sign is present\n(then the order is reversed - starting from the"
                " highest index instead of\nzero).\n`PhaseEncodingDirection` is defined"
                " as the direction along which phase is was\nmodulated which may result"
                " in visible distortions.\nNote that this is not the same as the DICOM"
                " term\n`InPlanePhaseEncodingDirection` which can have `ROW` or `COL`"
                " values.\n"
            )
        ),
    ] = None
    PhotoDescription: Annotated[
        Optional[str], Field(description="Description of the photo.\n")
    ] = None
    PixelSize: Annotated[
        Optional[list[PixelSizeItem]],
        Field(
            description=(
                "A 2- or 3-number array of the physical size of a pixel, either"
                " `[PixelSizeX, PixelSizeY]`\nor `[PixelSizeX, PixelSizeY,"
                " PixelSizeZ]`, where X is the width, Y the height and Z"
                " the\ndepth.\nIf the file format is OME-TIFF, these values need to be"
                " consistent with `PhysicalSizeX`,\n`PhysicalSizeY` and `PhysicalSizeZ`"
                " OME metadata fields, after converting in\n`PixelSizeUnits` according"
                " to `PhysicalSizeXunit`, `PhysicalSizeYunit` and\n`PhysicalSizeZunit`"
                " OME fields.\n"
            ),
            max_length=3,
            min_length=2,
        ),
    ] = None
    PixelSizeUnits: Annotated[
        Optional[PixelSizeUnits],
        Field(
            description=(
                'Unit format of the specified `"PixelSize"`. MUST be one of: `"mm"`'
                ' (millimeter), `"um"`\n(micrometer) or `"nm"` (nanometer).\n'
            )
        ),
    ] = None
    PlasmaAvail: Annotated[
        Optional[bool],
        Field(
            description="Boolean that specifies if plasma measurements are available.\n"
        ),
    ] = None
    PlasmaFreeFraction: Annotated[
        Optional[float],
        Field(
            description=(
                "Measured free fraction in plasma, meaning the concentration of free"
                " compound\nin plasma divided by total concentration of compound in"
                " plasma\n(Units: 0-100%).\n"
            ),
            ge=0.0,
            le=100.0,
        ),
    ] = None
    PlasmaFreeFractionMethod: Annotated[
        Optional[str], Field(description="Method used to estimate free fraction.\n")
    ] = None
    PostLabelingDelay: Annotated[
        Optional[Union[PostLabelingDelayItem, list[PostLabelingDelayItem]]],
        Field(
            description=(
                "This is the postlabeling delay (PLD) time, in seconds, after the end"
                ' of the\nlabeling (for `"CASL"` or `"PCASL"`) or middle of the'
                ' labeling pulse\n(for `"PASL"`) until the middle of the excitation'
                " pulse applied to the\nimaging slab (for 3D acquisition) or first"
                " slice (for 2D acquisition).\nCan be a number (for a single-PLD time"
                " series) or an array of numbers\n(for multi-PLD and Look-Locker).\nIn"
                " the latter case, the array of numbers contains the PLD of each"
                " volume,\nnamely each `control` and `label`, in the acquisition"
                " order.\nAny image within the time-series without a PLD, for example"
                " an `m0scan`,\nis indicated by a zero.\nBased on DICOM Tags 0018, 9079"
                " `Inversion Times` and 0018, 0082\n`InversionTime`.\n"
            )
        ),
    ] = None
    PowerLineFrequency: Annotated[
        Optional[Union[PowerLineFrequencyItem, PowerLineFrequencyEnum]],
        Field(
            description=(
                "Frequency (in Hz) of the power grid at the geographical location of"
                " the\ninstrument (for example, `50` or `60`).\n"
            )
        ),
    ] = None
    PromptRate: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "Prompt rate for each frame (same units as `Units`, for example,"
                ' `"Bq/mL"`).\n'
            )
        ),
    ] = None
    PulseSequenceDetails: Annotated[
        Optional[str],
        Field(
            description=(
                "Information beyond pulse sequence type that identifies the specific"
                ' pulse\nsequence used (for example,\n`"Standard Siemens Sequence'
                ' distributed with the VB17 software"`,\n`"Siemens WIP ### version'
                ' #.##,"` or\n`"Sequence written by X using a version compiled on'
                ' MM/DD/YYYY"`).\n'
            )
        ),
    ] = None
    PulseSequenceType: Annotated[
        Optional[str],
        Field(
            description=(
                "A general description of the pulse sequence used for the scan\n(for"
                ' example, `"MPRAGE"`, `"Gradient Echo EPI"`, `"Spin Echo'
                ' EPI"`,\n`"Multiband gradient echo EPI"`).\n'
            )
        ),
    ] = None
    Purity: Annotated[
        Optional[float],
        Field(
            description="Purity of the radiolabeled compound (between 0 and 100%).\n",
            ge=0.0,
            le=100.0,
        ),
    ] = None
    RandomRate: Annotated[
        Optional[list[float]],
        Field(
            description=(
                'Random rate for each frame (same units as `"Units"`, for example,'
                ' `"Bq/mL"`).\n'
            )
        ),
    ] = None
    RawSources: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "A list of paths relative to dataset root pointing to the BIDS-Raw"
                " file(s)\nthat were used in the creation of this derivative.\nThis"
                " field is DEPRECATED, and this metadata SHOULD be recorded in"
                " the\n`Sources` field using [BIDS"
                " URIs](SPEC_ROOT/02-common-principles.md#bids-uri)\nto distinguish"
                " sources from different datasets.\n"
            )
        ),
    ] = None
    ReceiveCoilActiveElements: Annotated[
        Optional[str],
        Field(
            description=(
                "Information describing the active/selected elements of the receiver"
                " coil.\nThis does not correspond to a tag in the DICOM ontology.\nThe"
                " vendor-defined terminology for active coil elements can go in this"
                " field.\n"
            )
        ),
    ] = None
    ReceiveCoilName: Annotated[
        Optional[str],
        Field(
            description=(
                "Information describing the receiver coil.\nCorresponds to DICOM Tag"
                " 0018, 1250 `Receive Coil Name`,\nalthough not all vendors populate"
                " that DICOM Tag,\nin which case this field can be derived from an"
                " appropriate\nprivate DICOM field.\n"
            )
        ),
    ] = None
    ReconFilterSize: Annotated[
        Optional[Union[float, list[float]]],
        Field(
            description=(
                'Kernel size of post-recon filter (FWHM) in default units `"mm"`.\n'
            )
        ),
    ] = None
    ReconFilterType: Annotated[
        Optional[Union[str, list[str]]],
        Field(description='Type of post-recon smoothing (for example, `["Shepp"]`).\n'),
    ] = None
    ReconMethodImplementationVersion: Annotated[
        Optional[str],
        Field(
            description=(
                "Identification for the software used, such as name and version.\n"
            )
        ),
    ] = None
    ReconMethodName: Annotated[
        Optional[str],
        Field(
            description=(
                'Reconstruction method or algorithm (for example, `"3d-op-osem"`).\n'
            )
        ),
    ] = None
    ReconMethodParameterLabels: Annotated[
        Optional[list[str]],
        Field(
            description=(
                'Names of reconstruction parameters (for example, `["subsets",'
                ' "iterations"]`).\n'
            )
        ),
    ] = None
    ReconMethodParameterUnits: Annotated[
        Optional[list[str]],
        Field(
            description=(
                'Unit of reconstruction parameters (for example, `["none", "none"]`).\n'
            )
        ),
    ] = None
    ReconMethodParameterValues: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "Values of reconstruction parameters (for example, `[21, 3]`).\n"
            )
        ),
    ] = None
    RecordingDuration: Annotated[
        Optional[float],
        Field(
            description="Length of the recording in seconds (for example, `3600`).\n"
        ),
    ] = None
    RecordingType: Annotated[
        Optional[RecordingType],
        Field(
            description=(
                'Defines whether the recording is `"continuous"`, `"discontinuous"`,'
                ' or\n`"epoched"`, where `"epoched"` is limited to time windows about'
                " events of\ninterest (for example, stimulus presentations or subject"
                " responses).\n"
            )
        ),
    ] = None
    ReferencesAndLinks: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "List of references to publications that contain information on the"
                " dataset.\nA reference may be textual or"
                " a\n[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator).\n"
            )
        ),
    ] = None
    RepetitionTime: Annotated[
        Optional[float],
        Field(
            description=(
                "The time in seconds between the beginning of an acquisition of one"
                " volume\nand the beginning of acquisition of the volume following it"
                " (TR).\nWhen used in the context of functional acquisitions this"
                " parameter best\ncorresponds to\n[DICOM Tag 0020,"
                " 0110](http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0020,0110)):\nthe"
                ' "time delta between images in a\ndynamic of functional set of'
                ' images" but may also be found in\n[DICOM Tag 0018,'
                ' 0080](http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0018,0080)):\n"the'
                " period of time in msec between the beginning\nof a pulse sequence and"
                " the beginning of the succeeding\n(essentially identical) pulse"
                ' sequence".\nThis definition includes time between scans (when no'
                " data has been acquired)\nin case of sparse acquisition schemes.\nThis"
                " value MUST be consistent with the 'pixdim[4]' field (after"
                " accounting\nfor units stored in 'xyzt_units' field) in the NIfTI"
                " header.\nThis field is mutually exclusive with VolumeTiming.\n"
            ),
            gt=0.0,
        ),
    ] = None
    RepetitionTimeExcitation: Annotated[
        Optional[float],
        Field(
            description=(
                "The interval, in seconds, between two successive excitations.\n[DICOM"
                " Tag 0018,"
                " 0080](http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0018,0080))\nbest"
                " refers to this parameter.\nThis field may be used together with the"
                ' `"RepetitionTimePreparation"` for\ncertain use cases, such'
                " as\n[MP2RAGE](https://doi.org/10.1016/j.neuroimage.2009.10.002).\nUse"
                " `RepetitionTimeExcitation` (in combination"
                ' with\n`"RepetitionTimePreparation"` if needed) for anatomy imaging'
                ' data rather than\n`"RepetitionTime"` as it is already defined as the'
                " amount of time that it takes\nto acquire a single volume in"
                " the\n[task imaging"
                " data](SPEC_ROOT/04-modality-specific-files/01-magnetic-resonance-\\\nimaging-data.md#task-including-resting-state-imaging-data)\nsection.\n"
            ),
            ge=0.0,
        ),
    ] = None
    RepetitionTimePreparation: Annotated[
        Optional[
            Union[RepetitionTimePreparationItem, list[RepetitionTimePreparationItem]]
        ],
        Field(
            description=(
                "The interval, in seconds, that it takes a preparation pulse block"
                " to\nre-appear at the beginning of the succeeding (essentially"
                " identical) pulse\nsequence block.\nThe data type number may apply to"
                " files from any MRI modality concerned with\na single value for this"
                " field.\nThe data type array provides a value for each volume in a 4D"
                " dataset and\nshould only be used when the volume timing is critical"
                " for interpretation of\nthe data, such as"
                " in\n[ASL](SPEC_ROOT/04-modality-specific-files/01-magnetic-resonance-imaging-data.md\\\n#arterial-spin-labeling-perfusion-data).\n"
            )
        ),
    ] = None
    Resolution: Annotated[
        Optional[Union[str, dict[str, str]]],
        Field(
            description=(
                "Specifies the interpretation of the resolution keyword.\nIf an object"
                " is used, then the keys should be values for the `res` entity\nand"
                " values should be descriptions of those `res` values.\n"
            )
        ),
    ] = None
    SEEGChannelCount: Annotated[
        Optional[int], Field(description="Number of SEEG channels.\n", ge=0)
    ] = None
    SampleEmbedding: Annotated[
        Optional[str],
        Field(
            description=(
                'Description of the tissue sample embedding (for example: `"Epoxy'
                ' resin"`).\n'
            )
        ),
    ] = None
    SampleEnvironment: Annotated[
        Optional[SampleEnvironment],
        Field(
            description=(
                'Environment in which the sample was imaged. MUST be one of: `"in'
                ' vivo"`, `"ex vivo"`\nor `"in vitro"`.\n'
            )
        ),
    ] = None
    SampleExtractionInstitution: Annotated[
        Optional[str],
        Field(
            description=(
                "The name of the institution in charge of the extraction of the"
                " sample,\nif different from the institution in charge of the equipment"
                " that produced the image.\n"
            )
        ),
    ] = None
    SampleExtractionProtocol: Annotated[
        Optional[str],
        Field(
            description=(
                "Description of the sample extraction protocol"
                " or\n[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)\n(for"
                " example from [protocols.io](https://www.protocols.io/)).\n"
            )
        ),
    ] = None
    SampleFixation: Annotated[
        Optional[str],
        Field(
            description=(
                'Description of the tissue sample fixation\n(for example: `"4%'
                ' paraformaldehyde, 2% glutaraldehyde"`).\n'
            )
        ),
    ] = None
    SampleOrigin: Annotated[
        Optional[SampleOrigin],
        Field(
            description=(
                "Describes from which tissue the genetic information was extracted.\n"
            )
        ),
    ] = None
    SamplePrimaryAntibody: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Description(s) of the primary antibody used for"
                " immunostaining.\nEither an [RRID](https://scicrunch.org/resources) or"
                " the name, supplier and catalogue\nnumber of a commercial"
                " antibody.\nFor non-commercial antibodies either an"
                " [RRID](https://scicrunch.org/resources) or the\nhost-animal and"
                ' immunogen used (for examples: `"RRID:AB_2122563"` or\n`"Rabbit'
                " anti-Human HTR5A Polyclonal Antibody, Invitrogen, Catalog #"
                ' PA1-2453"`).\nMAY be an array of strings if different antibodies are'
                " used in each channel of the file.\n"
            )
        ),
    ] = None
    SampleSecondaryAntibody: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Description(s) of the secondary antibody used for"
                " immunostaining.\nEither an [RRID](https://scicrunch.org/resources) or"
                " the name, supplier and catalogue\nnumber of a commercial"
                " antibody.\nFor non-commercial antibodies either an"
                " [RRID](https://scicrunch.org/resources) or the\nhost-animal and"
                ' immunogen used (for examples: `"RRID:AB_228322"` or\n`"Goat'
                " anti-Mouse IgM Secondary Antibody, Invitrogen, Catalog #"
                ' 31172"`).\nMAY be an array of strings if different antibodies are'
                " used in each channel of the file.\n"
            )
        ),
    ] = None
    SampleStaining: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Description(s) of the tissue sample staining (for example:"
                ' `"Osmium"`).\nMAY be an array of strings if different stains are used'
                ' in each channel of the file\n(for example: `["LFB", "PLP"]`).\n'
            )
        ),
    ] = None
    SamplingFrequency: Annotated[
        Optional[float],
        Field(
            description=(
                "Sampling frequency (in Hz) of all the data in the"
                " recording,\nregardless of their type (for example, `2400`).\n"
            )
        ),
    ] = None
    SamplingFrequency__nirs: Annotated[
        Optional[Union[float, SamplingFrequencyNir]],
        Field(
            description=(
                "Sampling frequency (in Hz) of all the data in the"
                " recording,\nregardless of their type (for example, `2400`).\n"
            )
        ),
    ] = None
    ScaleFactor: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "Scale factor for each frame. This field MUST be defined if the imaging"
                " data (`.nii[.gz]`) are scaled.\nIf this field is not defined, then it"
                " is assumed that the scaling factor is 1. Defining this field\nwhen"
                " the scaling factor is 1 is RECOMMENDED, for the sake of clarity.\n"
            )
        ),
    ] = None
    ScanDate: Annotated[
        Optional[date],
        Field(
            description=(
                'Date of scan in the format `"YYYY-MM-DD[Z]"`.\nThis field is'
                " DEPRECATED, and this metadata SHOULD be recorded in the `acq_time`"
                " column of the\ncorresponding [Scans"
                " file](SPEC_ROOT/03-modality-agnostic-files.md#scans-file).\n"
            )
        ),
    ] = None
    ScanOptions: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Parameters of ScanningSequence.\nCorresponds to DICOM Tag 0018, 0022"
                " `Scan Options`.\n"
            )
        ),
    ] = None
    ScanStart: Annotated[
        Optional[float],
        Field(
            description=(
                "Time of start of scan with respect to `TimeZero` in the default unit"
                " seconds.\n"
            )
        ),
    ] = None
    ScanningSequence: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Description of the type of data acquired.\nCorresponds to DICOM Tag"
                " 0018, 0020 `Scanning Sequence`.\n"
            )
        ),
    ] = None
    ScatterFraction: Annotated[
        Optional[list[ScatterFractionItem]],
        Field(description="Scatter fraction for each frame (Units: 0-100%).\n"),
    ] = None
    SequenceName: Annotated[
        Optional[str],
        Field(
            description=(
                "Manufacturer's designation of the sequence name.\nCorresponds to DICOM"
                " Tag 0018, 0024 `Sequence Name`.\n"
            )
        ),
    ] = None
    SequenceVariant: Annotated[
        Optional[Union[str, list[str]]],
        Field(
            description=(
                "Variant of the ScanningSequence.\nCorresponds to DICOM Tag 0018, 0021"
                " `Sequence Variant`.\n"
            )
        ),
    ] = None
    ShortChannelCount: Annotated[
        Optional[int],
        Field(
            description=(
                "The number of short channels. 0 indicates no short channels.\n"
            ),
            ge=0,
        ),
    ] = None
    SinglesRate: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "Singles rate for each frame (same units as `Units`, for example,"
                ' `"Bq/mL"`).\n'
            )
        ),
    ] = None
    SkullStripped: Annotated[
        Optional[bool],
        Field(
            description=(
                "Whether the volume was skull stripped (non-brain voxels set to zero)"
                " or not.\n"
            )
        ),
    ] = None
    SliceEncodingDirection: Annotated[
        Optional[SliceEncodingDirection],
        Field(
            description=(
                "The axis of the NIfTI data along which slices were acquired,\nand the"
                ' direction in which `"SliceTiming"` is defined with respect to.\n`i`,'
                " `j`, `k` identifiers correspond to the first, second and third axis"
                " of\nthe data in the NIfTI file.\nA `-` sign indicates that the"
                ' contents of `"SliceTiming"` are defined in\nreverse order - that is,'
                " the first entry corresponds to the slice with the\nlargest index, and"
                " the final entry corresponds to slice index zero.\nWhen present, the"
                ' axis defined by `"SliceEncodingDirection"` needs to be\nconsistent'
                " with the `slice_dim` field in the NIfTI header.\nWhen absent, the"
                ' entries in `"SliceTiming"` must be in the order of increasing\nslice'
                " index as defined by the NIfTI header.\n"
            )
        ),
    ] = None
    SliceThickness: Annotated[
        Optional[float],
        Field(
            description=(
                'Slice thickness of the tissue sample in the unit micrometers (`"um"`)'
                " (for example: `5`).\n"
            ),
            gt=0.0,
        ),
    ] = None
    SliceTiming: Annotated[
        Optional[list[SliceTimingItem]],
        Field(
            description=(
                "The time at which each slice was acquired within each volume (frame)"
                " of the\nacquisition.\nSlice timing is not slice order -- rather, it"
                " is a list of times containing\nthe time (in seconds) of each slice"
                " acquisition in relation to the beginning\nof volume acquisition.\nThe"
                " list goes through the slices along the slice axis in the slice"
                " encoding\ndimension (see below).\nNote that to ensure the proper"
                ' interpretation of the `"SliceTiming"` field,\nit is important to'
                " check if the OPTIONAL `SliceEncodingDirection` exists.\nIn"
                ' particular, if `"SliceEncodingDirection"` is negative,\nthe entries'
                ' in `"SliceTiming"` are defined in reverse order with respect to'
                ' the\nslice axis, such that the final entry in the `"SliceTiming"`'
                " list is the time\nof acquisition of slice 0. Without this parameter"
                " slice time correction will\nnot be possible.\n"
            )
        ),
    ] = None
    SoftwareFilters: Annotated[
        Optional[Union[dict[str, dict[str, Any]], SoftwareFilter]],
        Field(
            description=(
                "[Object](https://www.json.org/json-en.html)\nof temporal software"
                ' filters applied, or `"n/a"` if the data is\nnot available.\nEach'
                " key-value pair in the JSON object is a name of the filter and an"
                " object\nin which its parameters are defined as key-value pairs\n(for"
                ' example, `{"Anti-aliasing filter":\n{"half-amplitude cutoff (Hz)":'
                ' 500, "Roll-off": "6dB/Octave"}}`).\n'
            )
        ),
    ] = None
    SoftwareName: Annotated[
        Optional[str],
        Field(
            description="Name of the software that was used to present the stimuli.\n"
        ),
    ] = None
    SoftwareRRID: Annotated[
        Optional[str],
        Field(
            description=(
                "[Research Resource Identifier](https://scicrunch.org/resources) of"
                " the\nsoftware that was used to present the stimuli.\nExamples: The"
                " RRID for Psychtoolbox is 'SCR_002881',\nand that of PsychoPy is"
                " 'SCR_006571'.\n"
            )
        ),
    ] = None
    SoftwareVersion: Annotated[
        Optional[str],
        Field(
            description=(
                "Version of the software that was used to present the stimuli.\n"
            )
        ),
    ] = None
    SoftwareVersions: Annotated[
        Optional[str],
        Field(
            description=(
                "Manufacturer's designation of software version of the equipment that"
                " produced\nthe measurements.\n"
            )
        ),
    ] = None
    SourceDatasets: Annotated[
        Optional[list[SourceDataset]],
        Field(
            description=(
                "Used to specify the locations and relevant attributes of all source"
                ' datasets.\nValid keys in each object include `"URL"`, `"DOI"`'
                " (see\n[URI](SPEC_ROOT/02-common-principles.md#uniform-resource-indicator)),"
                ' and\n`"Version"`'
                " with\n[string](https://www.w3schools.com/js/js_json_datatypes.asp)\nvalues.\n"
            )
        ),
    ] = None
    Sources: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "A list of files with the paths specified using\n[BIDS"
                " URIs](SPEC_ROOT/02-common-principles.md#bids-uri);\nthese files were"
                " directly used in the creation of this derivative data file.\nFor"
                " example, if a derivative A is used in the creation of"
                " another\nderivative B, which is in turn used to generate C in a chain"
                ' of A->B->C,\nC should only list B in `"Sources"`, and B should only'
                ' list A in `"Sources"`.\nHowever, in case both X and Y are directly'
                " used in the creation of Z,\nthen Z should list X and Y in"
                ' `"Sources"`,\nregardless of whether X was used to generate Y.\nUsing'
                " paths specified relative to the dataset root"
                " is\n[DEPRECATED](SPEC_ROOT/02-common-principles.md#definitions).\n"
            )
        ),
    ] = None
    SourceType: Annotated[
        Optional[str],
        Field(
            description=(
                "Type of source. Preferably a specific model/part number is"
                " supplied.\nThis is a freeform description, but the following keywords"
                ' are suggested:\n`"LED"`, `"LASER"`, `"VCSEL"`. If individual channels'
                " have different SourceType,\nthen the field here should be specified"
                ' as "mixed"\nand this column should be included in optodes.tsv.\n'
            )
        ),
    ] = None
    SpatialReference: Annotated[
        Optional[
            Union[
                SpatialReferenceEnum,
                AnyUrl,
                str,
                dict[str, Union[SpatialReferenceEnum, AnyUrl, str]],
            ]
        ],
        Field(
            description=(
                "For images with a single reference, the value MUST be a single"
                " string.\nFor images with multiple references, such as surface and"
                " volume references,\na JSON object MUST be used.\n"
            )
        ),
    ] = None
    SpecificRadioactivity: Annotated[
        Optional[Union[float, SpecificRadioactivityEnum]],
        Field(
            description=(
                "Specific activity of compound injected.\n**Note this is not required"
                " for an FDG acquisition, since it is not available,\nand SHOULD be set"
                ' to `"n/a"`**.\n'
            )
        ),
    ] = None
    SpecificRadioactivityMeasTime: Annotated[
        Optional[time],
        Field(
            description=(
                "Time to which specific radioactivity measurement above applies in the"
                ' default\nunit `"hh:mm:ss"`.\n'
            )
        ),
    ] = None
    SpecificRadioactivityUnits: Annotated[
        Optional[Union[str, SpecificRadioactivityUnit]],
        Field(
            description=(
                "Unit format of specified specific radioactivity (for example,"
                ' `"Bq/g"`).\n**Note this is not required for an FDG acquisition, since'
                ' it is not available,\nand SHOULD be set to `"n/a"`**.\n'
            )
        ),
    ] = None
    SpoilingGradientDuration: Annotated[
        Optional[float],
        Field(
            description=(
                "The duration of the spoiler gradient lobe in seconds.\nThe duration of"
                " a trapezoidal lobe is defined as the summation of ramp-up\nand"
                " plateau times.\n"
            )
        ),
    ] = None
    SpoilingGradientMoment: Annotated[
        Optional[float],
        Field(
            description=(
                "Zeroth moment of the spoiler gradient lobe in\nmillitesla times second"
                " per meter (mT.s/m).\n"
            )
        ),
    ] = None
    SpoilingRFPhaseIncrement: Annotated[
        Optional[float],
        Field(
            description=(
                "The amount of incrementation described in degrees,\nwhich is applied"
                " to the phase of the excitation pulse at each TR period for\nachieving"
                " RF spoiling.\n"
            )
        ),
    ] = None
    SpoilingState: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean stating whether the pulse sequence uses any type of"
                " spoiling\nstrategy to suppress residual transverse magnetization.\n"
            )
        ),
    ] = None
    SpoilingType: Annotated[
        Optional[SpoilingType],
        Field(
            description=(
                "Specifies which spoiling method(s) are used by a spoiled sequence.\n"
            )
        ),
    ] = None
    StartTime: Annotated[
        Optional[float],
        Field(
            description=(
                "Start time in seconds in relation to the start of acquisition of the"
                " first\ndata sample in the corresponding neural dataset (negative"
                " values are allowed).\n"
            )
        ),
    ] = None
    StationName: Annotated[
        Optional[str],
        Field(
            description=(
                "Institution defined name of the machine that produced the"
                " measurements.\n"
            )
        ),
    ] = None
    StimulusPresentation: Annotated[
        Optional[StimulusPresentation],
        Field(
            description=(
                "Object containing key-value pairs related to the software used to"
                " present\nthe stimuli during the experiment,"
                ' specifically:\n`"OperatingSystem"`, `"SoftwareName"`,'
                ' `"SoftwareRRID"`, `"SoftwareVersion"` and\n`"Code"`.\nSee table below'
                " for more information.\n"
            )
        ),
    ] = None
    SubjectArtefactDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Freeform description of the observed subject artefact and its possible"
                ' cause\n(for example, `"Vagus Nerve Stimulator"`, `"non-removable'
                ' implant"`).\nIf this field is set to `"n/a"`, it will be interpreted'
                " as absence of major\nsource of artifacts except cardiac and blinks.\n"
            )
        ),
    ] = None
    TaskDescription: Annotated[
        Optional[str], Field(description="Longer description of the task.\n")
    ] = None
    TaskName: Annotated[
        Optional[str],
        Field(
            description=(
                "Name of the task.\nNo two tasks should have the same name.\nThe task"
                ' label included in the file name is derived from this `"TaskName"`'
                " field\nby removing all non-alphanumeric characters (that is, all"
                ' except those matching `[0-9a-zA-Z]`).\nFor example `"TaskName"`'
                ' `"faces n-back"` will correspond to task label\n`facesnback`.\n'
            )
        ),
    ] = None
    TermURL: Annotated[
        Optional[str],
        Field(
            description=(
                "URL pointing to a formal definition of this type of data in an"
                " ontology\navailable on the web.\n"
            )
        ),
    ] = None
    TimeZero: Annotated[
        Optional[time],
        Field(
            description=(
                "Time zero to which all scan and/or blood measurements have been"
                ' adjusted to,\nin the unit "hh:mm:ss".\nThis should be equal to'
                ' `"InjectionStart"` or `"ScanStart"`.\n'
            )
        ),
    ] = None
    TissueDeformationScaling: Annotated[
        Optional[float],
        Field(
            description=(
                "Estimated deformation of the tissue, given as a percentage of the"
                " original\ntissue size (for examples: for a shrinkage of 3%, the value"
                " is `97`;\nand for an expansion of 100%, the value is `200`).\n"
            ),
            gt=0.0,
        ),
    ] = None
    TissueOrigin: Annotated[
        Optional[TissueOrigin],
        Field(
            description=(
                'Describes the type of tissue analyzed for `"SampleOrigin"` `brain`.\n'
            )
        ),
    ] = None
    TotalAcquiredPairs: Annotated[
        Optional[float],
        Field(
            description=(
                "The total number of acquired `control`-`label` pairs.\nA single pair"
                " consists of a single `control` and a single `label` image.\n"
            ),
            gt=0.0,
        ),
    ] = None
    TotalReadoutTime: Annotated[
        Optional[float],
        Field(
            description=(
                'This is actually the "effective" total readout time,\ndefined as the'
                " readout duration, specified in seconds,\nthat would have generated"
                " data with the given level of distortion.\nIt is NOT the actual,"
                ' physical duration of the readout train.\nIf `"EffectiveEchoSpacing"`'
                " has been properly computed,\nit is just `EffectiveEchoSpacing *"
                " (ReconMatrixPE - 1)`.\n"
            )
        ),
    ] = None
    TracerMolecularWeight: Annotated[
        Optional[float],
        Field(description="Accurate molecular weight of the tracer used.\n"),
    ] = None
    TracerMolecularWeightUnits: Annotated[
        Optional[str],
        Field(
            description=(
                'Unit of the molecular weights measurement (for example, `"g/mol"`).\n'
            )
        ),
    ] = None
    TracerName: Annotated[
        Optional[str],
        Field(
            description='Name of the tracer compound used (for example, `"CIMBI-36"`)\n'
        ),
    ] = None
    TracerRadLex: Annotated[
        Optional[str],
        Field(description="ID of the tracer compound from the RadLex Ontology.\n"),
    ] = None
    TracerRadionuclide: Annotated[
        Optional[str],
        Field(description='Radioisotope labelling tracer (for example, `"C11"`).\n'),
    ] = None
    TracerSNOMED: Annotated[
        Optional[str],
        Field(
            description=(
                "ID of the tracer compound from the SNOMED Ontology\n(subclass of"
                " Radioactive isotope).\n"
            )
        ),
    ] = None
    TriggerChannelCount: Annotated[
        Optional[int],
        Field(
            description="Number of channels for digital (TTL bit level) triggers.\n",
            ge=0,
        ),
    ] = None
    TubingLength: Annotated[
        Optional[float],
        Field(
            description=(
                "The length of the blood tubing, from the subject to the detector in"
                " meters.\n"
            )
        ),
    ] = None
    TubingType: Annotated[
        Optional[str],
        Field(
            description=(
                "Description of the type of tubing used, ideally including the material"
                " and\n(internal) diameter.\n"
            )
        ),
    ] = None
    Type: Annotated[
        Optional[Type],
        Field(
            description=(
                'Short identifier of the mask.\nThe value `"Brain"` refers to a brain'
                ' mask.\nThe value `"Lesion"` refers to a lesion mask.\nThe value'
                ' `"Face"` refers to a face mask.\nThe value `"ROI"` refers to a region'
                " of interest mask.\n"
            )
        ),
    ] = None
    Units: Annotated[
        Optional[str],
        Field(
            description=(
                "Measurement units for the associated file.\nSI units in CMIXF"
                " formatting are RECOMMENDED\n(see"
                " [Units](SPEC_ROOT/02-common-principles.md#units)).\n"
            )
        ),
    ] = None
    VascularCrushing: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean indicating if Vascular Crushing is used.\nCorresponds to DICOM"
                " Tag 0018, 9259 `ASL Crusher Flag`.\n"
            )
        ),
    ] = None
    VascularCrushingVENC: Annotated[
        Optional[Union[float, list[float]]],
        Field(
            description=(
                "The crusher gradient strength, in centimeters per second.\nSpecify"
                " either one number for the total time-series, or provide an array"
                " of\nnumbers, for example when using QUASAR, using the value zero to"
                " identify\nvolumes for which `VascularCrushing` was turned"
                " off.\nCorresponds to DICOM Tag 0018, 925A `ASL Crusher Flow Limit`.\n"
            )
        ),
    ] = None
    VolumeTiming: Annotated[
        Optional[list[float]],
        Field(
            description=(
                "The time at which each volume was acquired during the acquisition.\nIt"
                " is described using a list of times referring to the onset of each"
                " volume\nin the BOLD series.\nThe list must have the same length as"
                " the BOLD series,\nand the values must be non-negative and"
                " monotonically increasing.\nThis field is mutually exclusive with"
                ' `"RepetitionTime"` and `"DelayTime"`.\nIf defined, this requires'
                ' acquisition time (TA) be defined via either\n`"SliceTiming"` or'
                ' `"AcquisitionDuration"` be defined.\n'
            ),
            min_length=1,
        ),
    ] = None
    WholeBloodAvail: Annotated[
        Optional[bool],
        Field(
            description=(
                "Boolean that specifies if whole blood measurements are available.\nIf"
                " `true`, the `whole_blood_radioactivity` column MUST be present in"
                " the\ncorresponding `*_blood.tsv` file.\n"
            )
        ),
    ] = None
    WithdrawalRate: Annotated[
        Optional[float],
        Field(
            description=(
                "The rate at which the blood was withdrawn from the subject.\nThe unit"
                ' of the specified withdrawal rate should be in `"mL/s"`.\n'
            )
        ),
    ] = None
    field_CoordUnits: Annotated[
        Optional[FieldCoordUnits], Field(alias="_CoordUnits")
    ] = None
    field_EEGCoordSys: Annotated[
        Optional[FieldEEGCoordSys], Field(alias="_EEGCoordSys")
    ] = None
    field_GeneticLevelEnum: Annotated[
        Optional[FieldGeneticLevelEnum], Field(alias="_GeneticLevelEnum")
    ] = None
    field_LandmarkCoordinates: Annotated[
        Optional[dict[str, list[float]]], Field(alias="_LandmarkCoordinates")
    ] = None
    field_MEGCoordSys: Annotated[
        Optional[FieldMEGCoordSys], Field(alias="_MEGCoordSys")
    ] = None
    field_StandardTemplateCoordSys: Annotated[
        Optional[FieldStandardTemplateCoordSys],
        Field(alias="_StandardTemplateCoordSys"),
    ] = None
    field_StandardTemplateDeprecatedCoordSys: Annotated[
        Optional[FieldStandardTemplateDeprecatedCoordSys],
        Field(alias="_StandardTemplateDeprecatedCoordSys"),
    ] = None
    field_iEEGCoordSys: Annotated[
        Optional[FieldIEEGCoordSys], Field(alias="_iEEGCoordSys")
    ] = None
    iEEGCoordinateProcessingDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Has any post-processing (such as projection) been done on the"
                ' electrode\npositions (for example, `"surface_projection"`,'
                ' `"none"`).\n'
            )
        ),
    ] = None
    iEEGCoordinateProcessingReference: Annotated[
        Optional[str],
        Field(
            description=(
                "A reference to a paper that defines in more detail the method used"
                " to\nlocalize the electrodes and to post-process the electrode"
                " positions.\n"
            )
        ),
    ] = None
    iEEGCoordinateSystem: Annotated[
        Optional[Any],
        Field(
            description=(
                "Defines the coordinate system for the iEEG sensors.\nSee"
                " the\n[Coordinate Systems"
                " Appendix](SPEC_ROOT/appendices/coordinate-systems.md)\nfor a list of"
                ' restricted keywords for coordinate systems.\nIf `"Other"`, provide'
                " definition of the coordinate system"
                " in\n`iEEGCoordinateSystemDescription`.\nIf positions correspond to"
                " pixel indices in a 2D image\n(of either a volume-rendering,"
                " surface-rendering, operative photo, or\noperative drawing), this MUST"
                ' be `"Pixels"`.\nFor more information, see the section on\n[2D'
                " coordinate"
                " systems](SPEC_ROOT/04-modality-specific-files/04-intracranial\\\n-electroencephalography.md#allowed-2d-coordinate-systems).\n"
            )
        ),
    ] = None
    iEEGCoordinateSystemDescription: Annotated[
        Optional[str],
        Field(
            description=(
                "Free-form text description of the coordinate system.\nMay also include"
                " a link to a documentation page or paper describing the\nsystem in"
                " greater detail.\n"
            )
        ),
    ] = None
    iEEGCoordinateUnits: Annotated[
        Optional[IEEGCoordinateUnits],
        Field(
            description=(
                'Units of the `*_electrodes.tsv`.\nMUST be `"pixels"` if'
                " `iEEGCoordinateSystem` is `Pixels`.\n"
            )
        ),
    ] = None
    iEEGElectrodeGroups: Annotated[
        Optional[str],
        Field(
            description=(
                "Field to describe the way electrodes are grouped into strips, grids or"
                ' depth\nprobes.\nFor example, `"grid1: 10x8 grid on left temporal'
                ' pole, strip2: 1x8 electrode\nstrip on xxx"`.\n'
            )
        ),
    ] = None
    iEEGGround: Annotated[
        Optional[str],
        Field(
            description=(
                'Description of the location of the ground electrode\n(`"placed on'
                ' right mastoid (M2)"`).\n'
            )
        ),
    ] = None
    iEEGPlacementScheme: Annotated[
        Optional[str],
        Field(
            description=(
                "Freeform description of the placement of the iEEG"
                ' electrodes.\nLeft/right/bilateral/depth/surface\n(for example, `"left'
                ' frontal grid and bilateral hippocampal depth"` or\n`"surface strip'
                ' and STN depth"` or\n`"clinical indication bitemporal, bilateral'
                ' temporal strips and left grid"`).\n'
            )
        ),
    ] = None
    iEEGReference: Annotated[
        Optional[str],
        Field(
            description=(
                "General description of the reference scheme used and (when applicable)"
                " of\nlocation of the reference electrode in the raw recordings\n(for"
                ' example, `"left mastoid"`, `"bipolar"`,\n`"T01"` for electrode with'
                ' name T01,\n`"intracranial electrode on top of a grid, not included'
                ' with data"`,\n`"upside down electrode"`).\nIf different channels have'
                " a different reference,\nthis field should have a general description"
                " and the channel specific\nreference should be defined in the"
                " `channels.tsv` file.\n"
            )
        ),
    ] = None
