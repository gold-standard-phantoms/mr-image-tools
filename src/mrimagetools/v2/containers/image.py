""" Classes for image encapsulation
Used to create a standard interface for ND images which can
be instantiated with either NIFTI files or using numpy arrays """

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np
import numpy.typing as npt
from dicom_core.nifti.dcm2niix import Dcm2NiixConfig
from dicom_core.nifti.load import (
    DicomToNiftiConfig,
    SeriesInstanceUID,
    SeriesSelection,
    SingleSeriesData,
    dicom_to_nifti,
)

from mrimagetools.v2.containers.image_metadata import ImageMetadata

UNITS_UNKNOWN = "unknown"
UNITS_METERS = "meter"
UNITS_MILLIMETERS = "mm"
UNITS_MICRONS = "micron"
UNITS_SECONDS = "sec"
UNITS_MILLISECONDS = "msec"
UNITS_MICROSECONDS = "usec"

SPATIAL_DOMAIN = "SPATIAL_DOMAIN"
INVERSE_DOMAIN = "INVERSE_DOMAIN"

REAL_IMAGE_TYPE = "REAL_IMAGE_TYPE"
IMAGINARY_IMAGE_TYPE = "IMAGINARY_IMAGE_TYPE"
MAGNITUDE_IMAGE_TYPE = "MAGNITUDE_IMAGE_TYPE"
PHASE_IMAGE_TYPE = "PHASE_IMAGE_TYPE"
COMPLEX_IMAGE_TYPE = "COMPLEX_IMAGE_TYPE"


VoxelSizeType = Union[list[float], tuple[float, ...], npt.NDArray[np.floating]]


class BaseImageContainer(ABC):
    """
    An abstract (interface) for an ND image. Defines
    a set of accessors
    :param data_domain: defines the data domain as SPATIAL_DOMAIN or
    INVERSE_DOMAIN (default is SPATIAL_DOMAIN)
    :param image_type: the image type. Must be one of:
    - REAL_IMAGE_TYPE
    - IMAGINARY_IMAGE_TYPE
    - MAGNITUDE_IMAGE_TYPE
    - PHASE_IMAGE_TYPE
    - COMPLEX_IMAGE_TYPE
    if this is not specified, it will be set to REAL_IMAGE_TYPE for scalar
    image dtypes and COMPLEX_IMAGE_TYPE for complex dtypes.
    If the space and time units are not set, they will be initialised to `mm` and
    `seconds` respectively.
    :param metadata: a metadata dictionary which is associated with the image data.
    This might contain, for example, timing parameters associated with the image
    acquisition.
    """

    def __init__(
        self,
        data_domain: str = SPATIAL_DOMAIN,
        image_type: Optional[str] = None,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ):
        if data_domain not in [SPATIAL_DOMAIN, INVERSE_DOMAIN]:
            raise ValueError(
                "data_domain is not of of SPATIAL_DOMAIN or INVERSE_DOMAIN"
            )
        self.data_domain = data_domain

        # Set the default image type based on the image dtype
        if image_type is None:
            if self.image.dtype in [np.complex64, np.complex128]:
                image_type = COMPLEX_IMAGE_TYPE
            else:
                image_type = REAL_IMAGE_TYPE

        # Check the image_type is in the valid options
        if image_type not in [
            IMAGINARY_IMAGE_TYPE,
            REAL_IMAGE_TYPE,
            PHASE_IMAGE_TYPE,
            COMPLEX_IMAGE_TYPE,
            MAGNITUDE_IMAGE_TYPE,
        ]:
            raise ValueError(f"{self} has bad value for image_type ({image_type})")

        # Check the image type matches the image dtype
        # complex dtypes must match COMPLEX_IMAGE_TYPE
        # non-complex dtype must not match COMPLEX_IMAGE_TYPE
        if (
            image_type is COMPLEX_IMAGE_TYPE
            and self.image.dtype not in [np.complex64, np.complex128]
            or image_type is not COMPLEX_IMAGE_TYPE
            and self.image.dtype in [np.complex64, np.complex128]
        ):
            raise ValueError(
                f"{self} created with image type of {image_type} "
                f"and image dtype is {self.image.dtype}"
            )

        self.image_type = image_type

        if metadata is None:
            metadata = ImageMetadata()

        if not isinstance(metadata, ImageMetadata):
            raise TypeError(f"metadata should be a ImageMetadata class, not {metadata}")
        self.metadata: ImageMetadata = metadata

        # Check that the space and time units are set. If not, set the defaults
        if self.space_units == UNITS_UNKNOWN:
            self.space_units = UNITS_MILLIMETERS
        if self.time_units == UNITS_UNKNOWN:
            self.time_units = UNITS_SECONDS

        # Check we aren't passed unexpected parameters
        if len(kwargs) != 0:
            raise TypeError(
                f"BaseImageContainer received unexpected arguments {kwargs}"
            )

    def clone(self) -> BaseImageContainer:
        """Makes a deep copy of all member variables in a new ImageContainer"""
        return deepcopy(self)

    @abstractmethod
    def as_numpy(self) -> NumpyImageContainer:
        """Return the image container as a NumpyImageContainer. If the container
        is already a NumpyImageContainer, return self"""

    @abstractmethod
    def as_nifti(self) -> NiftiImageContainer:
        """Return the image container as a NiftiImageContainer. If the container
        is already a NiftiImageContainer, return self"""

    @property
    @abstractmethod
    def has_nifti(self) -> bool:
        """Returns True if the image has an associated nifti container"""

    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        """Return the image data as a numpy array"""

    @image.setter
    def image(self, new_image: np.ndarray) -> None:
        """Sets the image data"""

    @property
    @abstractmethod
    def affine(self) -> np.ndarray:
        """Return a 4x4 numpy array with the image affine transformation"""

    @property
    @abstractmethod
    def space_units(self) -> str:
        """
        Uses the NIFTI header xyzt_units to extract the space units.
        Returns one of:
        'meter'
        'mm'
        'micron'
        """

    @space_units.setter
    def space_units(self, units: str) -> None:
        pass

    @property
    @abstractmethod
    def time_units(self) -> str:
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'sec'
        'msec'
        'usec'
        """

    @time_units.setter
    def time_units(self, units: str) -> None:
        pass

    @property
    @abstractmethod
    def voxel_size_mm(self) -> VoxelSizeType:
        """Returns the voxel size in mm"""

    @voxel_size_mm.setter
    def voxel_size_mm(self, voxel_size_mm: VoxelSizeType) -> None:
        """Returns the voxel size in mm"""

    @property
    @abstractmethod
    def time_step_seconds(self) -> float:
        """Return the time step in seconds"""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the image [x, y, z, t, etc]"""

    @staticmethod
    def _validate_time_units(units: str) -> None:
        """Checks whether the given time units is a valid string"""
        if units not in [UNITS_MICROSECONDS, UNITS_MILLISECONDS, UNITS_SECONDS]:
            raise ValueError(f'Unit "{units}" not in time units')

    @staticmethod
    def _validate_space_units(units: str) -> None:
        """Checks whether the given space units is a valid string"""
        if units not in [UNITS_MICRONS, UNITS_MILLIMETERS, UNITS_METERS]:
            raise ValueError(f'Unit "{units}" not in space units')

    @staticmethod
    def _time_units_to_seconds(units: str) -> float:
        """Returns the time units in seconds.
        Raises a ValueError if the string is an invalid time unit"""
        if units == UNITS_MILLISECONDS:
            return 1e3
        if units == UNITS_MICROSECONDS:
            return 1e6
        if units == UNITS_SECONDS:
            return 1.0
        raise ValueError(f'Unit "{units}" not in time units')

    @staticmethod
    def _space_units_to_mm(units: str) -> float:
        """Returns the space units in mm.
        Raises a ValueError if the string is an invalid space unit"""
        if units == UNITS_METERS:
            return 1e3
        if units == UNITS_MICRONS:
            return 1e-3
        if units == UNITS_MILLIMETERS:
            return 1.0
        raise ValueError(f'Unit "{units}" not in space units')


class NumpyImageContainer(BaseImageContainer):
    """A container for an ND image. Must be initialised with
    a a numpy array and some associated metadata."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image: np.ndarray,
        affine: np.ndarray = np.eye(4),
        space_units: str = UNITS_MILLIMETERS,
        time_units: str = UNITS_SECONDS,
        voxel_size: VoxelSizeType = np.array([1.0, 1.0, 1.0]),
        time_step: float = 1.0,
        **kwargs: Any,
    ):
        """Creates an image container from a numpy array. May
        provide one or more additional arguments.
        :param image: an ND numpy array containing voxel data
        :param affine: a 4x4 affine matrix (numpy array).
        :param space_units: the units across the space dimensions
        :param time_units: the units across the time dimension
        :param voxel_size: a tuple containing the voxel size (in units of space_units)
        :param time_step: the time step (in units of time_units)
        :param **kwargs: any additional arguments accepted by BaseImageContainer
        """
        self._image: np.ndarray = image
        self._affine: np.ndarray = affine
        self._space_units: str = space_units
        self._time_units: str = time_units
        self._voxel_size: npt.NDArray[np.floating] = np.array(voxel_size)
        self._time_step: float = time_step
        super().__init__(**kwargs)  # Call super last as we check member variables

    def as_numpy(self) -> NumpyImageContainer:
        """Returns self"""
        return self

    def as_nifti(self) -> NiftiImageContainer:
        """Return the NumpyImageContainer as a NiftiImageContainer."""
        new_image_container = NiftiImageContainer(
            nifti_img=nib.Nifti2Image(dataobj=self.image, affine=self._affine),
            data_domain=self.data_domain,
            image_type=self.image_type,
            metadata=deepcopy(self.metadata),
        )
        # Use the image setter as it sets the dtype in the header
        new_image_container.image = np.copy(self.image)
        # Set the units first
        new_image_container.space_units = self.space_units
        new_image_container.time_units = self.time_units
        # Now set the values
        new_image_container.voxel_size_mm = self.voxel_size_mm
        new_image_container.time_step_seconds = self.time_step_seconds
        return new_image_container

    @property
    def has_nifti(self) -> bool:
        """Returns True if the image has an associated nifti container"""
        return False

    @property
    def image(self) -> np.ndarray:
        """Return the image data as a numpy array"""
        return self._image

    @image.setter
    def image(self, new_image: np.ndarray) -> None:
        """Sets the image data - does not copy it!"""
        self._image = new_image

    @property
    def header(self) -> Union[nib.Nifti1Header, nib.Nifti2Header, None]:
        """Returns the NIFTI header if initialised from a NIFTI file,
        othewise returns None"""
        return None

    @property
    def affine(self) -> np.ndarray:
        """Return a 4x4 numpy array with the image affine transformation"""
        return self._affine

    @property
    def space_units(self) -> str:
        """
        Uses the NIFTI header xyzt_units to extract the space units.
        Returns one of:
        'meter'
        'mm'
        'micron'
        """
        return self._space_units

    @space_units.setter
    def space_units(self, units: str) -> None:
        self._validate_space_units(units)
        self._space_units = units

    @property
    def time_units(self) -> str:
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'sec'
        'msec'
        'usec'
        """
        return self._time_units

    @time_units.setter
    def time_units(self, units: str) -> None:
        self._validate_time_units(units)
        self._time_units = units

    @property
    def voxel_size_mm(self) -> VoxelSizeType:
        """Returns the voxel size in mm"""
        return self._voxel_size * self._space_units_to_mm(self.space_units)

    @voxel_size_mm.setter
    def voxel_size_mm(self, voxel_size_mm: VoxelSizeType) -> None:
        """Sets the voxel size in mm
        :param voxel_size: the voxel size in mm
        :type voxel size: list
        """
        self._voxel_size = np.array(voxel_size_mm) / self._space_units_to_mm(
            self.space_units
        )

    @property
    def time_step_seconds(self) -> float:
        """Return the time step in seconds"""
        return self._time_step * self._time_units_to_seconds(self.time_units)

    @time_step_seconds.setter
    def time_step_seconds(self, time_step: float) -> None:
        """Set the time step in seconds"""
        self._time_step = time_step / self._time_units_to_seconds(self._time_units)

    @property
    def shape(self) -> tuple:
        """Returns the shape of the image [x, y, z, t, etc]"""
        return self._image.shape


class DicomImageContainer(BaseImageContainer):
    """A container for a DICOM image"""

    def __init__(
        self,
        directory: pathlib.Path | str,
        series_instance_uid: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Load a DICOM image series from DICOM files in a given directory.
        The resulting NIfTI conversion must create a single NIfTI file, otherwise this
        process will fail.

        The DICOM information (including metadata) can be accessed using the
        `self.dicom_series` attribute. See :class:`dicom_core.nifti.load.SingleSeriesData`
        for more information.

        :param directory: the directory containing the DICOM files
        :param series_instance_uid: the DICOM series instance UID (optional - assumes
            the directory contains a single series if not specified)

        :raises ValueError: if more than one, or zero series, are found using the
            supplied config
        :raises DicomToNiftiError: if the DICOM to NIfTI conversion fails
        """
        if isinstance(directory, pathlib.Path):
            directory = directory.resolve()
        else:
            directory = pathlib.Path(directory).resolve()

        dicom_to_nifti_config = DicomToNiftiConfig(
            dcm2niix_config=Dcm2NiixConfig(),
            series_selection=SeriesSelection(
                whitelist=(
                    [SeriesInstanceUID(series_instance_uid)]
                    if series_instance_uid is not None
                    else []
                )
            ),
        )
        series_list = dicom_to_nifti(directory, dicom_to_nifti_config)
        if len(series_list) == 0:
            if series_instance_uid is not None:
                raise ValueError(
                    f"No series found with series instance uid {series_instance_uid}"
                )
            raise ValueError("No series found in DICOM directory")
        if len(series_list) > 1:
            raise ValueError("More than one series found in DICOM directory")
        self.dicom_series: SingleSeriesData = series_list[0]

        super().__init__(**kwargs)  # Call super last as we check member variables

    def as_nifti(self) -> NiftiImageContainer:
        """Return the image container as a NiftiImageContainer"""
        return NiftiImageContainer(
            nib.nifti1.load(self.dicom_series.dcm2niix_output.nifti_file)
        )

    def as_numpy(self) -> NumpyImageContainer:
        """Return the image container as a NumpyImageContainer"""
        return NiftiImageContainer(
            nib.nifti1.load(self.dicom_series.dcm2niix_output.nifti_file)
        ).as_numpy()

    @property
    def has_nifti(self) -> bool:
        """Returns True"""
        return True

    @property
    def image(self) -> np.ndarray:
        """Return the image data as a numpy array.
        Returns data in the type it is created (i.e. won't convert to float64 as
        .get_fdata() will)"""
        return self.as_numpy().image

    @image.setter
    def image(self, new_image: np.ndarray) -> None:
        """Sets the image data"""
        raise NotImplementedError

    @property
    def header(self) -> nib.nifti2.Nifti2Header | nib.nifti1.Nifti1Header:
        """Return the image header"""
        return self.as_nifti().header

    @property
    def shape(self) -> tuple:
        """Returns the shape of the image [x, y, z, t, etc]"""
        return self.as_numpy().shape

    @property
    def affine(self) -> np.ndarray:
        return self.as_numpy().affine

    @property
    def voxel_size_mm(self) -> VoxelSizeType:
        return self.as_numpy().voxel_size_mm

    @voxel_size_mm.setter
    def voxel_size_mm(self, voxel_size_mm: np.ndarray) -> None:
        """Not implemented"""
        raise NotImplementedError

    @property
    def time_step_seconds(self) -> float:
        return self.as_numpy().time_step_seconds

    @time_step_seconds.setter
    def time_step_seconds(self, time_step: float) -> None:
        """Not implemented"""
        raise NotImplementedError

    @property
    def space_units(self) -> str:
        return self.as_numpy().space_units

    @space_units.setter
    def space_units(self, space_units: str) -> None:
        """Not implemented"""
        raise NotImplementedError

    @property
    def time_units(self) -> str:
        return self.as_numpy().time_units

    @time_units.setter
    def time_units(self, time_units: str) -> None:
        """Not implemented"""
        raise NotImplementedError


class NiftiImageContainer(BaseImageContainer):
    """A container for an ND image. Must be initialised with
    a nibabel Nifti1Image or Nifti2Image"""

    def __init__(
        self,
        nifti_img: Union[
            nib.Nifti1Image, nib.Nifti2Image, nib.Nifti1Pair, nib.Nifti2Pair
        ],
        **kwargs: Any,
    ) -> None:
        """
        :param nifti_img: A nibabel Nifti1Image or Nifti2Image
        :param **kwargs: any additional arguments accepted by BaseImageContainer
        """
        self.nifti_image: Union[
            nib.Nifti1Image, nib.Nifti2Image, nib.Nifti1Pair, nib.Nifti2Pair
        ] = nifti_img
        super().__init__(**kwargs)  # Call super last as we check member variables

    @property
    def nifti_type(
        self,
    ) -> Union[
        type[nib.Nifti1Image],
        type[nib.Nifti2Image],
        type[nib.Nifti1Pair],
        type[nib.Nifti2Pair],
    ]:
        """Return the type of NIFTI data contained here"""
        return type(self.nifti_image)

    @property
    def has_nifti(self) -> bool:
        """Returns True if the image has an associated nifti container"""
        return True

    def clone(self) -> NiftiImageContainer:
        """Makes a deep copy of all member variables in a new ImageContainer"""
        return deepcopy(self)

    @property
    def image(self) -> np.ndarray:
        """Return the image data as a numpy array.
        Returns data in the type it is created (i.e. won't convert to float64 as
        .get_fdata() will)"""
        return np.asanyarray(self.nifti_image.dataobj)
        # return self.nifti_image.get_fdata()

    @image.setter
    def image(self, new_image: np.ndarray) -> None:
        """Sets the image data"""
        self.nifti_image = self.nifti_type(
            dataobj=new_image, affine=self.affine, header=self.header
        )
        # Make sure the header matches the new image data
        self.nifti_image.set_data_dtype(new_image.dtype)
        self.nifti_image.update_header()

    def as_numpy(self) -> NumpyImageContainer:
        """Return the NiftiImageContainer as a NumpyImageContainer."""
        new_image_container = NumpyImageContainer(
            image=self.image,
            affine=self.affine,
            data_domain=self.data_domain,
            image_type=self.image_type,
            metadata=deepcopy(self.metadata),
        )
        # Set the units first
        new_image_container.space_units = self.space_units
        new_image_container.time_units = self.time_units
        # Now set the values
        new_image_container.voxel_size_mm = self.voxel_size_mm
        new_image_container.time_step_seconds = self.time_step_seconds
        return new_image_container

    def as_nifti(self) -> NiftiImageContainer:
        """Returns self"""
        return self

    @property
    def header(self) -> Union[nib.Nifti1Header, nib.Nifti2Header]:
        """Returns the NIFTI header if initialised from a NIFTI file,
        othewise returns None"""
        return self.nifti_image.header

    @property
    def affine(self) -> np.ndarray:
        """Return a 4x4 numpy array with the image affine transformation"""
        return self.nifti_image.affine

    @property
    def space_units(self) -> str:
        """
        Uses the NIFTI header xyzt_units to extract the space units.
        Returns one of:
        'unknown'
        'meter'
        'mm'
        'micron'
        """
        return self.header.get_xyzt_units()[0]

    @space_units.setter
    def space_units(self, units: str) -> None:
        self._validate_space_units(units)
        self.header.set_xyzt_units(xyz=units, t=self.time_units)

    @property
    def time_units(self) -> str:
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'unknown'
        'sec'
        'msec'
        'usec'
        """
        return self.header.get_xyzt_units()[1]

    @time_units.setter
    def time_units(self, units: str) -> None:
        self._validate_time_units(units)
        self.header.set_xyzt_units(xyz=self.space_units, t=units)

    @property
    def voxel_size_mm(self) -> VoxelSizeType:
        """Returns the voxel size in mm"""
        return self.header["pixdim"][1:4] * self._space_units_to_mm(self.space_units)

    @voxel_size_mm.setter
    def voxel_size_mm(self, voxel_size_mm: VoxelSizeType):
        """Sets the voxel size in mm
        :param voxel_size: the voxel size in mm
        :type voxel size: list
        """
        self.header["pixdim"][1:4] = np.array(voxel_size_mm) / self._space_units_to_mm(
            self.space_units
        )

    @property
    def time_step_seconds(self) -> float:
        """Return the time step in seconds"""
        return self.header["pixdim"][4] * self._time_units_to_seconds(self.time_units)

    @time_step_seconds.setter
    def time_step_seconds(self, time_step: float) -> None:
        """Set the time step in seconds"""
        self.header["pixdim"][4] = time_step / self._time_units_to_seconds(
            self.time_units
        )

    @property
    def shape(self) -> tuple:
        """Returns the shape of the image [x, y, z, t, etc]"""
        return self.nifti_image.shape
