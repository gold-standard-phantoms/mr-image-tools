"""Transform resample image filter"""

import nibabel as nib

from mrimagetools.v2.containers.image import BaseImageContainer, NumpyImageContainer
from mrimagetools.v2.filters.basefilter import FilterInputValidationError
from mrimagetools.v2.filters.filter_block import BaseFilter
from mrimagetools.v2.filters.resample_filter import ResampleFilter
from mrimagetools.v2.utils.resampling import transform_resample_affine
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    from_list_validator,
    greater_than_validator,
    isinstance_validator,
    range_inclusive_validator,
)


class TransformResampleImageFilter(BaseFilter):
    r"""A filter that transforms and resamples an image in world space.  The field of view (FOV)
    of the resampled image is the same as the FOV of the input image.

    Conventions are for RAS+ coordinate systems only

    **Inputs**

    Input Parameters are all keyword arguments for the
    :class:`TransformResampleImageFilter.add_inputs()` member function.
    They are also accessible via class constants,
    for example :class:`TransformResampleImageFilter.KEY_ROTATION`

    :param 'image': The input image
    :type 'image': BaseImageContainer
    :param 'translation': :math:`[\Delta r_x,\Delta r_y,\Delta r_z]`
        amount to translate along the x, y and z axes. defaults to (0, 0, 0)
    :type translation: Tuple[float, float, float], optional
    :param 'rotation': :math:`[\theta_x,\theta_y,\theta_z]`
        angles to rotate about the x, y and z axes in degrees(-180 to 180 degrees inclusive),
        defaults to (0, 0, 0)
    :type 'rotation': Tuple[float, float, float], optional
    :param 'rotation_origin': :math:`[x_r,y_r,z_r]`
        coordinates of the point to perform rotations about, defaults to (0, 0, 0)
    :type 'rotation_origin': Tuple[float, float, float], optional
    :param target_shape: :math:`[L_t,M_t,N_t]` target shape for the resampled image
    :type target_shape: Tuple[int, int, int]
    :param 'interpolation': Defines the interpolation method for the resampling:

        :'continuous': order 3 spline interpolation (default method for ResampleFilter)
        :'linear': order 1 linear interpolation
        :'nearest': nearest neighbour interpolation

    :type 'interpolation': str, optional

    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`TransformResampleImageFilter.outputs` with the following entries

    :param 'image': The input image, resampled in accordance with the specified shape
        and applied world-space transformation.
    :type 'image': BaseImageContainer

    The metadata property of the :class:`TransformResampleImageFilter.outputs["image"]` is
    updated with the field ``voxel_size``, corresponding to the size of each voxel.


    The output image is resampled according to the target affine:

    .. math::

        &\mathbf{A}=(\mathbf{T(\Delta r_{\text{im}})}\mathbf{S}\mathbf{T(\Delta r)}
        \mathbf{T(r_0)}\mathbf{R}\mathbf{T(r_0)}^{-1})^{-1}\\
        \text{where,}&\\
        & \mathbf{T(r_0)} = \mathbf{T}(x_r, y_r, z_r)=
        \text{Affine for translation to rotation centre}\\
        & \mathbf{T(\Delta r)} = \mathbf{T}(\Delta r_x, \Delta r_y, \Delta r_z)=
        \text{Affine for translation of image in world space}\\
        & \mathbf{T(\Delta r_{\text{im}})} = \mathbf{T}(x_0/s_x,y_0/s_y,z_0/s_z)^{-1}
        =\text{Affine for translation to the input image origin} \\
        &\mathbf{T} =  \begin{pmatrix} 1 & 0 & 0 & \Delta x \\ 0 & 1& 0 & \Delta y \\
        0 & 0 & 1& \Delta z \\ 0& 0 & 0& 1 \end{pmatrix}=\text{translation matrix}\\
        &\mathbf{S} = \begin{pmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\
        0 & 0 & s_z & 0 \\ & 0 & 0& 1 \end{pmatrix}=\text{scaling matrix}\\
        & [s_x, s_y, s_z] = \frac{[L_t,M_t,N_t]}{[v_x, v_y, v_z]\cdot[L_i,M_i,N_i]}\\
        & \text{divisions and multiplications are element-wise (Hadamard)}\\
        & [L_i, M_i, N_i] = \text{shape of the input image}\\
        & [v_x, v_y, v_z] = \text{voxel dimensions of the input image}\\
        & [x_0, y_0, z_0] = \text{input image origin coordinates (vector part of input
        image's affine)}\\
        &\mathbf{R} = \mathbf{R_z} \mathbf{R_y} \mathbf{R_x} =
        \text{Affine for rotation of image in world space}\\
        &\mathbf{R_x} = \begin{pmatrix} 1 & 0 & 0 & 0\\ 0 & \cos{\theta_x}& -\sin{\theta_x} & 0\\
        0 & \sin{\theta_x} & \cos{\theta_x}& 0\\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{rotation about x matrix}\\
        &\mathbf{R_y} = \begin{pmatrix} \cos{\theta_y} & 0 & \sin{\theta_y} & 0\\
         0 & 1 & 0 & 0\\ -\sin{\theta_y} & 0 & \cos{\theta_y}& 0\\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{rotation about y matrix}\\
        &\mathbf{R_z} = \begin{pmatrix} \cos{\theta_z}& -\sin{\theta_z} & 0 & 0\\
        \sin{\theta_z} & \cos{\theta_z}& 0 &0\\ 0& 0& 1 & 0\\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{rotation about z matrix}\\

    After resampling the output image's affine is modified to only contain the scaling:

    .. math::

        \mathbf{A_{\text{new}}} = (\mathbf{T(\Delta r_{\text{im}})}\mathbf{S})^{-1}

    """
    KEY_TARGET_SHAPE = "target_shape"
    KEY_ROTATION_ORIGIN = "rotation_origin"
    KEY_ROTATION = "rotation"
    KEY_TRANSLATION = "translation"
    KEY_INTERPOLATION = ResampleFilter.KEY_INTERPOLATION
    KEY_IMAGE = "image"
    VOXEL_SIZE = "voxel_size"
    INTERPOLATION_LIST = ResampleFilter.INTERPOLATION_LIST

    def __init__(self) -> None:
        super().__init__(name="Transform and Resample Image")

    def _run(self) -> None:
        r"""Transforms the image in world-space, then creates a resampled image
        with the specified acquisition shape.
        """

        input_image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        translation = self.inputs[self.KEY_TRANSLATION]
        rotation = self.inputs[self.KEY_ROTATION]
        rotation_origin = self.inputs[self.KEY_ROTATION_ORIGIN]
        target_shape = self.inputs[self.KEY_TARGET_SHAPE]
        interpolation = self.inputs.get(self.KEY_INTERPOLATION)

        (
            target_affine_with_motion,
            target_affine_no_motion,
        ) = transform_resample_affine(
            input_image, translation, rotation, rotation_origin, target_shape
        )

        resample_filter = ResampleFilter()
        resample_filter.add_input(ResampleFilter.KEY_IMAGE, input_image)
        resample_filter.add_input(ResampleFilter.KEY_AFFINE, target_affine_with_motion)
        resample_filter.add_input(ResampleFilter.KEY_SHAPE, target_shape)
        resample_filter.add_input(ResampleFilter.KEY_INTERPOLATION, interpolation)
        resample_filter.run()

        self.outputs[self.KEY_IMAGE] = resample_filter.outputs[ResampleFilter.KEY_IMAGE]

        if isinstance(self.inputs[self.KEY_IMAGE], NumpyImageContainer):
            self.outputs[self.KEY_IMAGE].affine = target_affine_no_motion
        else:
            self.outputs[self.KEY_IMAGE].nifti_image.set_sform(target_affine_no_motion)

        # The ResampleFilter will have copied through the metadata from the input image, however
        # the voxel size should be recalculated as the output image doesn't have any rotations
        # applied. update metadata with the voxel sizes based on the updated affine
        self.outputs[self.KEY_IMAGE].metadata.voxel_size = list(
            nib.affines.voxel_sizes(self.outputs[self.KEY_IMAGE].affine)
        )

    def _validate_inputs(self) -> None:
        """Checks that the inputs meet their validation criteria
        `'object'` must be derived from BaseImageContainer
        `'target_shape'` (optional)must be a Tuple of ints of length 3, values > 0
        `'rotation'` (optional) must be a Tuple of floats of length 3, each value -180 to 180
        inclusive, default (optional) = (0.0, 0.0, 0.0)
        `'rotation_origin'` (optional) must be a Tuple of floats of length 3,
        default = (0.0, 0.0, 0.0)
        `'translation'` (optional) must be a Tuple of floats of length 3, default = (0.0, 0.0, 0.0)
        `'interpolation'` must be a string and either 'continuous',
        'linear' or 'nearest'
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_ROTATION: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                        for_each_validator(range_inclusive_validator(-180, 180)),
                    ],
                    optional=True,
                    default_value=(0.0, 0.0, 0.0),
                ),
                self.KEY_ROTATION_ORIGIN: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                    default_value=(0.0, 0.0, 0.0),
                ),
                self.KEY_TRANSLATION: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                    default_value=(0.0, 0.0, 0.0),
                ),
                self.KEY_TARGET_SHAPE: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(int)),
                        for_each_validator(greater_than_validator(0)),
                    ],
                    optional=True,
                    default_value=(9999, 9999, 9999),
                ),
                self.KEY_INTERPOLATION: Parameter(
                    validators=[
                        isinstance_validator(str),
                        from_list_validator(ResampleFilter.INTERPOLATION_LIST),
                    ],
                    optional=True,
                ),
            }
        )

        # validate, returning a dictionary which also includes default parameters
        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # Further validation that can't be handled by the parameter validator

        if new_params[self.KEY_TARGET_SHAPE] == (9999, 9999, 9999):
            new_params[self.KEY_TARGET_SHAPE] = self.inputs[self.KEY_IMAGE].shape

        # Check that the tuple self.KEY_ROTATION's length is 3
        if len(new_params[self.KEY_ROTATION]) != 3:
            raise FilterInputValidationError

        # Check that the tuple self.KEY_ROTATION_ORIGIN's length is 3
        if len(new_params[self.KEY_ROTATION_ORIGIN]) != 3:
            raise FilterInputValidationError

        # Check that the tuple self.KEY_TRANSLATION's length is 3
        if len(new_params[self.KEY_TRANSLATION]) != 3:
            raise FilterInputValidationError

        # Check that the tuple self.KEY_SCALE's length is 3
        if len(new_params[self.KEY_TARGET_SHAPE]) != 3:
            raise FilterInputValidationError

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}
