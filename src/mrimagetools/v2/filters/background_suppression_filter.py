"""Background Suppression Filter"""
from typing import List, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from mrimagetools.v2.containers.image import COMPLEX_IMAGE_TYPE, BaseImageContainer
from mrimagetools.v2.containers.image_metadata import ImageMetadata
from mrimagetools.v2.filters.basefilter import BaseFilter, FilterInputValidationError
from mrimagetools.v2.utils.typing import typed
from mrimagetools.v2.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    from_list_validator,
    greater_than_equal_to_validator,
    greater_than_validator,
    isinstance_validator,
    or_validator,
    range_inclusive_validator,
)


class BackgroundSuppressionFilter(BaseFilter):
    """A filter that simulates a background suppression
    pulse sequence on longitudinal magnetisation. It can either
    use explicitly supplied pulse timings, or calculate optimised
    pulse timings for specified T1s.

    **Inputs**

    Input Parameters are all keyword arguments for the
    :class:`BackgroundSuppressionFilter.add_inputs()` member function.
    They are also accessible via class constants,
    for example :class:`CombineTimeSeriesFilter.KEY_T1`.

    :param 'mag_z': Image of the initial longitudinal magnetisation.
      Image data must not be a complex data type.
    :type 'mag_z': BaseImageContainer
    :param 't1': Image of the longitudinal relaxation time. Image
      data must be greater than 0 and non-compex. Also its shape should
      match the shape of ``'mag_z'``.
    :type 't1': BaseImageContainer
    :param 'sat_pulse_time': The time, in seconds between the saturation
      pulse and the imaging excitation pulse. Must be greater than 0.
    :type 'sat_pulse_time': float
    :param 'inv_pulse_times': The inversion times for each inversion pulse,
      defined as the spacing between the inversion pulse and the imaging
      excitation pulse. Must be greater than 0. If omitted then optimal
      inversion times will be calculated for ``'num_inv_pulses'`` number
      of pulses, and the T1 times given by ``'t1_opt'``.
    :type 'inv_pulse_times': list[float], optional
    :param 't1_opt': T1 times, in seconds to optimise the pulse inversion
      times for. Each must be greater than 0, and if omitted then the
      unique values in the input ``t1`` will be used.
    :type 't1_opt': list[float]
    :param 'mag_time': The time, in seconds after the saturation pulse to
      sample the longitudinal magnetisation. The output magnetisation will
      only reflect the pulses that will have run by this time. Must be
      greater than 0. If omitted, defaults to the same value
      as ``'sat_pulse_time'``. If ``'mag_time'`` is longer than
      ``'sat_pulse_time'``, then this difference will be added to both
      ``'sat_pulse_time'`` and also ``'inv_pulse_times'`` (regardless of
      whether this has been supplied as an input or optimised values calculated).
      If the pulse timings already include an added delay to ensure the
      magnetisation is positive then this parameter should be omitted.
    :type 'mag_time': float
    :param 'num_inv_pulses': The number of inversion pulses to calculate
      optimised timings for. Must be greater than 0, and this parameter
      must be present if ``'inv_pulse_times'`` is omitted.
    :type 'num_inv_pulses: int
    :param 'pulse_efficiency': Defines the efficiency of the inversion
      pulses. Can take the values:

        :'realistic': Pulse efficiencies are calculated according to a
          model based on the T1. See
          :class:`BackgroundSuppressionFilter.calculate_pulse_efficiency`
          for details on implementation.
        :'ideal': Inversion pulses are 100% efficient.
        :-1 to 0: The efficiency is defined explicitly, with -1 being full
          inversion and 0 no inversion.

    :type 'pulse_efficiency': str or float

    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`BackgroundSuppressionFilter.outputs` with
    the following entries:

    :param 'mag_z': The longitudinal magnetisation at t=``mag_time``.
    :type 'mag_z': BaseImageContainer
    :param 'inv_pulse_times': The inversion pulse timings.
    :type 'inv_pulse_times': list[float]

    **Metadata**

    The output ``'mag_z'`` inherits metadata from the input ``'mag_z'``, and then has the
    following entries appended:

        :background_suppression: ``True``
        :background_suppression_inv_pulse_timing: ``'inv_pulse_times'``
        :background_suppression_sat_pulse_timing: ``'sat_pulse_time'``
        :background_suppression_num_pulses: The number of inversion pulses.


    **Background Suppression Model**

    Details on the model implemented can be found in
    :class:`BackgroundSuppressionFilter.calculate_mz`

    Details on how the pulse timings are optimised can be found in
    :class:`BackgroundSuppressionFilter.optimise_inv_pulse_times`

    """

    KEY_MAG_Z = "mag_z"
    KEY_T1 = "t1"
    KEY_SAT_PULSE_TIME = "sat_pulse_time"
    KEY_INV_PULSE_TIMES = "inv_pulse_times"
    KEY_T1_OPT = "t1_opt"
    KEY_MAG_TIME = "mag_time"
    KEY_NUM_INV_PULSES = "num_inv_pulses"
    KEY_PULSE_EFFICIENCY = "pulse_efficiency"

    M_BACKGROUND_SUPPRESSION = "background_suppression"
    M_BSUP_INV_PULSE_TIMING = "background_suppression_inv_pulse_timing"
    M_BSUP_SAT_PULSE_TIMING = "background_suppression_sat_pulse_timing"
    M_BSUP_NUM_PULSES = "background_suppression_num_pulses"

    EFF_IDEAL = "ideal"
    EFF_REALISTIC = "realistic"

    def __init__(self) -> None:
        super().__init__(name="Background Suppression Filter")

    def _run(self) -> None:
        """Runs the filter"""
        mag_z: BaseImageContainer = self.inputs[self.KEY_MAG_Z]
        t1: BaseImageContainer = self.inputs[self.KEY_T1]
        sat_pulse_time = self.inputs[self.KEY_SAT_PULSE_TIME]

        mag_time: float
        if self.inputs.get(self.KEY_MAG_TIME) is None:
            mag_time = typed(sat_pulse_time, float)
        else:
            mag_time = typed(self.inputs.get(self.KEY_MAG_TIME), float)

        inv_eff: Optional[Union[float, np.ndarray]] = None
        # determine the pulse efficiency mode
        if self.inputs[self.KEY_PULSE_EFFICIENCY] == self.EFF_IDEAL:
            inv_eff = -1.0
        elif self.inputs[self.KEY_PULSE_EFFICIENCY] == self.EFF_REALISTIC:
            # pulse efficiency calculation with static method
            inv_eff = self.calculate_pulse_efficiency(t1.image)
        elif isinstance(self.inputs[self.KEY_PULSE_EFFICIENCY], float):
            inv_eff = self.inputs[self.KEY_PULSE_EFFICIENCY]

        # determine whether the inversion pulse times have been provided
        # or if optimised times need to be calculated
        inv_pulse_times: np.ndarray
        if self.inputs.get(self.KEY_INV_PULSE_TIMES) is None:
            if inv_eff is None:
                raise ValueError("inv_eff cannot be None")

            # calculation required: minimise the least squares problem
            # argmin(||Mz(t=sat_time)||^2 + sum(Mz(t=sat_time) < 0))
            # to find the inversion pulse
            # times for the given T1's, whilst ensuring that the magnetisation
            # is always positive.
            t1_opt = self.inputs[self.KEY_T1_OPT]
            num_inv_pulses = self.inputs[self.KEY_NUM_INV_PULSES]
            # if `pulse_efficiency` is 'realistic' then calculate the pulse
            # efficiencies for the t1's to optimise over
            pulse_eff_opt: Union[float, np.ndarray]
            if self.inputs[self.KEY_PULSE_EFFICIENCY] == self.EFF_REALISTIC:
                pulse_eff_opt = self.calculate_pulse_efficiency(t1_opt)
            else:
                # otherwise just use pulse_eff
                pulse_eff_opt = inv_eff

            result = self.optimise_inv_pulse_times(
                sat_pulse_time, t1_opt, pulse_eff_opt, num_inv_pulses
            )
            inv_pulse_times = result.x
        else:
            inv_pulse_times = np.asarray(self.inputs[self.KEY_INV_PULSE_TIMES])

        # if mag_time > sat_pulse_time add the difference to sat_pulse_time
        # and inv_pulse_times
        if mag_time > sat_pulse_time:
            post_sat_delay = mag_time - sat_pulse_time
            sat_pulse_time += post_sat_delay
            inv_pulse_times += post_sat_delay

        # calculate the longitudinal magnetisation at mag_time based on
        # the inversion pulse times
        self.outputs[self.KEY_MAG_Z] = mag_z.clone()
        if inv_eff is None:
            raise ValueError("inv_eff cannot be None")
        self.outputs[self.KEY_MAG_Z].image = self.calculate_mz(
            self.outputs[self.KEY_MAG_Z].image,
            t1.image,
            inv_pulse_times,
            sat_pulse_time,
            mag_time,
            inv_eff,
        )
        metadata = ImageMetadata(
            background_suppression=True,
            background_suppression_inv_pulse_timing=inv_pulse_times.tolist(),
            background_suppression_sat_pulse_timing=mag_time,
            background_suppression_num_pulses=np.asarray(inv_pulse_times).size,
        )
        # merge the metadata
        self.outputs[self.KEY_MAG_Z].metadata = ImageMetadata(
            **{
                **self.outputs[self.KEY_MAG_Z].metadata.dict(exclude_none=True),
                **metadata.dict(exclude_none=True),
            }
        )

        self.outputs[self.KEY_INV_PULSE_TIMES] = inv_pulse_times

    def _validate_inputs(self) -> None:
        """Checks that the inputs meet their validation criteria
        'mag_z': BaseImageContainer and image_type != COMPLEX_IMAGE_TYPE
        't1': BaseImageContainer, >0 and image_type != COMPLEX_IMAGE_TYPE, shape should
        match
        'sat_pulse_time': float, >0
        'inv_pulse_times': list[float], each >0, optional,
        't1_opt': list[float], each >0,  optional
        'mag_time': float, >0, optional
        'num_inv_pulses': int, >0,  must be present if
            'inv_pulse_times' is omitted
        'pulse_efficiency': float or str, optional, default "ideal":
            str: "realistic" or "ideal"
            float: between -1 and 0 inclusive
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_MAG_Z: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
                self.KEY_T1: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_SAT_PULSE_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_validator(0),
                    ]
                ),
                self.KEY_INV_PULSE_TIMES: Parameter(
                    validators=[
                        for_each_validator(greater_than_validator(0)),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                ),
                self.KEY_MAG_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_validator(0),
                    ],
                    optional=True,
                ),
                self.KEY_PULSE_EFFICIENCY: Parameter(
                    validators=[
                        isinstance_validator((str, float)),
                        or_validator(
                            [
                                from_list_validator(
                                    [self.EFF_IDEAL, self.EFF_REALISTIC]
                                ),
                                range_inclusive_validator(-1, 0),
                            ]
                        ),
                    ],
                    default_value=self.EFF_IDEAL,
                ),
            }
        )

        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}
        keys_of_images = [self.KEY_MAG_Z, self.KEY_T1]

        # check the images - shapes must match and the data cannot be complex
        list_of_image_shapes = [self.inputs[key].shape for key in keys_of_images]
        if list_of_image_shapes.count(list_of_image_shapes[0]) != len(
            list_of_image_shapes
        ):
            raise FilterInputValidationError(
                [
                    "Input image shapes do not match.",
                    [
                        f"{keys_of_images[i]}: {list_of_image_shapes[i]}, "
                        for i in range(len(list_of_image_shapes))
                    ],
                ]
            )
        # Check that all the input images are not of image_type == "COMPLEX_IMAGE_TYPE"
        for key in keys_of_images:
            if self.inputs[key].image_type == COMPLEX_IMAGE_TYPE:
                raise FilterInputValidationError(
                    f"{key} has image type {COMPLEX_IMAGE_TYPE}, this is not supported"
                )

        # validate parameters for the case where 'inv_pulse_times' is not present,
        # therefore the optimal pulse times need to be calculated
        if self.inputs.get(self.KEY_INV_PULSE_TIMES) is None:
            calc_inv_times_validator = ParameterValidator(
                parameters={
                    self.KEY_T1_OPT: Parameter(
                        validators=[
                            for_each_validator(greater_than_validator(0)),
                            for_each_validator(isinstance_validator(float)),
                        ],
                        default_value=np.trim_zeros(
                            np.unique(self.inputs[self.KEY_T1].image).tolist()
                        ),
                    ),
                    self.KEY_NUM_INV_PULSES: Parameter(
                        validators=[
                            isinstance_validator(int),
                            greater_than_validator(0),
                        ],
                    ),
                },
            )
            p = calc_inv_times_validator.validate(
                self.inputs, error_type=FilterInputValidationError
            )
            # merge parameters
            new_params = {**new_params, **p}
            self.inputs = {**self._i, **new_params}

    @staticmethod
    def calculate_mz(
        initial_mz: Union[float, np.ndarray],
        t1: Union[float, np.ndarray],
        inv_pulse_times: Union[list[float], np.ndarray],
        sat_pulse_time: float,
        mag_time: float,
        inv_eff: Union[float, np.ndarray],
        sat_eff: Union[np.ndarray, float] = 1.0,
    ) -> np.ndarray:
        r"""Calculates the longitudinal magnetisation after
        a sequence of background suppression pulses :cite:p:`Mani1997`

        :param initial_mz: The initial longitudinal magnetisation, :math:`M_z(t=0)`
        :type initial_mz: np.ndarray
        :param t1: The longitudinal relaxation time, :math:`T_1`
        :type t1: np.ndarray
        :param inv_pulse_times: Inversion pulse times, with respect
          to the imaging excitation pulse, :math:`\{ \tau_i, \tau_{i+1}... \tau_{M-1}, \tau_M \}`
        :type inv_pulse_times: list[float]
        :param mag_time: The time at which to calculate the
          longitudinal magnetisation, :math:`t`, cannot be greater than
          sat_pulse_time
        :type mag_time: float
        :param sat_pulse_time: The time between the saturation pulse
          and the imaging excitation pulse, :math:`Q`.
        :type sat_pulse_time: float
        :param inv_eff: The efficiency of the inversion pulses, :math:`\chi`
          .-1 is complete inversion.
        :type inv_eff: np.ndarray
        :param sat_eff: The efficiency of the saturation pulses, :math:`\psi`. 1 is
          full saturation.
        :type sat_eff: np.ndarray
        :return: The longitudinal magnetisation after the background
          suppression sequence
        :rtype: np.ndarray

        **Equation**

        The longitudinal magnetisation at time :math:`t` after the start of a
        background suppression sequence has started is calculated using the equation
        below. Only pulses that have run at time :math:`t` contribute to the
        calculated magnetisation.

        .. math::

            \begin{align}
                &M_z(t)= M_z(t=0)\cdot (1 + ((1-\psi)-1)\chi^n e^{-\frac{t}{T_1} }+ \sum
                \limits_{m=1}^n(\chi^m - \chi^{m-1}) e^{-\frac{\tau_m}{T_1}})\\
                &\text{where}\\
                &M_z(t)=\text{longitudinal magnetisation at time t}\\
                &Q=\text{the delay between the saturation pulse and imaging excitation pulse}\\
                &\psi=\text{saturation pulse efficiency}, 0 \leq \psi \leq 1\\
                &\chi=\text{inversion pulse efficiency}, -1 \leq \chi \leq 0\\
                &\tau_m = \text{inversion time of the }m^\text{th}\text{ pulse}\\
                &T_1=\text{longitudinal relaxation time}\\
            \end{align}

        """
        # check that initial_mz, t1 and pulse_eff are broadcastable
        np.broadcast(initial_mz, t1, inv_eff, sat_eff)

        # check mag_time is not larger than sat_pulse_time
        if mag_time > sat_pulse_time:
            raise ValueError(
                "argument 'mag_time' must not be greater than sat_pulse_time"
            )

        # sort the inversion pulse times into ascending order
        # inv_pulse_times = np.sort(inv_pulse_times)
        # determine the number of inversion pulses that will have played
        # out by t=mag_time
        inv_pulse_times = np.asarray(inv_pulse_times)
        inv_pulse_times = inv_pulse_times[mag_time > sat_pulse_time - inv_pulse_times]
        num_pulses = len(inv_pulse_times)

        return initial_mz * (
            1
            + ((1 - sat_eff) - 1)
            * inv_eff**num_pulses
            * np.exp(-np.divide(mag_time, t1, out=np.zeros_like(t1), where=t1 != 0))
            + np.sum(
                [
                    ((inv_eff ** (m + 1)) - (inv_eff**m))
                    * np.exp(-np.divide(tm, t1, out=np.zeros_like(t1), where=t1 != 0))
                    for m, tm in enumerate(inv_pulse_times)
                ],
                0,
            )
        )

    @staticmethod
    def calculate_pulse_efficiency(t1: np.ndarray) -> np.ndarray:
        r"""Calculates the pulse efficiency per T1 based on a polynomial
        fit :cite:p:`Maleki2011`.

        :param t1: t1 times to calculate the pulse efficiencies for, seconds.
        :type t1: np.ndarray
        :return: The pulse efficiencies, :math:`\chi`
        :rtype: np.ndarray

        **Equation**

        .. math::

            \newcommand{\sn}[2]{#1 {\times} 10 ^ {#2}}
            \chi=
            \begin{cases}
            -0.998 & 250 \leq T_1 <450\\
            - \left ( \begin{align} \sn{-2.245}{-15}T_1^4 \\
            + \sn{2.378}{-11}T_1^3 \\
            - \sn{8.987}{-8}T_1^2\\
            + \sn{1.442}{-4}T_1\\
            + \sn{9.1555}{-1} \end{align}\right ) & 450 \leq T_1 < 2000\\
            -0.998 & 2000 \leq T_1 < 4200
            \end{cases}

        """
        # convert t1 to a ndarray for consistency
        t1 = np.asarray(t1)
        pulse_eff = np.zeros_like(t1)
        t1 = t1 * 1000  # paper gives polynomial based on ms, so convert t1 to ms
        mid_t1 = (t1 >= 450.0) & (t1 <= 2000.0)
        pulse_eff[(t1 >= 250.0) & (t1 < 450.0)] = -0.998
        pulse_eff[mid_t1] = -(
            (-2.245e-15) * t1[mid_t1] ** 4
            + (2.378e-11) * t1[mid_t1] ** 3
            - (8.987e-8) * t1[mid_t1] ** 2
            + (1.442e-4) * t1[mid_t1]
            + (9.1555e-1)
        )
        pulse_eff[(t1 > 2000.0) & (t1 <= 4200.0)] = -0.998
        return pulse_eff

    @staticmethod
    def optimise_inv_pulse_times(
        sat_time: float,
        t1: np.ndarray,
        pulse_eff: Union[np.ndarray, float],
        num_pulses: int,
        method: str = "Nelder-Mead",
    ) -> OptimizeResult:
        r"""Calculates optimised inversion pulse times
        for a background suppression pulse sequence.

        :param sat_time: The time, in seconds between the saturation pulse and
          the imaging excitation pulse, :math:`Q`.
        :type sat_time: float
        :param t1: The longitudinal relaxation times to optimise the pulses for, :math:`T_1`.
        :type t1: np.ndarray
        :param pulse_eff: The inversion pulse efficiency, :math:`\chi`. corresponding to each
          ``t1`` entry.
        :type pulse_eff: np.ndarray
        :param num_pulses: The number of inversion pulses to optimise times for, :math:`N`.
          Must be greater than 0.
        :type num_pulses: int
        :param method: The optimisation method to use, see
          :class:`scipy.optimize.minimize` for more details. Defaults to "Nelder-Mead".
        :type method: str, optional
        :raises ValueError: If the number of pulses is less than 1.
        :return: The result from the optimisation
        :rtype: OptimizeResult

        **Equation**

        A set of optimimal inversion times, :math:`\{ \tau_i, \tau_{i+1}... \tau_{M-1}, \tau_M \}`
        are calculated by minimising the sum-of-squares of the magnetisation of all the T1 species
        in ``t1_opt``:

        .. math::

            \begin{align}
                &\min \left (\sum\limits_i^N M_z^2(t=Q, T_{1,i},\chi, \psi, \tau)
                + \sum\limits_i^N
                \begin{cases}
                1 & M_z(t=Q, T_{1,i},\chi, \psi, \tau) < 0\\
                0 & M_z(t=Q, T_{1,i},\chi, \psi, \tau) \geq 0
                \end{cases}
                \right) \\
                &\text{where}\\
                &N = \text{The number of $T_1$ species to optimise for}\\
            \end{align}

        """
        if not num_pulses > 0:
            raise ValueError("num_pulses must be greater than 0")

        x0 = np.ones((num_pulses,))
        # create the objective function to optimise: ||Mz(t=sat_time)||^2 + sum(Mz(t=sat_time) < 0)
        fun = lambda x: np.sum(
            BackgroundSuppressionFilter.calculate_mz(
                np.ones_like(t1), t1, x, sat_time, sat_time, pulse_eff
            )
            ** 2
        ) + np.sum(
            BackgroundSuppressionFilter.calculate_mz(
                np.ones_like(t1), t1, x, sat_time, sat_time, pulse_eff
            )
            < 0
        )
        # perform the optimisation
        result = minimize(fun, x0, method=method)
        return result
