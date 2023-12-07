"""
ASL quantification functions.
"""

from typing import Union

import numpy as np
from scipy.optimize import curve_fit

from mrimagetools.filters.gkm_filter import calculate_delta_m_gkm

M0_TOL = 1e-6


def asl_quant_wp_casl(
    control: np.ndarray,
    label: np.ndarray,
    m0: np.ndarray,
    lambda_blood_brain: float,
    label_duration: float,
    post_label_delay: float,
    label_efficiency: float,
    t1_arterial_blood: float,
) -> np.ndarray:
    r"""
    Performs ASL quantification using the White Paper equation for
    (pseudo)continuous ASL :cite:p:`Alsop2014`.

    .. math::
        &f = \frac{6000 \cdot\ \lambda \cdot (\text{SI}_{\text{control}} -
        \text{SI}_{\text{label}}) \cdot
        e^{\frac{\text{PLD}}{T_{1,b}}}}{2 \cdot \alpha \cdot T_{1,b} \cdot \text{SI}_{\text{M0}}
        \cdot (1-e^{-\frac{\tau}{T_{1,b}}})}\\
        \text{where,}\\
        &f = \text{perfusion rate in ml/100g/min}\\
        &\text{SI}_{\text{control}} = \text{control image signal}\\
        &\text{SI}_{\text{label}} = \text{label image signal}\\
        &\text{SI}_{\text{M0}} = \text{equilibrium magnetision signal}\\
        &\tau = \text{label duration}\\
        &\text{PLD} = \text{Post Label Delay}\\
        &T_{1,b} = \text{longitudinal relaxation time of arterial blood}\\
        &\alpha = \text{labelling efficiency}\\
        &\lambda = \text{blood-brain partition coefficient}\\

    :param control: control image, :math:`\text{SI}_{\text{control}}`
    :type control: np.ndarray
    :param label: label image :math:`\text{SI}_{\text{label}}`
    :type label: np.ndarray
    :param m0: equilibrium magnetisation image, :math:`\text{SI}_{\text{M0}}`
    :type m0: np.ndarray
    :param lambda_blood_brain: blood-brain partition coefficient in ml/g, :math:`\lambda`
    :type lambda_blood_brain: float
    :param label_duration: label duration in seconds, :math:`\tau`
    :type label_duration: float
    :param post_label_delay: duration between the end of the label pulse
        and the start of the image acquisition in seconds, :math:`\text{PLD}`
    :type post_label_delay: float
    :param label_efficiency: labelling efficiency, :math:`\alpha`
    :type label_efficiency: float
    :param t1_arterial_blood: longitudinal relaxation time of arterial
        blood in seconds, :math:`T_{1,b}`
    :type t1_arterial_blood: float
    :return: the perfusion rate in ml/100g/min, :math:`f`
    :rtype: np.ndarray
    """
    control = np.asarray(control)
    label = np.asarray(label)
    m0 = np.asarray(m0)
    return np.divide(
        6000
        * lambda_blood_brain
        * (control - label)
        * np.exp(post_label_delay / t1_arterial_blood),
        2
        * label_efficiency
        * t1_arterial_blood
        * m0
        * (1 - np.exp(-label_duration / t1_arterial_blood)),
        out=np.zeros_like(m0),
        where=np.abs(m0) >= M0_TOL,
    )


def asl_quant_wp_pasl(
    control: np.ndarray,
    label: np.ndarray,
    m0: np.ndarray,
    lambda_blood_brain: float,
    bolus_duration: float,
    inversion_time: float,
    label_efficiency: float,
    t1_arterial_blood: float,
) -> np.ndarray:
    r"""
    Performs ASL quantification using the White Paper equation for
    pulsed ASL :cite:p:`Alsop2014`.

    .. math::
        &f = \frac{6000 \cdot\ \lambda \cdot (\text{SI}_{\text{control}}
        - \text{SI}_{\text{label}}) \cdot
        e^{\frac{\text{TI}}{T_{1,b}}}}{2 \cdot \alpha \cdot \text{TI}_1
        \cdot \text{SI}_{\text{M0}}}\\
        \text{where,}\\
        &f = \text{perfusion rate in ml/100g/min}\\
        &\text{SI}_{\text{control}} = \text{control image signal}\\
        &\text{SI}_{\text{label}} = \text{label image signal}\\
        &\text{SI}_{\text{M0}} = \text{equilibrium magnetision signal}\\
        &\text{TI} = \text{inversion time}\\
        &\text{TI}_1 = \text{bolus duration}\\
        &T_{1,b} = \text{longitudinal relaxation time of arterial blood}\\
        &\alpha = \text{labelling efficiency}\\
        &\lambda = \text{blood-brain partition coefficient}\\

    :param control: control image, :math:`\text{SI}_{\text{control}}`
    :type control: np.ndarray
    :param label: label image, :math:`\text{SI}_{\text{label}}`
    :type label: np.ndarray
    :param m0: equilibrium magnetisation image, :math:`\text{SI}_{\text{M0}}`
    :type m0: np.ndarray
    :param lambda_blood_brain: blood-brain partition coefficient in ml/g,
        :math:`\lambda`
    :type lambda_blood_brain: float
    :param inversion_time: time between the inversion pulse and the start
        of the image acquisition in seconds, :math:`\text{TI}`
    :type inversion_time: float
    :param bolus_duration: temporal duration of the labelled bolus in
        seconds, defined as the duration between the inversion pulse and
        the start of the bolus cutoff pulses (QUIPPSS, Q2-TIPS etc),
        :math:`\text{TI}_1`
    :type bolus_duration: float
    :param label_efficiency: labelling efficiency, :math:`\alpha`
    :type label_efficiency: float
    :param t1_arterial_blood: longitudinal relaxation time of arterial
        blood in seconds, :math:`T_{1,b}`
    :type t1_arterial_blood: float
    :return: the perfusion rate in ml/100g/min, :math:`f`
    :rtype: np.ndarray
    """
    control = np.asarray(control)
    label = np.asarray(label)
    m0 = np.asarray(m0)
    return np.divide(
        6000
        * lambda_blood_brain
        * (control - label)
        * np.exp(inversion_time / t1_arterial_blood),
        2 * label_efficiency * bolus_duration * m0,
        out=np.zeros_like(m0),
        where=np.abs(m0) >= M0_TOL,
    )


def asl_quant_lsq_gkm(
    control: np.ndarray,
    label: np.ndarray,
    m0_tissue: np.ndarray,
    lambda_blood_brain: np.ndarray,
    label_duration: float,
    post_label_delay: Union[np.ndarray, list[float]],
    label_efficiency: float,
    t1_arterial_blood: float,
    t1_tissue: np.ndarray,
    label_type: str,
) -> dict:
    """Calculates the perfusion and transit time by least-squares
    fitting to the ASL General Kinetic Model :cite:p:`Buxton1998`.

    Fitting is performed using :class:`scipy.optimize.curve_fit`.

    See :class:`.GkmFilter` and :class:`.GkmFilter.calculate_delta_m_gkm` for
    implementation details of the GKM function.

    :param control: control signal, must be 4D with signal for each
        post labelling delay on the 4th axis. Must have same dimensions as ``label``.
    :type control: np.ndarray
    :param label: label signal, must be 4D with signal for each post
        labelling delay on the 4th axis. Must have same dimensions as ``control``.
    :type label: np.ndarray
    :param m0_tissue: equilibrium magnetisation of the tissue.
    :type m0_tissue: np.ndarray
    :param lambda_blood_brain: tissue partition coefficient in g/ml.
    :type lambda_blood_brain: np.ndarray
    :param label_duration: duration of the labelling pulse in seconds.
    :type label_duration: float
    :param post_label_delay: array of post label delays, must be equal in
        length to the number of 3D volumes in ``control`` and ``label``.
    :type post_label_delay: np.ndarray
    :param label_efficiency: The degree of inversion of the labelling pulse.
    :type label_efficiency: float
    :param t1_arterial_blood: Longitudinal relaxation time of the arterial
        blood in seconds.
    :type t1_arterial_blood: float
    :param t1_tissue: Longitudinal relaxation time of the tissue in seconds.
    :type t1_tissue: np.ndarray
    :param label_type: The type of labelling: pulsed ('pasl') or continuous
        ('casl' or 'pcasl').
    :type label_type: str
    :return: A dictionary containing the following np.ndarrays:

        :'perfusion_rate': The estimated perfusion rate in ml/100g/min.
        :'transit_time': The estimated transit time in seconds.
        :'std_error': The standard error of the estimate of the fit.
        :'perfusion_rate_err': One standard deviation error in the fitted
            perfusion rate.
        :'transit_time_err': One standard deviation error in the fitted
            transit time.

    :rtype: dict

    ``control``, ``label``, ``m0_tissue``, ``t1_tissue`` and
    ``lambda_blood_brain`` must all have
    the same dimensions for the first 3 dimensions.


    """
    np.broadcast(m0_tissue, t1_tissue, lambda_blood_brain)
    np.broadcast(control, label)

    post_label_delay = np.asarray(post_label_delay)

    # subtract to get delta_m
    delta_m = control - label
    I, J, K = delta_m.shape[:3]
    perfusion_rate = np.zeros((I, J, K))
    transit_time = np.zeros((I, J, K))
    std_error = np.zeros((I, J, K))
    perfusion_rate_err = np.zeros((I, J, K))
    transit_time_err = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                # create an anonymous version of the function to solve
                func = lambda plds, perf, att: np.array(
                    [
                        calculate_delta_m_gkm(
                            perf,
                            att,
                            (
                                m0_tissue[i, j, k]
                                if isinstance(m0_tissue, np.ndarray)
                                else m0_tissue
                            ),
                            label_duration,
                            label_duration + pld,
                            label_efficiency,
                            (
                                lambda_blood_brain[i, j, k]
                                if isinstance(lambda_blood_brain, np.ndarray)
                                else lambda_blood_brain
                            ),
                            t1_arterial_blood,
                            (
                                t1_tissue[i, j, k]
                                if isinstance(t1_tissue, np.ndarray)
                                else t1_tissue
                            ),
                            label_type,
                        )
                        for pld in plds
                    ]
                )
                # fit for the perfusion rate and transit time
                obs = delta_m[i, j, k, :]
                popt, pcov = curve_fit(func, post_label_delay, obs)
                perfusion_rate[i, j, k] = popt[0]
                transit_time[i, j, k] = popt[1]
                std_error[i, j, k] = np.sqrt(
                    np.sum((obs - func(post_label_delay, *popt)) ** 2)
                    / post_label_delay.size
                )
                # compute one standard deviation errors of the parameters
                perr = np.sqrt(np.diag(pcov))
                perfusion_rate_err = perr[0]
                transit_time_err = perr[1]

    return {
        "perfusion_rate": perfusion_rate,
        "transit_time": transit_time,
        "perfusion_rate_err": perfusion_rate_err,
        "transit_time_err": transit_time_err,
        "std_error": std_error,
    }
