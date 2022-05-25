"""Test for create_qasper_ground_truth.py"""

import numpy as np


def calc_cbf_gkm_casl(
    delta_m,
    m0,
    f,
    transit_time,
    t1_tissue,
    t1_blood,
    blood_brain_pc,
    lab_dur,
    pld,
    lab_eff,
) -> np.ndarray:
    r"""Calculates CBF using the full general kinetic model for a single PLD.

    :param delta_m: Difference in magnetisation between control and label
    :type delta_m: np.ndarray
    :param m0: Equilibrium magnetisation
    :type m0: np.ndarray
    :param f: perfusion rate in ml/100g/min - yes the quantity required is needed to calculate it!
      This is only used to calculate :math:`\frac{f}{\lambda}, so if omitted then this is
      equivalent to approximating that :math:`T_1' == T_{1,\text{tissue}}`
    :type f: np.ndarray
    :param transit_time: The transit time in seconds.
    :type transit_time: np.ndarray
    :param t1_tissue: The tissue T1 in seconds.
    :type t1_tissue: np.ndarray
    :param t1_blood: The blood T1 in seconds.
    :type t1_blood: float
    :param blood_brain_pc: The blood brain partition coefficient, :math:`\lambda` in g/ml.
    :type blood_brain_pc: np.ndarray
    :param lab_dur: label duration in seconds.
    :type lab_dur: float
    :param pld: The post labelling delay in seconds.
    :type pld: float
    :param lab_eff: The labelling efficiency
    :type lab_eff: float
    :return: The calculated CBF
    :rtype: np.ndarray
    """
    # pre-calculate anything where an array is involved in a division using np.divide
    one_over_t1 = np.divide(
        1, t1_tissue, out=np.zeros_like(t1_tissue), where=t1_tissue != 0
    )
    flow_over_lambda = np.divide(
        f / 6000,
        blood_brain_pc,
        out=np.zeros_like(blood_brain_pc),
        where=blood_brain_pc != 0,
    )
    denominator = one_over_t1 + flow_over_lambda

    t1_prime = np.divide(
        1, denominator, out=np.zeros_like(denominator), where=denominator != 0
    )

    q_ss = 1 - np.exp(
        -np.divide(lab_dur, t1_prime, out=np.zeros_like(t1_prime), where=t1_prime != 0)
    )

    exp_pld_tt_t1p = np.exp(
        np.divide(
            (pld - transit_time),
            t1_prime,
            out=np.zeros_like(t1_prime),
            where=t1_prime != 0,
        )
    )
    exp_tt_t1b = np.exp(transit_time / t1_blood)
    norm_delta_m = np.divide(delta_m, m0, out=np.zeros_like(m0), where=m0 != 0)
    denominator = 2 * lab_eff * q_ss * t1_prime
    cbf = 6000 * (
        np.divide(
            blood_brain_pc * norm_delta_m,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        * exp_tt_t1b
        * exp_pld_tt_t1p
    )

    return cbf
