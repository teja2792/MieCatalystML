"""
rate_intensity_model.py

Fits the real light-intensity vs. photocatalytic rate constant data
from Table S2 of the Supporting Information of Addanki Tirumala et al.,
"Tuning Catalytic Activity and Selectivity in Photocatalysis on
Mie-Resonant Cuprous Oxide Particles," ACS Sustainable Chem. Eng. 2023,
11, 15931-15940. Fixed particle size (1300+/-200 nm), red LED, varying
incident light intensity.

Model form: k(I) = k_dark + a * I^n, with k_dark fixed to the real
measured dark-condition rate constant (not a free parameter) rather
than a pure power law k = a*I^n, which cannot represent the nonzero
dark-condition baseline rate. a and n are fit by grid search over n
with closed-form linear least squares for a at each n (pure numpy, no
external optimizer dependency).
"""

import numpy as np

# Real data, Table S2 (2023 paper's SI)
INTENSITY_MW_CM2 = np.array([0.0, 1.7, 4.9, 6.2, 7.4])
RATE_CONSTANT_HR1 = np.array([0.09, 0.25, 1.19, 1.95, 2.97])
RATE_CONSTANT_ERR_HR1 = np.array([0.01, 0.05, 0.28, 0.26, 0.33])
K_DARK_HR1 = 0.09  # measured dark-condition rate constant, used as a fixed baseline


def fit_model(n_grid_min=0.5, n_grid_max=3.0, n_grid_points=251):
    """
    Fits k(I) = K_DARK_HR1 + a * I^n to the real data above. Returns
    (a, n, r_squared).
    """
    resid = RATE_CONSTANT_HR1 - K_DARK_HR1
    mask = INTENSITY_MW_CM2 > 0

    best_r2, best_n, best_a = -np.inf, None, None
    for n_try in np.linspace(n_grid_min, n_grid_max, n_grid_points):
        a_try = (np.sum(resid[mask] * INTENSITY_MW_CM2[mask] ** n_try) /
                 np.sum(INTENSITY_MW_CM2[mask] ** (2 * n_try)))
        pred = K_DARK_HR1 + a_try * INTENSITY_MW_CM2 ** n_try
        pred[0] = K_DARK_HR1
        ss_res = np.sum((RATE_CONSTANT_HR1 - pred) ** 2)
        ss_tot = np.sum((RATE_CONSTANT_HR1 - np.mean(RATE_CONSTANT_HR1)) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2, best_n, best_a = r2, n_try, a_try

    return best_a, best_n, best_r2


def predict_rate_constant(intensity_mw_cm2, a, n):
    """Predicts the rate constant (hr^-1) at a given light intensity (mW/cm^2)."""
    intensity_mw_cm2 = np.atleast_1d(intensity_mw_cm2).astype(float)
    return K_DARK_HR1 + a * intensity_mw_cm2 ** n