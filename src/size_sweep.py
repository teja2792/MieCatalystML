"""
size_sweep.py

Reproduces the size-dependent scattering/absorption trend shown in
Figure S3 of the source paper's SI: cross sections vs wavelength for
Cu2O spheres across a range of diameters. The SI computed this with
FDTD; here it's computed with the Mie engine validated in Phase 1/2,
using the same real Cu2O n,k data (Table S1) throughout this repo.

Cross sections (m^2) are computed from the dimensionless Mie
efficiencies as sigma = Q * pi * r^2, matching the SI's y-axis units of
cross section x10^13 (m^2).
"""

import numpy as np
from mie_engine import mie_efficiencies


def cross_sections(diameter_nm, wavelengths_nm, n_medium=1.0):
    """
    Returns (sigma_ext, sigma_sca, sigma_abs) in m^2 for a sphere of the
    given diameter, evaluated at each wavelength in wavelengths_nm,
    using the real Cu2O n,k table (Table S1) via mie_engine.
    """
    radius_nm = diameter_nm / 2.0
    radius_m = radius_nm * 1e-9
    area_m2 = np.pi * radius_m ** 2

    sigma_ext, sigma_sca, sigma_abs = [], [], []
    for wl in wavelengths_nm:
        qext, qsca, qabs = mie_efficiencies(radius_nm, wl, n_medium=n_medium)
        sigma_ext.append(qext * area_m2)
        sigma_sca.append(qsca * area_m2)
        sigma_abs.append(qabs * area_m2)
    return np.array(sigma_ext), np.array(sigma_sca), np.array(sigma_abs)