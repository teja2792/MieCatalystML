"""
mie_engine.py

Wraps miepython 3.2.0 (Prahl, https://github.com/scottprahl/miepython) to
compute Mie scattering/absorption/extinction efficiencies, and per-multipole
(ED/MD/EQ/MQ) contributions, for a Cu2O sphere using the real optical
constants in optical_constants.py.

Refractive index convention: m = n - 1j*k (negative imaginary part for
absorbing media) -- miepython's own convention, verified against its source.

Two miepython 3.2.0 API issues discovered and worked around here, both
confirmed empirically against the lossless (k=0) n=2.45, R=330nm sphere
benchmark from Figure S7/S8 of the source paper's SI:

1. `efficiencies_mx(..., e_field=True/False)` is unimplemented for scalar
   inputs (miepython's own source: "currently unused in scalar aggregate
   efficiencies"). It returns the same value regardless of e_field, so it
   CANNOT be used to separate electric vs magnetic multipole contributions.
   Do not use it for that purpose.
2. `coefficients(m, x, n_pole=n)` returns an array of length n_pole+1; the
   coefficient for multipole order n is the LAST element, not the first.

Per-multipole scattering efficiency is instead computed directly from the
Mie coefficients using the standard formula:
    Qsca_n(electric) = 2*(2n+1)*|a_n|^2 / x^2
    Qsca_n(magnetic) = 2*(2n+1)*|b_n|^2 / x^2
which was validated to sum to the library's own total Qsca (efficiencies_mx
with n_pole=0) to within numerical precision -- see
scripts/validate_mie_engine.py.
"""

import numpy as np
import miepython
from optical_constants import cu2o_refractive_index

N_MEDIUM_AIR = 1.0
N_MEDIUM_WATER = 1.33  # used in the source paper's medium-effect simulations


def _lookup_nk(wavelength_nm, n, k):
    if n is None or k is None:
        n_arr, k_arr = cu2o_refractive_index(wavelength_nm)
        n, k = float(np.atleast_1d(n_arr)[0]), float(np.atleast_1d(k_arr)[0])
    return n, k


def mie_efficiencies(radius_nm, wavelength_nm, n_medium=N_MEDIUM_AIR, n=None, k=None):
    """
    Returns (Qext, Qsca, Qabs) for a sphere of given radius at a given
    wavelength (both in nm). If n/k are not provided, looks them up from
    the real Cu2O table via optical_constants.cu2o_refractive_index.
    """
    n, k = _lookup_nk(wavelength_nm, n, k)
    m = n - 1j * k
    diameter_nm = 2 * radius_nm

    qext, qsca, qback, g = miepython.efficiencies(m, diameter_nm, wavelength_nm, n_env=n_medium)
    qabs = qext - qsca
    return float(qext), float(qsca), float(qabs)


def mie_spectrum(radius_nm, wavelengths_nm, n_medium=N_MEDIUM_AIR, n=None, k=None):
    """Same as mie_efficiencies, but vectorized over an array of wavelengths."""
    qext_list, qsca_list, qabs_list = [], [], []
    for wl in wavelengths_nm:
        qext, qsca, qabs = mie_efficiencies(radius_nm, wl, n_medium=n_medium, n=n, k=k)
        qext_list.append(qext)
        qsca_list.append(qsca)
        qabs_list.append(qabs)
    return np.array(qext_list), np.array(qsca_list), np.array(qabs_list)


def _multipole_qsca(m, x, n_pole):
    """Qsca contribution of a single multipole order, split into electric (a_n) and magnetic (b_n)."""
    a, b = miepython.coefficients(m, x, n_pole=n_pole)
    a = np.atleast_1d(a)[-1]
    b = np.atleast_1d(b)[-1]
    cn = 2 * n_pole + 1
    qsca_electric = 2.0 * cn * abs(a) ** 2 / x ** 2
    qsca_magnetic = 2.0 * cn * abs(b) ** 2 / x ** 2
    return qsca_electric, qsca_magnetic


def multipole_contributions(radius_nm, wavelength_nm, n_medium=N_MEDIUM_AIR, n=None, k=None):
    """
    Decomposes scattering efficiency into ED, MD (dipole, n=1) and EQ, MQ
    (quadrupole, n=2) contributions, matching Figure S7/S8 of the source
    paper's SI. Returns {'ED':..., 'MD':..., 'EQ':..., 'MQ':...}.
    """
    n, k = _lookup_nk(wavelength_nm, n, k)
    m = (n - 1j * k) / n_medium
    diameter_nm = 2 * radius_nm
    x = np.pi * diameter_nm / (wavelength_nm / n_medium)

    ed, md = _multipole_qsca(m, x, 1)
    eq, mq = _multipole_qsca(m, x, 2)
    return {"ED": float(ed), "MD": float(md), "EQ": float(eq), "MQ": float(mq)}