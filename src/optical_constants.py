"""
optical_constants.py

Real Cu2O complex refractive index (n, k) vs. wavelength, digitized from
Table S1 of the Supporting Information of Mohammadparast, Ramakrishnan,
Khatri, Tirumala, Tan, Kalkan, Andiappan, "Cuprous Oxide Cubic Particles
with Strong and Tunable Mie Resonances for Use as Nanoantennas," ACS
Appl. Nano Mater. 2020, 3, 6806-6815 (DOI: 10.1021/acsanm.0c01201).

Originally sourced in that paper from: Palik, E. D. Handbook of Optical
Constants of Solids; Academic Press, 1998. This is the exact dataset used
in that paper's own FDTD simulations -- not an independent approximation.
"""

import numpy as np

_WAVELENGTH_NM = np.array([
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
    850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 2000
])
_N = np.array([
    2.00, 2.40, 2.80, 3.06, 3.12, 3.10, 3.02, 2.90, 2.83, 2.77, 2.70,
    2.66, 2.63, 2.61, 2.60, 2.59, 2.58, 2.57, 2.57, 2.57, 2.56
])
_K = np.array([
    1.850, 1.440, 0.990, 0.600, 0.350, 0.190, 0.130, 0.100, 0.083, 0.070, 0.060,
    0.053, 0.048, 0.043, 0.040, 0.033, 0.027, 0.021, 0.017, 0.013, 0.002
])

WAVELENGTH_RANGE_NM = (float(_WAVELENGTH_NM.min()), float(_WAVELENGTH_NM.max()))


def cu2o_refractive_index(wavelength_nm):
    """
    Returns (n, k) at the given wavelength(s) in nm, via linear
    interpolation on the real Palik-sourced table above. Raises outside
    the tabulated range (300-2000 nm) rather than silently extrapolating.
    """
    wavelength_nm = np.atleast_1d(wavelength_nm).astype(float)
    if np.any(wavelength_nm < WAVELENGTH_RANGE_NM[0]) or np.any(wavelength_nm > WAVELENGTH_RANGE_NM[1]):
        raise ValueError(
            f"Wavelength out of the tabulated range {WAVELENGTH_RANGE_NM} nm "
            f"(Table S1 data does not extend beyond this)."
        )
    n = np.interp(wavelength_nm, _WAVELENGTH_NM, _N)
    k = np.interp(wavelength_nm, _WAVELENGTH_NM, _K)
    return n, k