"""
resonance_dataset.py

Training data for the size -> resonance ML surrogate. These are NOT
independently measured values -- they are the resonance
wavelength/peak scattering cross section extracted from this repo's
own already-validated Mie engine (Phases 3/3b, cross-checked against
SI Figure S3b within 1-4% for d=300/350/400nm). The surrogate is
learning to approximate our own physics simulation faster, not
learning something new about the real world -- that distinction is
called out explicitly in the README.

Only diameters 150-400 nm are included: smaller spheres (25-125 nm)
showed no interior resonance within the real optical-constant data
range (300-2000 nm, Table S1) during the Phase 3 sweep, so there is no
resonance wavelength to fit a model to for those sizes.
"""

import numpy as np

# diameter_nm, resonance_wavelength_nm, peak_sigma_sca_m2
_DATA = [
    (150, 498, 0.420e-13),
    (175, 556, 0.883e-13),
    (200, 617, 1.382e-13),
    (250, 726, 2.473e-13),
    (275, 777, 3.086e-13),
    (300, 831, 3.745e-13),
    (350, 951, 5.254e-13),
    (400, 1080, 7.087e-13),
]

DIAMETER_NM = np.array([row[0] for row in _DATA], dtype=float)
RESONANCE_WAVELENGTH_NM = np.array([row[1] for row in _DATA], dtype=float)
PEAK_SIGMA_SCA_M2 = np.array([row[2] for row in _DATA], dtype=float)