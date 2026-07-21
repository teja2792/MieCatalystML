"""
catalytic_rate_model.py

Implements the relative-rate overlap model from two companion papers,
using the physical descriptor each paper actually specifies:

1. Direct photocatalysis (green LED), eq 5 of the SI of Addanki
   Tirumala et al., ACS Catalysis 2022, 12, 7975-7985: rate ratio uses
   volume-normalized ABSORPTION cross section (sigma_abs/V).

2. Dye-sensitized degradation (red LED), eq 1 of the SI of Addanki
   Tirumala et al., ACS Appl. Nano Mater. 2022, 5, 6699-6707: rate ratio
   uses volume-normalized EXTINCTION cross section (sigma_ext/V).

Both real experiments were run in DMF solvent, not air -- neither paper
states DMF's refractive index or corrects for it in their own
simulation-vs-experiment comparison (their FDTD simulations were run at
n_medium=1). This module uses n_medium=1.43 (a standard external
literature value for DMF, not sourced from either paper) as a physical
correction beyond what the original papers did.

Validated, not trained, against two independent real ratios:
  - green LED, 145/42 nm spheres: real ratio = 9.62 +/- 1.63
  - red LED, 145/37 nm spheres: real ratio = 2.56 +/- 0.38
"""

import numpy as np
from size_sweep import cross_sections
from led_spectra import green_led_spectrum, red_led_spectrum

N_MEDIUM_DMF = 1.43  # standard literature value for DMF, not stated in either source paper

REAL_RATIOS = {
    "green": {"d_resonant_nm": 145, "d_reference_nm": 42, "ratio": 9.62,
              "ratio_uncertainty": 1.63, "descriptor": "absorption"},
    "red":   {"d_resonant_nm": 145, "d_reference_nm": 37, "ratio": 2.56,
              "ratio_uncertainty": 0.38, "descriptor": "extinction"},
}


def volume_normalized_overlap(diameter_nm, led_spectrum_fn, wavelengths_nm,
                               descriptor="absorption", n_medium=1.0):
    """
    Computes integral (sigma/V) * I0(lambda) dlambda for a sphere of the
    given diameter, using either the absorption or extinction cross
    section depending on `descriptor`, at the given medium refractive
    index.
    """
    sigma_ext, sigma_sca, sigma_abs = cross_sections(diameter_nm, wavelengths_nm, n_medium=n_medium)
    sigma = sigma_abs if descriptor == "absorption" else sigma_ext

    radius_m = (diameter_nm / 2.0) * 1e-9
    volume_m3 = (4.0 / 3.0) * np.pi * radius_m ** 3
    intensity = led_spectrum_fn(wavelengths_nm)
    integrand = (sigma / volume_m3) * intensity
    return np.trapz(integrand, wavelengths_nm)


def relative_rate(diameter_nm, reference_diameter_nm, led_spectrum_fn, wavelengths_nm,
                   descriptor="absorption", n_medium=1.0):
    """
    Predicted photocatalytic rate of a sphere of diameter_nm relative to
    a reference sphere of reference_diameter_nm, under the given LED
    spectrum, using the specified physical descriptor and medium.
    """
    overlap_target = volume_normalized_overlap(diameter_nm, led_spectrum_fn, wavelengths_nm, descriptor, n_medium)
    overlap_reference = volume_normalized_overlap(reference_diameter_nm, led_spectrum_fn, wavelengths_nm, descriptor, n_medium)
    return overlap_target / overlap_reference