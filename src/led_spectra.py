"""
led_spectra.py

Approximate emission spectra for the green and red LEDs used in the
source papers' photocatalysis experiments, modeled as Gaussians fit to
the real reported peak/range values -- NOT digitized measured spectra.
This approximation is explicit and labeled, not presented as measured
data.

Green LED: peak ~519 nm, range ~485-555 nm (Addanki Tirumala et al.,
ACS Appl. Nano Mater. 2022, 5, 6699-6707).
Red LED: range ~615-660 nm (Addanki Tirumala et al., ACS Sustainable
Chem. Eng. 2023, 11, 15931-15940). No single peak value stated in the
text; the midpoint of the range is used as the approximate peak.
"""

import numpy as np

GREEN_PEAK_NM = 519.0
GREEN_FWHM_NM = 555.0 - 485.0  # 70 nm, from the paper's stated range

RED_PEAK_NM = (615.0 + 660.0) / 2.0  # 637.5 nm, midpoint of stated range
RED_FWHM_NM = 660.0 - 615.0  # 45 nm, from the paper's stated range


def _gaussian(wavelength_nm, peak_nm, fwhm_nm):
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((wavelength_nm - peak_nm) / sigma) ** 2)


def green_led_spectrum(wavelength_nm):
    """Approximate green LED relative intensity at the given wavelength(s)."""
    return _gaussian(wavelength_nm, GREEN_PEAK_NM, GREEN_FWHM_NM)


def red_led_spectrum(wavelength_nm):
    """Approximate red LED relative intensity at the given wavelength(s)."""
    return _gaussian(wavelength_nm, RED_PEAK_NM, RED_FWHM_NM)