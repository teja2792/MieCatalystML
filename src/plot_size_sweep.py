import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from size_sweep import cross_sections
from optical_constants import WAVELENGTH_RANGE_NM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Matches SI Figure S3(a)/(c): Cu2O spheres, diameters 25-175 nm
DIAMETERS_NM = [25, 50, 75, 100, 125, 150, 175]

# Search the FULL real data range (Table S1, 300-2000 nm) for peaks, so a
# resonance that falls below the SI's plotted window isn't missed. The
# plot itself is still clipped to 400 nm on the low end to match Figure
# S3's displayed x-axis, for a fair visual comparison.
WAVELENGTHS_NM = np.linspace(WAVELENGTH_RANGE_NM[0], WAVELENGTH_RANGE_NM[1], 500)
PLOT_MIN_NM = 400  # matches SI Figure S3's displayed x-axis window

fig_sca, ax_sca = plt.subplots(figsize=(6, 4.5))
fig_abs, ax_abs = plt.subplots(figsize=(6, 4.5))

results = {}
for d in DIAMETERS_NM:
    sigma_ext, sigma_sca, sigma_abs = cross_sections(d, WAVELENGTHS_NM)
    results[d] = (sigma_sca, sigma_abs)
    ax_sca.plot(WAVELENGTHS_NM, sigma_sca * 1e13, label=f"{d} nm")
    ax_abs.plot(WAVELENGTHS_NM, sigma_abs * 1e13, label=f"{d} nm")

    local_max_idx = [i for i in range(1, len(sigma_sca) - 1)
                      if sigma_sca[i] > sigma_sca[i - 1] and sigma_sca[i] > sigma_sca[i + 1]]
    global_idx = int(np.argmax(sigma_sca))
    global_wl = WAVELENGTHS_NM[global_idx]
    in_window = "inside" if global_wl >= PLOT_MIN_NM else "BELOW the SI's 400nm plot window"
    print(f"d={d:>3} nm: global max at {global_wl:.0f} nm ({in_window}), "
          f"sigma_sca = {sigma_sca[global_idx]*1e13:.3f} x1e-13 m^2")
    if local_max_idx:
        bumps = ", ".join(f"{WAVELENGTHS_NM[i]:.0f}nm={sigma_sca[i]*1e13:.3f}" for i in local_max_idx)
        print(f"           local maxima (interior bumps, if any): {bumps}")
    else:
        print(f"           no interior local maxima -- monotonic across the full 300-2000nm range")

ax_sca.set_xlim(PLOT_MIN_NM, WAVELENGTH_RANGE_NM[1])
ax_sca.set_xlabel("Wavelength (nm)")
ax_sca.set_ylabel(r"Scattering x $10^{13}$ (m$^2$)")
ax_sca.set_title("Cu2O spheres, 25-175 nm diameter (cf. SI Figure S3a)")
ax_sca.legend(fontsize=8)
fig_sca.tight_layout()
fig_sca.savefig(os.path.join(RESULTS_DIR, "size_sweep_scattering.png"), dpi=150)

ax_abs.set_xlim(PLOT_MIN_NM, WAVELENGTH_RANGE_NM[1])
ax_abs.set_xlabel("Wavelength (nm)")
ax_abs.set_ylabel(r"Absorption x $10^{13}$ (m$^2$)")