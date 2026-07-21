import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from size_sweep import cross_sections
from optical_constants import WAVELENGTH_RANGE_NM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Matches SI Figure S3(b)/(d): Cu2O spheres, diameters 200-400 nm
DIAMETERS_NM = [200, 250, 275, 300, 350, 400]

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

    for name, arr in (("scattering", sigma_sca), ("absorption", sigma_abs)):
        local_max_idx = [i for i in range(1, len(arr) - 1)
                          if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]]
        global_idx = int(np.argmax(arr))
        global_wl = WAVELENGTHS_NM[global_idx]
        in_window = "inside" if global_wl >= PLOT_MIN_NM else "BELOW 400nm window"
        print(f"d={d:>3} nm [{name}]: global max at {global_wl:.0f} nm ({in_window}), "
              f"sigma = {arr[global_idx]*1e13:.3f} x1e-13 m^2")
        if len(local_max_idx) > 1:
            bumps = ", ".join(f"{WAVELENGTHS_NM[i]:.0f}nm={arr[i]*1e13:.3f}" for i in local_max_idx)
            print(f"              additional local maxima (multi-lobe structure): {bumps}")

ax_sca.set_xlim(PLOT_MIN_NM, WAVELENGTH_RANGE_NM[1])
ax_sca.set_xlabel("Wavelength (nm)")
ax_sca.set_ylabel(r"Scattering x $10^{13}$ (m$^2$)")
ax_sca.set_title("Cu2O spheres, 200-400 nm diameter (cf. SI Figure S3b)")
ax_sca.legend(fontsize=8)
fig_sca.tight_layout()
fig_sca.savefig(os.path.join(RESULTS_DIR, "size_sweep_scattering_large.png"), dpi=150)

ax_abs.set_xlim(PLOT_MIN_NM, WAVELENGTH_RANGE_NM[1])
ax_abs.set_xlabel("Wavelength (nm)")
ax_abs.set_ylabel(r"Absorption x $10^{13}$ (m$^2$)")
ax_abs.set_title("Cu2O spheres, 200-400 nm diameter (cf. SI Figure S3d)")
ax_abs.legend(fontsize=8)
fig_abs.tight_layout()
fig_abs.savefig(os.path.join(RESULTS_DIR, "size_sweep_absorption_large.png"), dpi=150)

for d in DIAMETERS_NM:
    sigma_sca, sigma_abs = results[d]
    csv_path = os.path.join(RESULTS_DIR, f"size_sweep_large_d{d}nm.csv")
    rows = np.column_stack([WAVELENGTHS_NM, sigma_sca, sigma_abs])
    np.savetxt(csv_path, rows, delimiter=",", header="wavelength_nm,sigma_sca_m2,sigma_abs_m2", comments="")

print()
print("Compare size_sweep_scattering_large.png / size_sweep_absorption_large.png against")
print("SI Figure S3b / S3d. Reference (read from S3b): 400nm peak ~7.2e-13 m^2 near ~1050-1100nm,")
print("350nm ~5.3e-13 near ~950nm, 300nm ~3.7e-13 near ~800nm. Figure S3d (absorption) shows")
print("multiple lobes per curve at these larger sizes -- expect several local maxima per diameter here,")
print("unlike the single clean peaks seen for 150/175nm.")