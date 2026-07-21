import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from near_field import near_field_map, peak_enhancement

RADIUS_NM = 87.5       # 175 nm diameter -- matches SI Figure S5a/b
WAVELENGTH_NM = 542.0  # real Mie resonance peak wavelength, from SI caption

# Sanity check: far from the sphere, the total field should approach the
# incident plane wave (|E/E0|^2 -> 1, |H/H0|^2 -> 1), since the scattered
# contribution decays with distance while the incident field doesn't.
_, _, E2_far, H2_far = near_field_map(RADIUS_NM, WAVELENGTH_NM, extent_factor=8.0, grid_points=41, plane="yz")
edge_E2 = np.mean(E2_far[0, :])
edge_H2 = np.mean(H2_far[0, :])
print(f"Far-field sanity check (should be close to 1.0): |E/E0|^2={edge_E2:.3f}  |H/H0|^2={edge_H2:.3f}")
print()

# Real validation target: 175nm-diameter Cu2O sphere at its real,
# paper-reported Mie resonance (542 nm).
maxE2, maxH2 = peak_enhancement(RADIUS_NM, WAVELENGTH_NM, plane="yz", extent_factor=1.5, grid_points=121)
_, maxH2_e = peak_enhancement(RADIUS_NM, WAVELENGTH_NM, plane="xy", extent_factor=1.5, grid_points=121)
print(f"At resonance (542 nm): peak |H/H0|^2 (YZ plane) = {maxH2:.2f}")
print(f"At resonance (542 nm): peak |E/E0|^2 (XY plane) = {maxE2:.2f}")

# Off-resonance comparison -- enhancement should be markedly lower.
maxE2_off, maxH2_off = peak_enhancement(RADIUS_NM, 650.0, plane="yz", extent_factor=1.5, grid_points=121)
print(f"Off-resonance (650 nm): peak |H/H0|^2 = {maxH2_off:.2f}")
print()
print("Expect: far-field check ~1.0; on-resonance peaks clearly higher than off-resonance.")
print("Compare spatial pattern (not exact magnitude) against SI Figure S5a (H^2/H0^2, YZ) and S5b (E^2/E0^2, XY).")