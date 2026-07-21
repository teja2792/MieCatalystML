import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mie_engine import mie_spectrum, multipole_contributions

# Reproduces the SI's Figure S7/S8 benchmark: lossless dielectric sphere,
# constant n=2.45, k=0, R=330nm (d=660nm). Paper reports:
#   - MD and ED contribute equally near 1548 nm
#   - lowest-energy resonance at 1714 nm, MD slightly dominant over ED
RADIUS_NM = 330
N_CONST, K_CONST = 2.45, 0.0

wavelengths = np.linspace(800, 2500, 1701)
qext, qsca, qabs = mie_spectrum(RADIUS_NM, wavelengths, n=N_CONST, k=K_CONST)

peak_idx = [i for i in range(1, len(wavelengths) - 1)
            if qsca[i] > qsca[i - 1] and qsca[i] > qsca[i + 1]]
print("Total Qsca local maxima:")
for i in peak_idx:
    print(f"  {wavelengths[i]:.0f} nm -> Qsca = {qsca[i]:.3f}")

print()
print("Multipole decomposition near reported feature wavelengths:")
for target in (1548, 1714):
    parts = multipole_contributions(RADIUS_NM, float(target), n=N_CONST, k=K_CONST)
    dominant = "MD" if parts["MD"] > parts["ED"] else "ED"
    print(f"  {target} nm: ED={parts['ED']:.3f} MD={parts['MD']:.3f} "
          f"EQ={parts['EQ']:.3f} MQ={parts['MQ']:.3f}  (dominant: {dominant})")

print()
print("Expected from SI: MD ~= ED near 1548 nm; MD slightly > ED near 1714 nm "
      "(lowest-energy resonance, MD+ED dominated)")