import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from near_field import near_field_map

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RADIUS_NM = 87.5  # 175 nm diameter Cu2O sphere, matches SI Figure S5a/b
D_SPHERE_NM = 2 * RADIUS_NM

# Both H2 and E2 use the "xy" plane (miepython frame, z=0 -- the plane
# transverse to the propagation axis). This was settled empirically, not
# assumed: an earlier version used "yz" for H2, which produced a peak
# (37.49) far above the SI's reported ~25 and a dark interference "null"
# spot not present in Figure S5a. Switching H2 to "xy" gave 29.21 (much
# closer to ~25) and the null spot vanished -- consistent with "yz"
# containing the propagation axis and picking up standing-wave fringing
# that a purely transverse cut doesn't show. A parallel hypothesis that
# E2 should move to "xz" was tested and rejected: it made the match
# worse (12.96 vs the SI's ~8, compared to "xy"'s 10.94). See
# results/*_kplane_reference.png and results/*_kEplane_check.png for the
# rejected alternatives kept as a record of that investigation.
CASES = [
    ("H2_resonance_542nm",    542.0, "xy", "H"),
    ("H2_offresonance_650nm", 650.0, "xy", "H"),
    ("E2_resonance_542nm",    542.0, "xy", "E"),
    ("E2_offresonance_650nm", 650.0, "xy", "E"),
]

for name, wavelength, plane, field in CASES:
    U, V, E2, H2 = near_field_map(RADIUS_NM, wavelength, plane=plane, extent_factor=1.5, grid_points=161)
    Z = H2 if field == "H" else E2
    label = "|H/H0|^2" if field == "H" else "|E/E0|^2"

    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    header = f"{plane}_axis1_nm,{plane}_axis2_nm,{label}"
    rows = np.column_stack([U.ravel(), V.ravel(), Z.ravel()])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.pcolormesh(U, V, Z, shading="auto", cmap="inferno")
    circle = plt.Circle((0, 0), RADIUS_NM, fill=False, edgecolor="white", linewidth=1.2, linestyle="--")
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{label}, XY plane (miepython frame), {wavelength:.0f} nm\n"
                 f"Cu2O sphere, d={D_SPHERE_NM:.0f} nm")
    fig.colorbar(im, ax=ax, label=label)
    png_path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"Saved {name}: peak {label} = {np.max(Z):.2f}  -> {png_path}")

print()
print("Dashed white circle marks the sphere boundary.")
print("Reference values from SI Figure S5a/b: H2 peak ~25 (on-resonance), E2 peak ~8 (on-resonance).")