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

# H2 uses "xy" (miepython frame, z=0) and E2 uses "xz" (miepython frame,
# y=0) -- these are DIFFERENT planes, deliberately, matching the SI's
# own convention of plotting H2 in its "YZ" plane and E2 in its "XY"
# plane (a different physical cut for each field). The correspondence
# was derived from the SI's stated FDTD axes (k||x, E||y, H||z) mapped
# role-by-role onto miepython's internal convention (k||z, E||x, H||y):
#   paper's YZ (spans E,H)   -> miepython's xy plane -> H2
#   paper's XY (spans k,E)   -> miepython's xz plane -> E2
#
# Known open issue: on this plane, E2 peaks at ~12.96 vs an SI colorbar
# reading of ~8 (an earlier version used "xy" for E2 too, which gave
# 10.94 -- numerically closer, but on the WRONG plane per the derivation
# above). Keeping "xz" here because the plane correspondence is exact
# algebra, while the ~8 is a value read off a printed colorbar by eye,
# which is a much softer number. The remaining gap could be colorbar
# misreading, or a real difference between analytic Mie near-field and
# full FDTD (mesh/PML effects) -- not yet resolved, flagged honestly
# rather than tuned away.
CASES = [
    ("H2_resonance_542nm",    542.0, "xy", "H"),
    ("H2_offresonance_650nm", 650.0, "xy", "H"),
    ("E2_resonance_542nm",    542.0, "xz", "E"),
    ("E2_offresonance_650nm", 650.0, "xz", "E"),
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