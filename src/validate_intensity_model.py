import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from rate_intensity_model import (
    fit_model, predict_rate_constant,
    INTENSITY_MW_CM2, RATE_CONSTANT_HR1, RATE_CONSTANT_ERR_HR1, K_DARK_HR1,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

a, n, r2 = fit_model()
print(f"Fitted model: k(I) = {K_DARK_HR1} + {a:.4f} * I^{n:.3f}")
print(f"R^2 = {r2:.4f}")
print()

pred_at_data = predict_rate_constant(INTENSITY_MW_CM2, a, n)
print("Intensity (mW/cm2) | Real k (hr-1) | Predicted k (hr-1)")
for i, real, pred in zip(INTENSITY_MW_CM2, RATE_CONSTANT_HR1, pred_at_data):
    print(f"   {i:>5.1f}           |   {real:.3f}       |   {pred:.3f}")

I_smooth = np.linspace(0, 8, 200)
k_smooth = predict_rate_constant(I_smooth, a, n)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(INTENSITY_MW_CM2, RATE_CONSTANT_HR1, yerr=RATE_CONSTANT_ERR_HR1,
            fmt="o", color="black", label="Real data (Table S2, 2023 paper)")
ax.plot(I_smooth, k_smooth, "-", color="crimson",
        label=f"Fit: k = {K_DARK_HR1} + {a:.3f}*I^{n:.2f} (R2={r2:.3f})")
ax.set_xlabel("Incident light intensity (mW/cm2)")
ax.set_ylabel("Apparent rate constant, k (hr-1)")
ax.set_title("MB conversion rate vs. red LED intensity, 1300nm Cu2O particles")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "rate_intensity_fit.png"), dpi=150)

csv_path = os.path.join(RESULTS_DIR, "rate_intensity_fit.csv")
rows = np.column_stack([I_smooth, k_smooth])
np.savetxt(csv_path, rows, delimiter=",", header="intensity_mW_cm2,predicted_k_hr1", comments="")

print()
print(f"Saved plot to {os.path.join(RESULTS_DIR, 'rate_intensity_fit.png')}")