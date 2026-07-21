import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut

from resonance_dataset import DIAMETER_NM, RESONANCE_WAVELENGTH_NM, PEAK_SIGMA_SCA_M2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X = DIAMETER_NM.reshape(-1, 1)
y_wl = RESONANCE_WAVELENGTH_NM
y_mag = np.log(PEAK_SIGMA_SCA_M2 * 1e13)  # log-space: values span ~17x, grows steeply with size

kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=100, length_scale_bounds=(10, 1000)) \
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))


def loo_predictions(X, y):
    """Leave-one-out predictions -- the honest way to check a surrogate fit on 8 points."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
        gp.fit(X[train_idx], y[train_idx])
        preds[test_idx] = gp.predict(X[test_idx])
    return preds


# Leave-one-out validation (real held-out error, not training fit)
loo_wl = loo_predictions(X, y_wl)
loo_mag = loo_predictions(X, y_mag)

rmse_wl = np.sqrt(np.mean((loo_wl - y_wl) ** 2))
rmse_mag_pct = np.sqrt(np.mean((np.exp(loo_mag) - np.exp(y_mag)) ** 2 / np.exp(y_mag) ** 2)) * 100

print("Leave-one-out cross-validation (held-out predictions, not training fit):")
print()
print("Diameter | Real wavelength | LOO predicted | Real peak sigma_sca | LOO predicted")
for i in range(len(X)):
    print(f"  {DIAMETER_NM[i]:>3.0f} nm | {y_wl[i]:>6.0f} nm      | {loo_wl[i]:>6.0f} nm     | "
          f"{np.exp(y_mag[i]):>6.2f}e-13 m2   | {np.exp(loo_mag[i]):>6.2f}e-13 m2")
print()
print(f"LOO RMSE (resonance wavelength): {rmse_wl:.1f} nm")
print(f"LOO RMSE (peak scattering, relative): {rmse_mag_pct:.1f}%")
print()

# Fit final models on all data for smooth prediction curves
gp_wl = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
gp_wl.fit(X, y_wl)
gp_mag = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
gp_mag.fit(X, y_mag)

d_smooth = np.linspace(150, 400, 200).reshape(-1, 1)
wl_pred, wl_std = gp_wl.predict(d_smooth, return_std=True)
mag_pred, mag_std = gp_mag.predict(d_smooth, return_std=True)
sigma_pred = np.exp(mag_pred)
sigma_hi = np.exp(mag_pred + mag_std)
sigma_lo = np.exp(mag_pred - mag_std)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.fill_between(d_smooth.ravel(), wl_pred - wl_std, wl_pred + wl_std, alpha=0.2, color="steelblue")
ax1.plot(d_smooth, wl_pred, "-", color="steelblue", label="GP surrogate")
ax1.scatter(DIAMETER_NM, RESONANCE_WAVELENGTH_NM, color="black", zorder=5, label="Mie engine (real sim data)")
ax1.set_xlabel("Diameter (nm)")
ax1.set_ylabel("Resonance wavelength (nm)")
ax1.set_title("Size -> resonance wavelength")
ax1.legend(fontsize=8)

ax2.fill_between(d_smooth.ravel(), sigma_lo, sigma_hi, alpha=0.2, color="darkorange")
ax2.plot(d_smooth, sigma_pred, "-", color="darkorange", label="GP surrogate")
ax2.scatter(DIAMETER_NM, PEAK_SIGMA_SCA_M2 * 1e13, color="black", zorder=5, label="Mie engine (real sim data)")
ax2.set_xlabel("Diameter (nm)")
ax2.set_ylabel(r"Peak scattering ($\times10^{13}$ m$^2$)")
ax2.set_title("Size -> peak scattering strength")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "resonance_surrogate.png"), dpi=150)
print(f"Saved plot to {os.path.join(RESULTS_DIR, 'resonance_surrogate.png')}")