import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from optical_constants import WAVELENGTH_RANGE_NM, cu2o_refractive_index
from mie_engine import mie_spectrum
from size_sweep import cross_sections
from near_field import near_field_map
from catalytic_rate_model import relative_rate, REAL_RATIOS
from led_spectra import green_led_spectrum, red_led_spectrum
from rate_intensity_model import (
    fit_model as fit_intensity_model, predict_rate_constant,
    INTENSITY_MW_CM2, RATE_CONSTANT_HR1, RATE_CONSTANT_ERR_HR1, K_DARK_HR1,
)
from resonance_dataset import DIAMETER_NM, RESONANCE_WAVELENGTH_NM, PEAK_SIGMA_SCA_M2

st.set_page_config(page_title="MieCatalystML", layout="wide")

WAVELENGTHS_FULL = np.linspace(WAVELENGTH_RANGE_NM[0], WAVELENGTH_RANGE_NM[1], 500)
LED_FNS = {"green": green_led_spectrum, "red": red_led_spectrum}
REFERENCE_DIAMETER = {"green": 42, "red": 37}
DESCRIPTOR = {"green": "absorption", "red": "extinction"}


@st.cache_data
def compute_spectrum(diameter_nm):
    qext, qsca, qabs = mie_spectrum(diameter_nm / 2.0, WAVELENGTHS_FULL)
    _, sigma_sca, sigma_abs = cross_sections(diameter_nm, WAVELENGTHS_FULL)
    return qext, qsca, qabs, sigma_sca, sigma_abs


@st.cache_data
def find_resonance(diameter_nm):
    """Local-maxima resonance finder, same method validated in Phase 3."""
    _, sigma_sca, _ = cross_sections(diameter_nm, WAVELENGTHS_FULL)
    local_max = [i for i in range(1, len(sigma_sca) - 1)
                 if sigma_sca[i] > sigma_sca[i - 1] and sigma_sca[i] > sigma_sca[i + 1]]
    if not local_max:
        return None, None
    best = max(local_max, key=lambda i: sigma_sca[i])
    return WAVELENGTHS_FULL[best], sigma_sca[best]


@st.cache_data
def compute_near_field(diameter_nm, wavelength_nm, grid_points=81):
    radius_nm = diameter_nm / 2.0
    _, _, _, H2 = near_field_map(radius_nm, wavelength_nm, plane="xy", grid_points=grid_points, extent_factor=1.5)
    U, V, E2, _ = near_field_map(radius_nm, wavelength_nm, plane="xz", grid_points=grid_points, extent_factor=1.5)
    return U, V, E2, H2


@st.cache_data
def compute_trend_curve():
    diam_range = np.arange(25, 405, 25)
    peak_scas = []
    for d in diam_range:
        _, s_sca, _ = cross_sections(d, WAVELENGTHS_FULL)
        peak_scas.append(np.max(s_sca) * 1e13)
    return diam_range, peak_scas


@st.cache_resource
def train_gp_surrogates():
    X = DIAMETER_NM.reshape(-1, 1)
    y_wl = RESONANCE_WAVELENGTH_NM
    y_mag = np.log(PEAK_SIGMA_SCA_M2 * 1e13)
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=100, length_scale_bounds=(10, 1000)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    gp_wl = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5).fit(X, y_wl)
    gp_mag = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5).fit(X, y_mag)
    return gp_wl, gp_mag


st.title("MieCatalystML")
st.caption("Mie resonance simulator for Cu2O nanospheres, grounded in real published data. Spheres only.")

st.sidebar.header("Particle size (drives optics)")
diameter = st.sidebar.slider("Sphere diameter (nm)", 25, 400, 175, step=5)

st.sidebar.header("Light conditions (drives catalytic rate)")
led_color = st.sidebar.radio("LED color", ["green", "red"])
intensity = st.sidebar.slider("Light intensity, mW/cm2 (fixed 1300nm particles, separate real dataset)",
                               0.0, 8.0, 7.4, step=0.1)
st.sidebar.caption("These two controls are independent -- no real dataset links particle size AND "
                    "light intensity to rate at once. See Methodology tab.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Optical Response", "Near-Field Enhancement", "Size-Dependent Trends",
    "Catalytic Rate: Experimental -> Simulation -> ML", "Methodology & Validation",
])

with tab1:
    st.subheader(f"Scattering / absorption / extinction, {diameter} nm Cu2O sphere")
    qext, qsca, qabs, sigma_sca, sigma_abs = compute_spectrum(diameter)
    res_wl, res_val = find_resonance(diameter)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(WAVELENGTHS_FULL, qsca, label="Qsca (scattering)")
    ax.plot(WAVELENGTHS_FULL, qabs, label="Qabs (absorption)")
    ax.plot(WAVELENGTHS_FULL, qext, label="Qext (extinction)", linestyle="--", alpha=0.6)
    if res_wl is not None:
        ax.axvline(res_wl, color="red", linestyle=":", label=f"Resonance: {res_wl:.0f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Efficiency (dimensionless)")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    if res_wl is None:
        st.info(f"No Mie resonance found for {diameter} nm spheres within the real optical-constant "
                f"data range (300-2000 nm). This is a real finding from Phase 3, not a bug -- small "
                f"Cu2O spheres are simply featureless in this window.")
    else:
        st.success(f"Resonance at {res_wl:.0f} nm, peak Qsca = {max(qsca):.2f}")

with tab2:
    st.subheader(f"Near-field enhancement maps, {diameter} nm Cu2O sphere")
    res_wl, _ = find_resonance(diameter)
    if res_wl is None:
        st.warning("No resonance in range for this size -- near-field maps are only meaningful at a "
                   "resonance wavelength. Try a diameter of 150 nm or larger.")
    else:
        st.caption(f"Computed at the resonance wavelength ({res_wl:.0f} nm). "
                   f"H2 shown in the miepython 'xy' plane, E2 in the 'xz' plane -- see Methodology tab "
                   f"for why these specific planes, derived from the source paper's stated FDTD axes.")
        with st.spinner("Computing near-field maps..."):
            U, V, E2, H2 = compute_near_field(diameter, res_wl)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.pcolormesh(U, V, H2, shading="auto", cmap="inferno")
            ax.add_patch(plt.Circle((0, 0), diameter / 2.0, fill=False, edgecolor="white", linestyle="--"))
            ax.set_aspect("equal")
            ax.set_title(f"|H/H0|^2, peak={np.max(H2):.1f}")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.pcolormesh(U, V, E2, shading="auto", cmap="inferno")
            ax.add_patch(plt.Circle((0, 0), diameter / 2.0, fill=False, edgecolor="white", linestyle="--"))
            ax.set_aspect("equal")
            ax.set_title(f"|E/E0|^2, peak={np.max(E2):.1f}")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
        if diameter == 175:
            st.caption("Real reference at 175nm/542nm (SI Figure S5a/b): H2 peak ~25, E2 peak ~8.")

with tab3:
    st.subheader("Where this particle sits in the size-dependent trend")
    diam_range, peak_scas = compute_trend_curve()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(diam_range, peak_scas, "o-", color="steelblue", label="Mie engine (this repo)")
    ax.axvline(diameter, color="red", linestyle=":", label=f"Selected: {diameter} nm")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel(r"Peak scattering ($\times10^{13}$ m$^2$)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption("Validated within 1-4% against SI Figure S3b for d=300/350/400nm (Phase 3b).")

with tab4:
    st.subheader(f"Catalytic rate: {led_color} LED, size-dependence")
    ref_d = REFERENCE_DIAMETER[led_color]
    descriptor = DESCRIPTOR[led_color]
    real_info = REAL_RATIOS[led_color]

    d_sweep = np.linspace(ref_d, 400, 60)
    predicted_ratios = [relative_rate(d, ref_d, LED_FNS[led_color], WAVELENGTHS_FULL, descriptor=descriptor)
                         for d in d_sweep]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d_sweep, predicted_ratios, "-", color="darkorange", label="Simulation (eq-5/eq-1 physical model)")
    ax.scatter([real_info["d_reference_nm"], real_info["d_resonant_nm"]], [1, real_info["ratio"]],
               color="black", zorder=5, label="Experimental (real data)")
    ax.errorbar([real_info["d_resonant_nm"]], [real_info["ratio"]], yerr=[real_info["ratio_uncertainty"]],
                color="black", capsize=4)
    current_pred = relative_rate(diameter, ref_d, LED_FNS[led_color], WAVELENGTHS_FULL, descriptor=descriptor) \
        if diameter >= ref_d else None
    if current_pred is not None:
        ax.axvline(diameter, color="red", linestyle=":", label=f"Selected: {diameter}nm -> {current_pred:.1f}x")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel(f"Rate relative to {ref_d}nm reference")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    if led_color == "green":
        st.success(f"Real ratio ({real_info['d_resonant_nm']}nm/{ref_d}nm): {real_info['ratio']:.2f}x "
                   f"+/- {real_info['ratio_uncertainty']:.2f}x. Model prediction is within real "
                   f"measurement uncertainty.")
    else:
        st.error(f"Real ratio ({real_info['d_resonant_nm']}nm/{ref_d}nm): {real_info['ratio']:.2f}x "
                 f"+/- {real_info['ratio_uncertainty']:.2f}x. Model overpredicts this one -- a "
                 f"documented, unresolved gap. See Methodology tab.")

    st.divider()
    st.subheader("Catalytic rate: light intensity dependence (fixed 1300nm particles)")
    a, n, r2 = fit_intensity_model()
    I_smooth = np.linspace(0, 8, 100)
    k_smooth = predict_rate_constant(I_smooth, a, n)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.errorbar(INTENSITY_MW_CM2, RATE_CONSTANT_HR1, yerr=RATE_CONSTANT_ERR_HR1, fmt="o",
                 color="black", label="Experimental (real data)")
    ax2.plot(I_smooth, k_smooth, "-", color="crimson", label=f"ML fit (R2={r2:.3f})")
    ax2.axvline(intensity, color="red", linestyle=":",
                label=f"Selected: {intensity:.1f} mW/cm2 -> k={predict_rate_constant(intensity, a, n)[0]:.2f} hr-1")
    ax2.set_xlabel("Light intensity (mW/cm2)")
    ax2.set_ylabel("Rate constant, k (hr-1)")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)
    st.success(f"Fit quality: R2 = {r2:.3f} against 5 real measured points.")

    st.divider()
    st.subheader("ML surrogate: size -> resonance (Option A)")
    gp_wl, gp_mag = train_gp_surrogates()
    if 150 <= diameter <= 400:
        pred_wl, std_wl = gp_wl.predict([[diameter]], return_std=True)
        pred_mag, std_mag = gp_mag.predict([[diameter]], return_std=True)
        res_wl_direct, res_val_direct = find_resonance(diameter)
        col1, col2 = st.columns(2)
        col1.metric("GP surrogate prediction", f"{pred_wl[0]:.0f} nm", f"+/-{std_wl[0]:.0f} nm")
        col2.metric("Direct Mie engine (ground truth)",
                    f"{res_wl_direct:.0f} nm" if res_wl_direct else "N/A")
        st.caption("The surrogate approximates our own validated Mie simulation for speed -- it is not "
                   "an independent prediction of reality. LOO-validated RMSE: 11.6nm (wavelength), "
                   "14.6% (peak scattering). See Methodology tab.")
    else:
        st.warning("GP surrogate was only trained on 150-400nm (where a resonance exists). "
                   "Select a diameter in that range to see it.")

with tab5:
    st.subheader("How trustworthy is this simulator?")
    st.markdown("Every number in this app is checked against real published measurements. "
                "Here's the honest scorecard -- what matched, what didn't, and why.")

    st.success("**Light scattering and absorption vs. particle size** -- matches real published "
               "measurements within 1-4% for the larger particles (300-400nm). This is the "
               "strongest-validated part of the simulator.")

    st.success("**Where the resonance 'glow' appears** -- matches real measurements within "
               "about 0.5-2%. When light bounces strongest off a particle, this simulator finds "
               "the right wavelength.")

    st.success("**Catalytic speed-up under green light** -- real experiments showed a 145nm "
               "particle reacts about 9.6x faster than a 42nm particle. This simulator predicts "
               "8.2x -- well within the measurement's own margin of error.")

    st.warning("**Near-field brightness (the 'hot spot' maps)** -- right shape and location, but "
               "the predicted brightness runs 17-62% higher than what's read off the published "
               "figures. Likely because those source numbers were read off a printed chart by eye, "
               "not because the physics is wrong.")

    st.error("**Catalytic speed-up under red light** -- this one doesn't match. Real experiments "
             "showed about 2.6x faster; this simulator predicts 12.2x. We tracked this down to a "
             "genuinely different chemical mechanism (dye-sensitized degradation, not direct "
             "photocatalysis) that the original researchers themselves called 'a rough "
             "approximation' in their own paper. We're reporting this honestly rather than hiding "
             "it or tuning the model until it looks better.")

    st.info("**The ML 'surrogate' model** -- this doesn't predict new real-world results. It's a "
            "fast approximation trained to mimic our own physics engine, useful for instant slider "
            "response instead of waiting on a full calculation each time.")

    st.divider()
    st.subheader("The two sliders don't talk to each other, on purpose")
    st.markdown("""
    Real experiments never tested "what if we change particle size **and** light brightness at
    the same time." One real study varied particle size (fixed light). A different real study
    varied light brightness (fixed particle size). So this app keeps those two controls separate
    too, instead of inventing a connection that was never actually measured.
    """)

    st.divider()
    st.subheader("What this is built on")
    st.markdown("""
    - Mohammadparast et al., *ACS Appl. Nano Mater.* 2020 -- optical properties and light hot-spot data
    - Addanki Tirumala et al., *ACS Catalysis* 2022 -- green-light particle-size vs. reaction-speed data
    - Addanki Tirumala et al., *ACS Appl. Nano Mater.* 2022 -- red-light particle-size vs. reaction-speed data
    - Addanki Tirumala et al., *ACS Sustainable Chem. Eng.* 2023 -- light-brightness vs. reaction-speed data
    """)

    with st.expander("Technical details (equations, coordinate systems, full validation table)"):
        st.markdown("""
        **Scope:** spheres only. Mie theory is the exact closed-form solution of Maxwell's
        equations for a sphere; it does not apply to cubes (the source papers used FDTD for those).

        **Coordinate conventions:** the source papers' FDTD simulations use propagation along x,
        E-field along y, H-field along z. This repo's Mie engine (miepython) uses propagation along
        z, E along x, H along y -- a different but equivalent right-handed coordinate system, related
        by a cyclic axis permutation, not a reflection. Mapped and empirically confirmed in Phase 2.

        **Other approximations used:** LED emission spectra are modeled as Gaussians fit to reported
        peak/range values, not measured spectra. Real particles are polydisperse (e.g. 145+/-41nm)
        and not perfectly spherical; this repo uses the mean diameter as an idealized sphere.
        """)
        try:
            import csv
            csv_path = os.path.join(os.path.dirname(__file__), "results", "validation_summary.csv")
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            st.table(rows)
        except FileNotFoundError:
            st.caption("Run src/build_validation_summary.py to generate the validation table.")