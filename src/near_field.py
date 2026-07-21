"""
near_field.py

Near-field intensity enhancement maps (|E/E0|^2, |H/H0|^2) around a Cu2O
sphere, using miepython.field (Prahl, miepython 3.2.0).

Field convention, derived from miepython's source (field.py,
_incident_e_spherical / _incident_h_spherical): the incident plane wave
has E0 polarized along x-hat, H0 along y-hat, propagating along z-hat,
with unit amplitude by default. That means the enhancement factors here
are literally |E_computed|^2 and |H_computed|^2 -- no extra normalization
needed.

IMPORTANT -- axis convention mismatch discovered from the source paper's
SI (methods text, not just figure captions): the paper's own FDTD
simulations use propagation along x, E-field polarization along y, and
H-field polarization along z. That is a DIFFERENT axis labeling than
miepython's internal convention (k along z, E along x, H along y).
Mapping by physical ROLE rather than axis label:

    plane "xy" (z=0 in miepython's frame): contains E-axis and H-axis,
        i.e. transverse to k. This is the physical plane the paper calls
        its "YZ plane" (their y=E-axis, z=H-axis, x=k-axis).
    plane "yz" (x=0 in miepython's frame): contains H-axis and k-axis.
    plane "xz" (y=0 in miepython's frame): contains E-axis and k-axis.
        This is the physical plane the paper calls its "XY plane"
        (their x=k-axis, y=E-axis).

So by role-matching: paper's H^2 map (their "YZ", the E-H transverse
plane) corresponds to miepython's "xy" plane here, and the paper's E^2
map (their "XY", the k-E plane) corresponds to miepython's "xz" plane
here -- NOT the "yz"/"xy" pairing used in earlier versions of this file.
This has not been empirically confirmed yet (see scripts comparing all
three planes against Figure S5a/b) -- treat "xy" for H2 and "xz" for E2
as the leading hypothesis, not a settled fix.
"""

import numpy as np
from miepython.field import eh_near_cartesian
from optical_constants import cu2o_refractive_index

N_MEDIUM_AIR = 1.0
N_MEDIUM_WATER = 1.33


def near_field_map(radius_nm, wavelength_nm, n_medium=N_MEDIUM_AIR,
                    n=None, k=None, extent_factor=2.5, grid_points=121, plane="yz"):
    """
    Computes |E/E0|^2 and |H/H0|^2 on a 2D grid through the sphere center.

    plane: one of 'yz' (x=0, contains H-axis and k), 'xy' (z=0, contains
    E-axis and H-axis, transverse to k), or 'xz' (y=0, contains E-axis
    and k). See module docstring for how these map to the paper's own
    axis labels.

    Returns: (U, V, E2, H2) where U, V are the 2D coordinate grids (nm)
    and E2, H2 are |E/E0|^2, |H/H0|^2 on that grid.
    """
    if n is None or k is None:
        n_arr, k_arr = cu2o_refractive_index(wavelength_nm)
        n, k = float(np.atleast_1d(n_arr)[0]), float(np.atleast_1d(k_arr)[0])

    m_sphere = n - 1j * k
    d_sphere = 2 * radius_nm
    extent = extent_factor * radius_nm

    u = np.linspace(-extent, extent, grid_points)
    v = np.linspace(-extent, extent, grid_points)
    U, V = np.meshgrid(u, v, indexing="xy")

    if plane == "yz":
        X, Y, Z = np.zeros_like(U), U, V
    elif plane == "xy":
        X, Y, Z = U, V, np.zeros_like(U)
    elif plane == "xz":
        X, Y, Z = U, np.zeros_like(U), V
    else:
        raise ValueError("plane must be 'yz', 'xy', or 'xz'")

    E_xyz, H_xyz = eh_near_cartesian(wavelength_nm, d_sphere, m_sphere, n_medium, X, Y, Z)

    E2 = np.sum(np.abs(E_xyz) ** 2, axis=0)
    H2 = np.sum(np.abs(H_xyz) ** 2, axis=0)

    return U, V, E2, H2


def peak_enhancement(radius_nm, wavelength_nm, n_medium=N_MEDIUM_AIR, n=None, k=None, plane="yz", **kwargs):
    """Convenience wrapper: returns (max E^2/E0^2, max H^2/H0^2) over the grid."""
    _, _, E2, H2 = near_field_map(radius_nm, wavelength_nm, n_medium, n, k, plane=plane, **kwargs)
    return float(np.max(E2)), float(np.max(H2))