"""Plane-wave Delay-and-Sum (DAS) beamformer
Python port of PICMUS das_rf.m

Reference
---------
Rodriguez-Molares & Bernard, PICMUS Challenge, 2016
  archive_to_download/code/src/beamformers/das_rf.m
"""

import numpy as np


def das_pw_rf(rf_3d, angles, probe_geometry, initial_time, fs, c,
              x_axis, z_axis, rx_f_number=1.75, chunk_size=40000,
              verbose=True):
    """Plane-wave DAS-RF beamforming with Tukey-25 % receive apodization.

    Parameters
    ----------
    rf_3d : (n_pw, n_elem, n_samp)
        RF data (real-valued).
    angles : (n_pw,)
        Plane-wave steering angles in radians.
    probe_geometry : (3, n_elem)
        Element positions [x; y; z] in metres.
    initial_time : float
        Time of first RF sample (s).
    fs : float
        Sampling frequency (Hz).
    c : float
        Speed of sound (m/s).
    x_axis : (n_x,)
        Lateral pixel grid (m).
    z_axis : (n_z,)
        Axial pixel grid (m).
    rx_f_number : float
        Receive f-number for dynamic aperture (default 1.75).
    chunk_size : int
        Pixels per chunk (controls peak memory ≈ chunk_size × n_elem × 8 B).
    verbose : bool

    Returns
    -------
    envelope : (n_z, n_x)
        DAS envelope (positive, linear scale).
    """
    from scipy.signal import hilbert as _hilbert

    n_pw, n_elem, n_samp = rf_3d.shape
    n_x, n_z = len(x_axis), len(z_axis)

    X, Z = np.meshgrid(x_axis, z_axis)
    xf = X.ravel().astype(np.float64)
    zf = Z.ravel().astype(np.float64)
    n_pix = len(xf)

    px = probe_geometry[0].astype(np.float64)
    pz = probe_geometry[2].astype(np.float64)
    rf = rf_3d.astype(np.float64)
    ang = angles.astype(np.float64)
    t0 = float(initial_time)

    beamformed = np.zeros(n_pix, dtype=np.float64)
    er = np.arange(n_elem)

    if verbose:
        print(f"    DAS: {n_pw} pw × {n_elem} elem → {n_z}×{n_x} pixels")

    for cs in range(0, n_pix, chunk_size):
        ce = min(cs + chunk_size, n_pix)
        xc, zc = xf[cs:ce], zf[cs:ce]
        nc = ce - cs

        aperture = zc / rx_f_number
        lat_dist = np.abs(xc[:, None] - px[None, :])
        apod = _tukey25(lat_dist, aperture[:, None])

        rx_delay = np.sqrt(
            (px[None, :] - xc[:, None]) ** 2 +
            (pz[None, :] - zc[:, None]) ** 2
        )

        buf = np.zeros(nc, dtype=np.float64)
        for pw in range(n_pw):
            tx = zc * np.cos(ang[pw]) + xc * np.sin(ang[pw])
            sf = ((tx[:, None] + rx_delay) / c - t0) * fs
            i0 = np.floor(sf).astype(np.intp)
            frac = sf - i0
            i1 = i0 + 1

            ok = (i0 >= 0) & (i1 < n_samp)
            i0c = np.clip(i0, 0, n_samp - 1)
            i1c = np.clip(i1, 0, n_samp - 1)

            rp = rf[pw]
            v = rp[er[None, :], i0c] * (1.0 - frac) + rp[er[None, :], i1c] * frac
            v *= ok
            buf += np.sum(apod * v, axis=1)

        beamformed[cs:ce] = buf

        if verbose:
            print(f"\r    DAS: {ce * 100 // n_pix:3d}%", end="", flush=True)

    if verbose:
        print()

    bf = beamformed.reshape(n_z, n_x)
    return np.abs(_hilbert(bf, axis=0))


def _tukey25(distance, aperture):
    """Tukey-25 % spatial receive apodization.

    Parameters
    ----------
    distance : (n, m)
        Lateral distance pixel → element (m).
    aperture : (n, 1) or broadcastable
        Full receive aperture = z / f_number (m).
    """
    alpha = 0.25
    r = 2.0 * distance / (aperture + 1e-10)

    out = np.zeros_like(r)
    flat = r <= (1.0 - alpha)
    out[flat] = 1.0
    taper = (r > (1.0 - alpha)) & (r <= 1.0)
    out[taper] = 0.5 * (1.0 + np.cos(np.pi / alpha * (r[taper] - 1.0 + alpha)))
    return out
