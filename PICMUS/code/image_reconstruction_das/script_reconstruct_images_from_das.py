#!/usr/bin/env python3
# -- Script to be used as an example to manipulate the provided dataset
#
# -- After choosing the specific configuration through acquisition_type,
# -- phantom_type and data_type parameters, this script allows reconstructing
# -- images for evaluation for different choices of steered plane waves involved
# -- in the compounding scheme (specified by the pw_indices parameter)
#
# -- The implemented method (to be used as example) corresponds to the standard Delay
# -- And Sum (DAS) technique with apodization in reception
#
# -- Authors: Olivier Bernard (olivier.bernard@creatis.insa-lyon.fr)
# --          Alfonso Rodriguez-Molares (alfonso.r.molares@ntnu.no)
#
# -- $Date: 2016/03/01 $

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

import h5py
import numpy as np

try:
    from scipy.interpolate import interp1d
    from scipy.signal import hilbert
except Exception as exc:
    raise ImportError(
        "This script requires scipy (for spline interpolation and Hilbert envelope). "
        "Please install scipy first."
    ) from exc


# -- utilities
def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _as_column_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.float64)
    return x.reshape(-1, 1)


def _read_scalar(dset) -> float:
    return np.asarray(dset[()]).reshape(-1)[0].item()


def _interp1_spline(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    # MATLAB: interp1(...,'spline',0) with out-of-range = 0
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y)
    xq = np.asarray(xq).astype(np.float64)
    if np.iscomplexobj(y):
        f_real = interp1d(x, y.real, kind="cubic", bounds_error=False, fill_value=0.0)
        f_imag = interp1d(x, y.imag, kind="cubic", bounds_error=False, fill_value=0.0)
        return f_real(xq) + 1j * f_imag(xq)
    f = interp1d(x, y, kind="cubic", bounds_error=False, fill_value=0.0)
    return f(xq)


def _interp1_linear_2d(x: np.ndarray, y2d: np.ndarray, xq: np.ndarray) -> np.ndarray:
    # Interp along axis=0 for each column, out-of-range = 0
    x = np.asarray(x).astype(np.float64)
    xq = np.asarray(xq).astype(np.float64)
    out = np.zeros((len(xq), y2d.shape[1]), dtype=y2d.dtype)
    for col in range(y2d.shape[1]):
        out[:, col] = np.interp(xq, x, y2d[:, col], left=0.0, right=0.0)
    return out


def apodization(distance: np.ndarray, aperture: np.ndarray, window: str) -> np.ndarray:
    # -- Function which assigns different apodization to a set of pixels and elements
    if window == "none":
        return np.ones_like(distance, dtype=np.float64)
    if window == "boxcar":
        return (distance <= aperture / 2).astype(np.float64)
    if window == "hanning":
        return (distance <= aperture / 2).astype(np.float64) * (
            0.5 + 0.5 * np.cos(2 * np.pi * distance / aperture)
        )
    if window == "hamming":
        return (distance <= aperture / 2).astype(np.float64) * (
            0.53836 + 0.46164 * np.cos(2 * np.pi * distance / aperture)
        )
    if window.startswith("tukey"):
        roll = {"tukey25": 0.25, "tukey50": 0.5, "tukey75": 0.75}.get(window)
        if roll is None:
            raise ValueError(
                "Unknown window type. Known types are: boxcar, hamming, hanning, "
                "tukey25, tukey50, tukey75."
            )
        return (
            (distance < (aperture / 2 * (1 - roll)))
            + (distance > (aperture / 2 * (1 - roll)))
            * (distance < (aperture / 2))
            * 0.5
            * (1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2)))
        ).astype(np.float64)
    raise ValueError(
        "Unknown window type. Known types are: boxcar, hamming, hanning, "
        "tukey25, tukey50, tukey75."
    )


def envelope(p: np.ndarray) -> np.ndarray:
    # -- Function which computes the envelope of a modulated signal
    p = np.asarray(p)
    if p.ndim == 3:
        reshaped = p.reshape(p.shape[0], -1, order="F")
        env = np.abs(hilbert(reshaped, axis=0))
        return env.reshape(p.shape, order="F")
    return np.abs(hilbert(p, axis=0))


@dataclass
class linear_scan:
    x_axis: np.ndarray | None = None
    z_axis: np.ndarray | None = None

    # derived
    x_matrix: np.ndarray | None = None
    z_matrix: np.ndarray | None = None
    x: np.ndarray | None = None
    z: np.ndarray | None = None
    pixels: int | None = None

    def _update(self) -> None:
        if self.x_axis is None or self.z_axis is None:
            return
        x_matrix, z_matrix = np.meshgrid(self.x_axis, self.z_axis, indexing="xy")
        self.x_matrix = x_matrix
        self.z_matrix = z_matrix
        self.x = x_matrix.reshape(-1, order="F")
        self.z = z_matrix.reshape(-1, order="F")
        self.pixels = self.x.size

    def read_file(self, filename: str) -> None:
        with h5py.File(filename, "r") as f:
            g = f["/US/US_DATASET0000"]
            self.x_axis = np.array(g["x_axis"]).astype(np.float64).reshape(-1)
            self.z_axis = np.array(g["z_axis"]).astype(np.float64).reshape(-1)
        self._update()


@dataclass
class us_dataset:
    name: str | None = None
    creation_date: str | None = None
    probe_geometry: np.ndarray | None = None  # (channels, 3)
    data: np.ndarray | None = None  # (samples, channels, firings)
    c0: float | None = None
    initial_time: float | None = None
    sampling_frequency: float | None = None
    modulation_frequency: float | None = None
    PRF: float | None = None
    angles: np.ndarray | None = None
    samples: int | None = None
    channels: int | None = None
    firings: int | None = None

    def read_file(self, filename: str) -> None:
        with h5py.File(filename, "r") as f:
            g = f["/US/US_DATASET0000"]
            self.c0 = _read_scalar(g["sound_speed"])
            self.initial_time = _read_scalar(g["initial_time"])
            self.sampling_frequency = _read_scalar(g["sampling_frequency"])
            self.PRF = _read_scalar(g["PRF"])
            self.modulation_frequency = _read_scalar(g["modulation_frequency"])
            self.angles = np.array(g["angles"]).astype(np.float64).reshape(-1)
            probe_geometry = np.array(g["probe_geometry"]).astype(np.float64)
            if probe_geometry.shape[0] == 3 and probe_geometry.shape[1] != 3:
                probe_geometry = probe_geometry.T
            if probe_geometry.shape[1] != 3:
                raise ValueError("probe_geometry must be shaped as (channels, 3)")
            self.probe_geometry = probe_geometry
            real = np.array(g["data/real"]).astype(np.float64)
            imag = np.array(g["data/imag"]).astype(np.float64)
            data = real + 1j * imag

        channels = self.probe_geometry.shape[0]
        firings = self.angles.size
        # HDF5 ordering from MATLAB often appears as (firings, channels, samples)
        if data.ndim == 3 and data.shape[0] == firings and data.shape[1] == channels:
            data = np.transpose(data, (2, 1, 0))
        elif data.ndim == 3 and data.shape[1] == firings and data.shape[2] == channels:
            data = np.transpose(data, (0, 2, 1))
        self.data = data
        self.samples = data.shape[0]
        self.channels = data.shape[1]
        self.firings = data.shape[2]


@dataclass
class us_image:
    name: str = ""
    author: str = ""
    affiliation: str = ""
    algorithm: str = ""
    creation_date: str = ""
    scan: linear_scan | None = None
    number_plane_waves: np.ndarray | None = None
    data: np.ndarray | None = None  # envelope (z, x, frames)
    transmit_f_number: float | None = None
    transmit_apodization_window: str = ""
    receive_f_number: float | None = None
    receive_apodization_window: str = ""
    version: str = "v0.10"

    def show(self, dynamic_range: float = 60, frame_list: Iterable[int] | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not installed; skip image display.")
            return
        if self.data is None or self.scan is None:
            return
        if frame_list is None:
            frame_list = range(self.data.shape[2])
        x_mm = self.scan.x_axis * 1e3
        z_mm = self.scan.z_axis * 1e3
        x_lim = [np.min(self.scan.x_matrix) * 1e3, np.max(self.scan.x_matrix) * 1e3]
        z_lim = [np.min(self.scan.z_matrix) * 1e3, np.max(self.scan.z_matrix) * 1e3]
        for f in frame_list:
            env = self.data[:, :, f]
            im = 20 * np.log10(env / np.max(env))
            vrange = (-dynamic_range, 0)
            plt.figure()
            plt.imshow(
                im,
                extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
                cmap="gray",
                vmin=vrange[0],
                vmax=vrange[1],
                aspect="equal",
            )
            plt.xlabel("x [mm]")
            plt.ylabel("z [mm]")
            plt.title(f"{self.name}\n {self.number_plane_waves[f]} plane waves")
            plt.xlim(x_lim)
            plt.ylim(z_lim[::-1])
            plt.colorbar()
            plt.show()

    def write_file(self, filename: str) -> None:
        if not filename.lower().endswith(".hdf5"):
            raise ValueError("Only .hdf5 is supported in this python rewrite.")
        _ensure_dir(filename)
        with h5py.File(filename, "w") as f:
            f.attrs["version"] = "v.0.0.41"
            us = f.require_group("US")
            g = us.create_group("US_DATASET0000")
            g.attrs["type"] = "SR"
            g.attrs["signal_format"] = "ENV"
            g.attrs["name"] = self.name
            g.attrs["author"] = self.author
            g.attrs["affiliation"] = self.affiliation
            g.attrs["algorithm"] = self.algorithm
            g.attrs["version"] = self.version
            g.attrs["creation_date"] = self.creation_date

            scan_grp = g.create_group("scan")
            scan_grp.create_dataset("x_axis", data=np.asfortranarray(self.scan.x_axis.astype(np.float32)))
            scan_grp.create_dataset("z_axis", data=np.asfortranarray(self.scan.z_axis.astype(np.float32)))

            g.create_dataset("transmit_f_number", data=np.float32(self.transmit_f_number))
            g.create_dataset("receive_f_number", data=np.float32(self.receive_f_number))
            g.create_dataset(
                "transmit_apodization_window",
                data=np.string_(self.transmit_apodization_window),
            )
            g.create_dataset(
                "receive_apodization_window",
                data=np.string_(self.receive_apodization_window),
            )
            g.create_dataset(
                "number_plane_waves",
                data=np.asarray(self.number_plane_waves, dtype=np.float32),
            )

            data_grp = g.create_group("data")
            real = np.asfortranarray(self.data.astype(np.float32))
            imag = np.zeros_like(real, dtype=np.float32)
            data_grp.create_dataset("real", data=real)
            data_grp.create_dataset("imag", data=imag)


def das_iq(scan: linear_scan, dataset: us_dataset, pw_indices: List[np.ndarray]) -> us_image:
    # -- IQ DAS beamforming
    if dataset.modulation_frequency == 0:
        raise ValueError("The supplied dataset is not IQ")

    rx_f_number = 1.75
    rx_aperture = scan.z / rx_f_number
    rx_aperture_distance = np.abs(
        scan.x[:, None] - dataset.probe_geometry[:, 0][None, :]
    )
    receive_apodization = apodization(
        rx_aperture_distance, rx_aperture[:, None], "tukey25"
    )
    angular_apodization = np.ones((scan.pixels, dataset.firings), dtype=np.float64)

    beamformed_data = np.zeros((scan.pixels, len(pw_indices)), dtype=np.complex128)
    time_vector = dataset.initial_time + np.arange(dataset.samples) / dataset.sampling_frequency
    w0 = 2 * np.pi * dataset.modulation_frequency

    for f, pws in enumerate(pw_indices):
        for pw in pws:
            pw0 = int(pw) - 1  # MATLAB -> Python index
            transmit_delay = scan.z * np.cos(dataset.angles[pw0]) + scan.x * np.sin(
                dataset.angles[pw0]
            )
            for nrx in range(dataset.channels):
                receive_delay = np.sqrt(
                    (dataset.probe_geometry[nrx, 0] - scan.x) ** 2
                    + (dataset.probe_geometry[nrx, 2] - scan.z) ** 2
                )
                delay = (transmit_delay + receive_delay) / dataset.c0
                phase_shift = np.exp(1j * w0 * (delay - 2 * scan.z / dataset.c0))
                interp = _interp1_spline(
                    time_vector, dataset.data[:, nrx, pw0], delay
                )
                beamformed_data[:, f] += (
                    phase_shift
                    * angular_apodization[:, pw0]
                    * receive_apodization[:, nrx]
                    * interp
                )
            print(f"{pw} / {len(pws)}")

    envelope_beamformed_data = np.abs(
        beamformed_data.reshape(
            (len(scan.z_axis), len(scan.x_axis), len(pw_indices)), order="F"
        )
    )

    image = us_image("DAS-IQ beamforming")
    image.author = "Alfonso Rodriguez-Molares <alfonso.r.molares@ntnu.no>"
    image.affiliation = "Norwegian University of Science and Technology (NTNU)"
    image.algorithm = "Delay-and-Sum (IQ version)"
    image.creation_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    image.scan = scan
    image.number_plane_waves = np.array([len(pws) for pws in pw_indices], dtype=int)
    image.data = envelope_beamformed_data
    image.transmit_f_number = 0
    image.receive_f_number = rx_f_number
    image.transmit_apodization_window = "none"
    image.receive_apodization_window = "Tukey 25%"
    return image


def das_rf(scan: linear_scan, dataset: us_dataset, pw_indices: List[np.ndarray]) -> us_image:
    # -- RF DAS beamforming
    if dataset.modulation_frequency != 0:
        raise ValueError("The supplied dataset is not RF")

    time = np.arange(dataset.samples) / dataset.sampling_frequency + dataset.initial_time
    z_axis = time * dataset.c0 / 2
    rf_scan = linear_scan(scan.x_axis, z_axis)
    rf_scan._update()

    rx_f_number = 1.75
    rx_aperture = rf_scan.z / rx_f_number
    rx_aperture_distance = np.abs(
        rf_scan.x[:, None] - dataset.probe_geometry[:, 0][None, :]
    )
    receive_apodization = apodization(
        rx_aperture_distance, rx_aperture[:, None], "tukey25"
    )
    angular_apodization = np.ones((rf_scan.pixels, dataset.firings), dtype=np.float64)

    beamformed_data = np.zeros((rf_scan.pixels, len(pw_indices)), dtype=np.float64)
    time_vector = dataset.initial_time + np.arange(dataset.samples) / dataset.sampling_frequency

    for f, pws in enumerate(pw_indices):
        for pw in pws:
            pw0 = int(pw) - 1
            transmit_delay = rf_scan.z * np.cos(dataset.angles[pw0]) + rf_scan.x * np.sin(
                dataset.angles[pw0]
            )
            for nrx in range(dataset.channels):
                receive_delay = np.sqrt(
                    (dataset.probe_geometry[nrx, 0] - rf_scan.x) ** 2
                    + (dataset.probe_geometry[nrx, 2] - rf_scan.z) ** 2
                )
                delay = (transmit_delay + receive_delay) / dataset.c0
                interp = _interp1_spline(
                    time_vector, dataset.data[:, nrx, pw0], delay
                )
                beamformed_data[:, f] += (
                    angular_apodization[:, pw0] * receive_apodization[:, nrx] * interp
                )
            print(f"{pw} / {len(pws)}")

    beamformed_data = np.nan_to_num(beamformed_data)
    reshaped = beamformed_data.reshape(
        (len(rf_scan.z_axis), len(rf_scan.x_axis), len(pw_indices)), order="F"
    )
    envelope_beamformed_data = envelope(reshaped)

    resampled = np.zeros(
        (len(scan.z_axis), len(scan.x_axis), len(pw_indices)), dtype=np.float64
    )
    for f in range(len(pw_indices)):
        resampled[:, :, f] = _interp1_linear_2d(
            rf_scan.z_axis, envelope_beamformed_data[:, :, f], scan.z_axis
        )

    image = us_image("DAS-RF beamforming")
    image.author = "Alfonso Rodriguez-Molares <alfonso.r.molares@ntnu.no>"
    image.affiliation = "Norwegian University of Science and Technology (NTNU)"
    image.algorithm = "Delay-and-Sum (RF version)"
    image.creation_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    image.scan = scan
    image.number_plane_waves = np.array([len(pws) for pws in pw_indices], dtype=int)
    image.data = resampled
    image.transmit_f_number = 0
    image.receive_f_number = rx_f_number
    image.transmit_apodization_window = "none"
    image.receive_apodization_window = "Tukey 25%"
    return image


# -- Parameters
acquisition_type = 1       # -- 1 = simulation || 2 = experiments
phantom_type = 2           # -- 1 = resolution & distorsion || 2 = contrast & speckle quality
data_type = 1              # -- 1 = IQ || 2 = RF


# -- Parsing parameter choices
if acquisition_type == 1:
    acquisition = "simulation"
    acqui = "simu"
elif acquisition_type == 2:
    acquisition = "experiments"
    acqui = "expe"
else:  # -- Do deal with bad values
    acquisition = "simulation"
    acqui = "simu"

if phantom_type == 1:
    phantom = "resolution_distorsion"
elif phantom_type == 2:
    phantom = "contrast_speckle"
else:  # -- Do deal with bad values
    phantom = "resolution"

if data_type == 1:
    data = "iq"
elif data_type == 2:
    data = "rf"
else:  # -- Do deal with bad values
    data = "iq"


# -- Create path to load corresponding files
path_dataset = (
    "../../database/"
    + acquisition
    + "/"
    + phantom
    + "/"
    + phantom
    + "_"
    + acqui
    + "_dataset_"
    + data
    + ".hdf5"
)
path_scan = (
    "../../database/"
    + acquisition
    + "/"
    + phantom
    + "/"
    + phantom
    + "_"
    + acqui
    + "_scan.hdf5"
)
path_reconstruted_img = (
    "../../reconstructed_image/"
    + acquisition
    + "/"
    + phantom
    + "/"
    + phantom
    + "_"
    + acqui
    + "_img_from_"
    + data
    + ".hdf5"
)


# -- Read the corresponding dataset and the region where to reconstruct the image
dataset = us_dataset()
dataset.read_file(path_dataset)
scan = linear_scan()
scan.read_file(path_scan)


# -- Indices of plane waves to be used for each reconstruction
pw_indices = []
pw_indices.append(np.array([38], dtype=int))
pw_indices.append(
    np.clip(np.rint(np.linspace(1, dataset.firings, 3)), 1, dataset.firings).astype(int)
)
pw_indices.append(
    np.clip(np.rint(np.linspace(1, dataset.firings, 11)), 1, dataset.firings).astype(int)
)
pw_indices.append(np.arange(1, dataset.firings + 1, dtype=int))


# -- Reconstruct Bmode images for each pw_indices
print(
    f"Starting image reconstruction from {acquisition} for {phantom} using {data} dataset"
)
if data_type == 1:
    image = das_iq(scan, dataset, pw_indices)
elif data_type == 2:
    image = das_rf(scan, dataset, pw_indices)
else:  # -- Do deal with bad values
    image = das_iq(scan, dataset, pw_indices)
print("Reconstruction Done")
print(f'Result saved in "{path_reconstruted_img}"')


# -- Show the corresponding beamformed images
dynamic_range = 60
image.show(dynamic_range)


# -- Save results
image.write_file(path_reconstruted_img)
