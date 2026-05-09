#!/usr/bin/env python3
# coding: utf-8

"""Compute cylindrical 2D power spectra from wedge-filtered mean-subtracted noisy Tb maps and build covariance.

This script is designed for HPC runs.

Inputs
------
- Training wedge-filtered mean-subtracted noisy Tb map dictionary (.npy): seed -> 3D cube (mK)
- Validation wedge-filtered mean-subtracted noisy Tb map dictionary (.npy): seed -> 3D cube (mK)

Outputs (written to --outdir)
---------------------------
1) 2DPS files
   - training 2DPS dictionary (.npy)
   - validation 2DPS dictionary (.npy)
2) Diagnostic plots
   - First 5 training + first 5 validation 2DPS (2 rows x 5 columns)
3) Covariance products (training+validation combined)
   - covariance matrix (.npy)
   - correlation matrix (.npy)
   - side-by-side covariance & correlation plot (.png/.pdf)

Notes on units
--------------
These cubes are already brightness-temperature fluctuations in mK (and mean-subtracted).
Therefore, pass `units=''` to `script.ionization_map.get_binned_powspec_cylindrical` so that
SCRIPT does NOT apply an extra T̄_b(z)^2 conversion.

Implementation note
-------------------
To avoid re-initializing FFTW plans for every seed, this script builds a minimal dummy
`matter_fields`-like object once and passes it to SCRIPT's `ionization_map` so FFTW plans
and k-grids are cached.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tqdm import tqdm

import script
import script_fortran_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cylindrical 2DPS (linear k-bins) from wedge-filtered mean-subtracted noisy Tb maps, "
            "save training/validation 2DPS dictionaries, then build covariance + correlation "
            "from the combined 2DPS set."
        )
    )
    parser.add_argument(
        "--train-input",
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "maps_wedge_filtered/training_wedge_filtered_main.npy"
        ),
        help="Path to training wedge-filtered mean-subtracted noisy Tb map dictionary (.npy).",
    )
    parser.add_argument(
        "--val-input",
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "maps_wedge_filtered/validation_wedge_filtered_main.npy"
        ),
        help="Path to validation wedge-filtered mean-subtracted noisy Tb map dictionary (.npy).",
    )
    parser.add_argument(
        "--outdir",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/cov_files"
        ),
        help="Directory to store outputs (2DPS, covariance, plots, logs).",
    )
    parser.add_argument(
        "--logfile",
        default="temp_plus_noise_ms_wedge_filtered_to_2dps_and_covariance_script_linear.log",
        help="Log filename inside outdir.",
    )

    # Geometry / binning
    parser.add_argument("--box", type=float, default=128.0, help="Box size in cMpc/h.")
    parser.add_argument("--nbins_par", type=int, default=10, help="Number of k_parallel bins.")
    parser.add_argument("--nbins_perp", type=int, default=10, help="Number of k_perp bins.")

    # Cosmology passed to SCRIPT backend (kept for consistent initialization)
    parser.add_argument("--z_target", type=float, default=7.0, help="Target redshift passed to script backend.")
    parser.add_argument("--omega_m", type=float, default=0.308, help="Omega_m passed to script backend.")
    parser.add_argument("--hubble", type=float, default=0.678, help="h parameter passed to script backend.")

    # SCRIPT power-spectrum options
    parser.add_argument(
        "--units",
        default="",
        choices=["", "mK", "K", "xHI"],
        help=(
            "Units option passed to SCRIPT get_binned_powspec_cylindrical. "
            "For brightness-temperature maps already in mK, keep this as '' (default)."
        ),
    )
    parser.add_argument(
        "--convolve",
        action="store_true",
        help="If set, deconvolve the PS using SCRIPT smoothing kernel.",
    )

    # Plotting
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output figures.")

    # Optional: quick test on a subset
    parser.add_argument(
        "--max_maps",
        type=int,
        default=None,
        help="If set, process only the first N maps from each dataset (debug option).",
    )

    return parser.parse_args()


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_seed_dict(path: str, label: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} input not found: {path}")

    data = np.load(path, allow_pickle=True)
    if hasattr(data, "item"):
        data = data.item()

    if not isinstance(data, dict):
        raise TypeError(f"{label} input must be a seed-keyed dictionary: {path}")

    out: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        out[str(k)] = np.asarray(v)

    return out


def validate_cubic_3d(field: np.ndarray, *, label: str) -> None:
    if field.ndim != 3:
        raise ValueError(f"{label} map must be 3D. Got shape {field.shape}.")
    if not (field.shape[0] == field.shape[1] == field.shape[2]):
        raise ValueError(f"{label} map must be cubic. Got shape {field.shape}.")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{label} map contains non-finite values.")


def setup_linear_k_bins(
    ngrid: int,
    box: float,
    nbins_par: int,
    nbins_perp: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear k-bin edges based on the FFT grid."""

    k1d = np.abs(np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi).astype(np.float64)
    nonzero = k1d[k1d > 0.0]
    if nonzero.size == 0:
        raise ValueError(f"Invalid FFT k-grid for ngrid={ngrid}.")

    kmin = float(np.min(nonzero))
    kmax = float(np.max(nonzero))
    if kmax <= kmin:
        raise ValueError(f"Invalid k-range: kmin={kmin}, kmax={kmax}")

    k_par_edges = np.linspace(kmin, kmax, nbins_par + 1)
    k_perp_edges = np.linspace(kmin, kmax, nbins_perp + 1)
    k_par_bins = 0.5 * (k_par_edges[:-1] + k_par_edges[1:])
    k_perp_bins = 0.5 * (k_perp_edges[:-1] + k_perp_edges[1:])
    return k_par_edges, k_perp_edges, k_par_bins, k_perp_bins


def build_script_ionization_map(
    shape: Tuple[int, int, int],
    *,
    box: float,
    z_target: float,
    omega_m: float,
    hubble: float,
) -> "script.ionization_map":
    omega_l = float(1.0 - omega_m)

    class _DummyCosmo:
        def __init__(self, omega_m_: float, omega_l_: float, h_: float):
            self.omega_m = float(omega_m_)
            self.omega_l = float(omega_l_)
            self.h = float(h_)

    class _DummySimData:
        def __init__(self, box_: float, z_: float, omega_m_: float, omega_l_: float, h_: float):
            self.box = float(box_)
            self.z = float(z_)
            self.cosmo = _DummyCosmo(omega_m_, omega_l_, h_)

    class _DummyMatterFields:
        def __init__(
            self,
            shape_: Tuple[int, int, int],
            box_: float,
            z_: float,
            omega_m_: float,
            omega_l_: float,
            h_: float,
        ):
            self.ngrid = int(shape_[0])
            self.default_simulation_data = _DummySimData(box_, z_, omega_m_, omega_l_, h_)

            # Fields unused for PS computation but required by SCRIPT initialization.
            self.densitycontr_arr = np.zeros(shape_, dtype=np.float32)
            self.velocity_arr = np.zeros((3, self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)

            # Cache FFTW plan and k-grids once.
            FFTW_ESTIMATE = 64
            FFTW_IFINV = 1
            self.plan, self.kfft = script_fortran_modules.powspec.initialize_plan(
                self.ngrid, FFTW_IFINV, FFTW_ESTIMATE, float(box_)
            )
            self.kmag = script_fortran_modules.powspec.get_kmag(self.kfft)
            self.k_par, self.k_perp = script_fortran_modules.powspec.get_kparperp(self.kfft)

    dummy_mf = _DummyMatterFields(shape, float(box), float(z_target), float(omega_m), omega_l, float(hubble))
    return script.ionization_map(matter_fields=dummy_mf, method="PC")


def compute_ps2d_for_field(
    ion_map: "script.ionization_map",
    field: np.ndarray,
    *,
    k_par_edges: np.ndarray,
    k_perp_edges: np.ndarray,
    convolve: bool,
    units: str,
) -> Tuple[np.ndarray, np.ndarray]:
    ps2d, kount = ion_map.get_binned_powspec_cylindrical(
        field.astype(np.float32, copy=False),
        k_par_edges,
        k_perp_edges,
        convolve=bool(convolve),
        units=str(units),
    )

    ps2d = np.asarray(ps2d, dtype=np.float64)
    kount = np.asarray(kount, dtype=np.float64)

    expected_shape = (k_par_edges.size - 1, k_perp_edges.size - 1)
    if ps2d.shape != expected_shape:
        raise RuntimeError(f"Expected ps2d shape {expected_shape}, got {ps2d.shape}.")
    if kount.shape != expected_shape:
        raise RuntimeError(f"Expected kount shape {expected_shape}, got {kount.shape}.")

    empty = kount == 0
    if np.any(empty):
        ps2d = ps2d.copy()
        ps2d[empty] = 0.0

    if not np.all(np.isfinite(ps2d)):
        raise RuntimeError("Non-finite entries found in 2DPS result.")

    return ps2d, kount


def iter_first_n(keys: List[str], n: Optional[int]) -> Iterable[str]:
    if n is None:
        return keys
    return keys[: max(0, int(n))]


def convert_dataset_to_2dps(
    maps_dict: Dict[str, np.ndarray],
    *,
    label: str,
    expected_shape: Tuple[int, int, int],
    ion_map: "script.ionization_map",
    k_par_edges: np.ndarray,
    k_perp_edges: np.ndarray,
    convolve: bool,
    units: str,
    max_maps: Optional[int],
    logger: logging.Logger,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    keys = list(maps_dict.keys())
    if len(keys) == 0:
        raise ValueError(f"{label} dataset is empty")

    ps_dict: Dict[str, np.ndarray] = {}
    ps_list: List[np.ndarray] = []

    kount_ref: Optional[np.ndarray] = None

    for seed in tqdm(iter_first_n(keys, max_maps), desc=f"{label}: maps -> 2DPS", ncols=100):
        field = np.asarray(maps_dict[seed], dtype=np.float64)
        validate_cubic_3d(field, label=f"{label} seed {seed}")
        if field.shape != expected_shape:
            raise ValueError(
                f"{label} map shape mismatch for seed {seed}: {field.shape} vs {expected_shape}"
            )

        ps2d, kount = compute_ps2d_for_field(
            ion_map,
            field,
            k_par_edges=k_par_edges,
            k_perp_edges=k_perp_edges,
            convolve=convolve,
            units=units,
        )

        if kount_ref is None:
            kount_ref = kount
        else:
            if not np.array_equal(kount_ref, kount):
                raise RuntimeError(f"{label}: kount changed for seed {seed} (should be constant).")

        ps2d32 = ps2d.astype(np.float32)
        ps_dict[str(seed)] = ps2d32
        ps_list.append(ps2d32)

    ps_array = np.stack(ps_list, axis=0)  # (N, nbins_par, nbins_perp)

    if kount_ref is None:
        raise RuntimeError(f"{label}: no kount produced")

    logger.info("%s converted: %d maps -> 2DPS shape %s", label, ps_array.shape[0], ps_array.shape)
    return ps_dict, ps_array, kount_ref.astype(np.float64)


def plot_first5_train_val_2dps(
    *,
    train_keys: List[str],
    val_keys: List[str],
    train_ps_dict: Dict[str, np.ndarray],
    val_ps_dict: Dict[str, np.ndarray],
    k_par_edges: np.ndarray,
    k_perp_edges: np.ndarray,
    out_png: str,
    out_pdf: str,
    dpi: int,
    cbar_label: str,
) -> None:
    n_show = 5
    train_show = train_keys[: min(n_show, len(train_keys))]
    val_show = val_keys[: min(n_show, len(val_keys))]

    # Build list for global log scale
    ps_to_scale: List[np.ndarray] = []
    for seed in train_show:
        ps_to_scale.append(np.asarray(train_ps_dict[seed], dtype=np.float64))
    for seed in val_show:
        ps_to_scale.append(np.asarray(val_ps_dict[seed], dtype=np.float64))

    positive_vals: List[float] = []
    max_val = 0.0
    for arr in ps_to_scale:
        finite = np.isfinite(arr)
        pos = arr[finite & (arr > 0.0)]
        if pos.size > 0:
            positive_vals.append(float(np.min(pos)))
            max_val = max(max_val, float(np.max(pos)))

    use_log = len(positive_vals) > 0 and max_val > 0.0
    if use_log:
        vmin = float(np.min(positive_vals))
        vmax = float(max_val)
        if vmax <= vmin:
            vmax = vmin * 1.0001
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Dedicated GridSpec column for the shared colorbar (avoids overlap).
    fig = plt.figure(figsize=(18.5, 7.2))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=6,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.06],
        wspace=0.25,
        hspace=0.28,
    )
    axes = np.empty((2, 5), dtype=object)
    for row in range(2):
        for col in range(5):
            axes[row, col] = fig.add_subplot(gs[row, col])
    cax = fig.add_subplot(gs[:, 5])

    extent = [
        float(k_perp_edges[0]),
        float(k_perp_edges[-1]),
        float(k_par_edges[0]),
        float(k_par_edges[-1]),
    ]

    im_for_cbar = None

    # Row 0: training
    for col in range(5):
        ax = axes[0, col]
        if col < len(train_show):
            seed = train_show[col]
            ps2d = np.asarray(train_ps_dict[seed], dtype=np.float64)
            ps_plot = np.where(ps2d > 0.0, ps2d, np.nan) if use_log else ps2d
            im = ax.imshow(
                ps_plot,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                norm=norm,
            )
            im_for_cbar = im
            ax.set_title(f"Train seed {seed}", fontsize=9)
        else:
            ax.axis("off")

        if col == 0:
            ax.set_ylabel(r"$k_\parallel$ [h/cMpc]", fontsize=10)
        ax.set_xlabel(r"$k_\perp$ [h/cMpc]", fontsize=10)

    # Row 1: validation
    for col in range(5):
        ax = axes[1, col]
        if col < len(val_show):
            seed = val_show[col]
            ps2d = np.asarray(val_ps_dict[seed], dtype=np.float64)
            ps_plot = np.where(ps2d > 0.0, ps2d, np.nan) if use_log else ps2d
            im = ax.imshow(
                ps_plot,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                norm=norm,
            )
            im_for_cbar = im
            ax.set_title(f"Val seed {seed}", fontsize=9)
        else:
            ax.axis("off")

        if col == 0:
            ax.set_ylabel(r"$k_\parallel$ [h/cMpc]", fontsize=10)
        ax.set_xlabel(r"$k_\perp$ [h/cMpc]", fontsize=10)

    fig.suptitle(
        "First 5 cylindrical 2D power spectra (wedge-filtered; training: top, validation: bottom)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, cax=cax)
        cbar.set_label(cbar_label)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_cov_corr(flat_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cov, corr) for flat sample matrix shaped (Nsamples, Nbins)."""
    cov = np.cov(flat_samples, rowvar=False, ddof=1)

    diag = np.diag(cov)
    std = np.sqrt(np.maximum(diag, 0.0))
    denom = np.outer(std, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, cov / denom, 0.0)

    corr = np.clip(corr, -1.0, 1.0)
    return cov.astype(np.float64), corr.astype(np.float64)


def plot_cov_and_corr(
    *,
    cov: np.ndarray,
    corr: np.ndarray,
    out_png: str,
    out_pdf: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0))

    im0 = axes[0].imshow(cov, origin="lower", cmap="viridis", aspect="equal")
    axes[0].set_title("Covariance matrix")
    axes[0].set_xlabel("Flattened 2DPS bin index")
    axes[0].set_ylabel("Flattened 2DPS bin index")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$\mathrm{Cov}_{ij}$")

    im1 = axes[1].imshow(corr, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    axes[1].set_title("Correlation matrix")
    axes[1].set_xlabel("Flattened 2DPS bin index")
    axes[1].set_ylabel("Flattened 2DPS bin index")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"$\rho_{ij}$")

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, args.logfile)
    logger = setup_logging(log_path)

    start = time.time()

    logger.info("=" * 90)
    logger.info("WEDGE-FILTERED MS TB MAPS -> CYLINDRICAL 2DPS -> COV/CORR (SCRIPT, LINEAR) STARTED")
    logger.info("=" * 90)
    logger.info("Training input: %s", args.train_input)
    logger.info("Validation input: %s", args.val_input)
    logger.info("Output directory: %s", args.outdir)
    logger.info("Units: '%s' | Convolve: %s", args.units, args.convolve)

    # Load training first to determine shape and build SCRIPT backend once.
    training_maps = load_seed_dict(args.train_input, "training")
    logger.info("Loaded training maps: %d", len(training_maps))

    first_key = next(iter(training_maps.keys()))
    first_map = np.asarray(training_maps[first_key], dtype=np.float64)
    validate_cubic_3d(first_map, label=f"training seed {first_key}")

    ngrid = int(first_map.shape[0])
    logger.info("Detected ngrid = %d", ngrid)

    ion_map = build_script_ionization_map(
        first_map.shape,
        box=args.box,
        z_target=args.z_target,
        omega_m=args.omega_m,
        hubble=args.hubble,
    )

    k_par_edges, k_perp_edges, _, _ = setup_linear_k_bins(
        ngrid=ngrid,
        box=args.box,
        nbins_par=args.nbins_par,
        nbins_perp=args.nbins_perp,
    )

    logger.info("k_par_edges: %s", np.array2string(k_par_edges, precision=6))
    logger.info("k_perp_edges: %s", np.array2string(k_perp_edges, precision=6))

    train_ps_dict, train_ps_array, kount_ref_train = convert_dataset_to_2dps(
        training_maps,
        label="training",
        expected_shape=first_map.shape,
        ion_map=ion_map,
        k_par_edges=k_par_edges,
        k_perp_edges=k_perp_edges,
        convolve=args.convolve,
        units=args.units,
        max_maps=args.max_maps,
        logger=logger,
    )

    # Free memory from large map dictionary before loading validation.
    del training_maps

    validation_maps = load_seed_dict(args.val_input, "validation")
    logger.info("Loaded validation maps: %d", len(validation_maps))

    val_ps_dict, val_ps_array, kount_ref_val = convert_dataset_to_2dps(
        validation_maps,
        label="validation",
        expected_shape=first_map.shape,
        ion_map=ion_map,
        k_par_edges=k_par_edges,
        k_perp_edges=k_perp_edges,
        convolve=args.convolve,
        units=args.units,
        max_maps=args.max_maps,
        logger=logger,
    )

    del validation_maps

    if not np.array_equal(kount_ref_train, kount_ref_val):
        raise RuntimeError("Training and validation kount arrays do not match")

    train_2dps_out = os.path.join(args.outdir, "2dps_training_wedge_filtered_main.npy")
    val_2dps_out = os.path.join(args.outdir, "2dps_validation_wedge_filtered_main.npy")

    first5_png = os.path.join(args.outdir, "first5_training_validation_2dps_wedge_filtered_main.png")
    first5_pdf = os.path.join(args.outdir, "first5_training_validation_2dps_wedge_filtered_main.pdf")

    cov_out = os.path.join(args.outdir, "covariance_training_plus_validation_2dps_wedge_filtered_main.npy")
    corr_out = os.path.join(args.outdir, "correlation_training_plus_validation_2dps_wedge_filtered_main.npy")

    covcorr_png = os.path.join(
        args.outdir,
        "covariance_correlation_training_plus_validation_2dps_wedge_filtered_main.png",
    )
    covcorr_pdf = os.path.join(
        args.outdir,
        "covariance_correlation_training_plus_validation_2dps_wedge_filtered_main.pdf",
    )

    np.save(train_2dps_out, train_ps_dict)
    np.save(val_2dps_out, val_ps_dict)

    cbar_label = r"$P(k_\parallel, k_\perp)$"
    plot_first5_train_val_2dps(
        train_keys=list(train_ps_dict.keys()),
        val_keys=list(val_ps_dict.keys()),
        train_ps_dict=train_ps_dict,
        val_ps_dict=val_ps_dict,
        k_par_edges=k_par_edges,
        k_perp_edges=k_perp_edges,
        out_png=first5_png,
        out_pdf=first5_pdf,
        dpi=args.dpi,
        cbar_label=cbar_label,
    )

    combined = np.concatenate([train_ps_array, val_ps_array], axis=0)
    flat = combined.reshape(combined.shape[0], -1).astype(np.float64)

    logger.info("Combined 2DPS array shape: %s", combined.shape)
    logger.info("Flattened samples shape: %s", flat.shape)

    cov, corr = compute_cov_corr(flat)

    np.save(cov_out, cov)
    np.save(corr_out, corr)

    plot_cov_and_corr(cov=cov, corr=corr, out_png=covcorr_png, out_pdf=covcorr_pdf, dpi=args.dpi)

    elapsed = time.time() - start

    logger.info("Saved training 2DPS dict: %s", train_2dps_out)
    logger.info("Saved validation 2DPS dict: %s", val_2dps_out)
    logger.info("Saved first-5 plot: %s", first5_png)
    logger.info("Saved covariance: %s", cov_out)
    logger.info("Saved correlation: %s", corr_out)
    logger.info("Saved cov/corr plot: %s", covcorr_png)
    logger.info("Runtime: %.2f s (%.2f min)", elapsed, elapsed / 60.0)
    logger.info("Done")


if __name__ == "__main__":
    main()
