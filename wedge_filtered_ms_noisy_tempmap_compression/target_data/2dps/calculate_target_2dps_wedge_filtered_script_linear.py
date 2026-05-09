#!/usr/bin/env python3
# coding: utf-8

"""Compute wedge-filtered noisy-target cylindrical 2DPS and plot full pipeline.

Expected prepared inputs:
1. Raw target temperature map.
2. Noise map.
3. Noise-added target temperature map.
4. Mean-subtracted noisy target temperature map.
5. Wedge-filtered mean-subtracted noisy target temperature map.

The script computes a 10x10 linearly binned cylindrical 2DPS from the
wedge-filtered mean-subtracted noisy cube using the `script` backend and
creates a 2x7 diagnostic figure with row-wise operators.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 10x10 linear cylindrical 2DPS from a wedge-filtered "
            "mean-subtracted noisy target Tb cube and generate a 2x7 plot."
        )
    )
    parser.add_argument(
        "--raw_map",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/"
            "target_data/target_tempmap_seed1259935638.npy"
        ),
        help="Path to raw target Tb map (.npy), in mK.",
    )
    parser.add_argument(
        "--noise_map_npz",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/"
            "target_data/AAstar_noise21cm_cube_seed1259935638_target.npz"
        ),
        help="Path to target noise map container (.npz).",
    )
    parser.add_argument(
        "--noisy_map",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/"
            "target_data/target_tempmap_w_noise_seed1259935638.npy"
        ),
        help="Path to noise-added target Tb map (.npy), in mK.",
    )
    parser.add_argument(
        "--mean_subtracted_noisy_map",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/"
            "target_data/target_tempmap_w_noise_ms_seed1259935638.npy"
        ),
        help="Path to mean-subtracted noisy target Tb map (.npy), in mK.",
    )
    parser.add_argument(
        "--wedge_filtered_ms_noisy_map",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/"
            "target_data/target_tempmap_w_noise_ms_wedge_filtered_seed1259935638.npy"
        ),
        help="Path to wedge-filtered mean-subtracted noisy target Tb map (.npy), in mK.",
    )
    parser.add_argument(
        "--outdir",
        default=(
            "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb/"
            "target_data"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--output_prefix",
        default="target_tempmap_w_noise_ms_wedge_filtered_seed1259935638",
        help="Prefix for output filenames.",
    )
    parser.add_argument(
        "--logfile",
        default="calculate_target_2dps_wedge_filtered_script_linear.log",
        help="Log filename written in outdir.",
    )
    parser.add_argument("--box", type=float, default=128.0, help="Box size in cMpc/h.")
    parser.add_argument("--nbins_par", type=int, default=10, help="Number of k_parallel bins.")
    parser.add_argument("--nbins_perp", type=int, default=10, help="Number of k_perp bins.")
    parser.add_argument("--z_target", type=float, default=7.0, help="Redshift passed to script backend.")
    parser.add_argument("--omega_m", type=float, default=0.308, help="Omega_m passed to script backend.")
    parser.add_argument("--hubble", type=float, default=0.678, help="h passed to script backend.")
    parser.add_argument("--wedge_C", type=float, default=3.140347, help="Wedge slope C for k_parallel = C * k_perp.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
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


def validate_cube(cube: np.ndarray, name: str) -> None:
    if cube.ndim != 3:
        raise ValueError(f"{name} must be a 3D array. Got shape {cube.shape}.")
    if not (cube.shape[0] == cube.shape[1] == cube.shape[2]):
        raise ValueError(f"{name} must be cubic. Got shape {cube.shape}.")
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"{name} contains non-finite values.")


def load_noise_cube(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Noise map file not found: {npz_path}")

    with np.load(npz_path) as data:
        for key in ("noisecube_21cm", "noise_cube", "noise"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64)

        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=np.float64)

        raise KeyError(
            f"Could not determine noise-map key in {npz_path}. "
            f"Available keys: {data.files}"
        )


def setup_linear_k_bins(
    ngrid: int,
    box: float,
    nbins_par: int,
    nbins_perp: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Match the Fisher-script linear bin convention.
    k1d = np.abs(np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi).astype(np.float64)
    nonzero = k1d[k1d > 0.0]
    if nonzero.size == 0:
        raise ValueError(f"Invalid FFT k-grid for ngrid={ngrid}.")

    kmin = float(np.min(nonzero))
    kmax = float(np.max(nonzero))
    k_par_edges = np.linspace(kmin, kmax, nbins_par + 1)
    k_perp_edges = np.linspace(kmin, kmax, nbins_perp + 1)
    k_par_bins = 0.5 * (k_par_edges[:-1] + k_par_edges[1:])
    k_perp_bins = 0.5 * (k_perp_edges[:-1] + k_perp_edges[1:])
    return k_par_edges, k_perp_edges, k_par_bins, k_perp_bins


def compute_2dps_script_linear(
    mean_subtracted_tb_mk: np.ndarray,
    *,
    box: float,
    z_target: float,
    omega_m: float,
    hubble: float,
    k_par_edges: np.ndarray,
    k_perp_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    import script

    # The script backend only needs cosmology/box setup for PS helpers here.
    dummy_density = np.zeros_like(mean_subtracted_tb_mk, dtype=np.float32)
    ionization_map = script.ionization_map(
        matter_fields=None,
        densitycontr_arr=dummy_density,
        box=float(box),
        z=float(z_target),
        omega_m=float(omega_m),
        h=float(hubble),
        method="PC",
    )

    ps2d, kount = ionization_map.get_binned_powspec_cylindrical(
        mean_subtracted_tb_mk.astype(np.float32, copy=False),
        k_par_edges,
        k_perp_edges,
        convolve=False,
        units="",
    )

    ps2d = np.asarray(ps2d, dtype=np.float64)
    kount = np.asarray(kount, dtype=np.float64)

    expected_shape = (k_par_edges.size - 1, k_perp_edges.size - 1)
    if ps2d.shape != expected_shape:
        raise RuntimeError(f"Expected 2DPS shape {expected_shape}, got {ps2d.shape}.")
    if kount.shape != expected_shape:
        raise RuntimeError(f"Expected kount shape {expected_shape}, got {kount.shape}.")

    empty = kount == 0
    if np.any(empty):
        ps2d[empty] = 0.0
    if not np.all(np.isfinite(ps2d)):
        raise RuntimeError("Non-finite entries found in 2DPS result.")

    return ps2d, kount


def make_slice_xy(cube: np.ndarray, z_index: int) -> np.ndarray:
    return cube[:, :, z_index].T


def make_slice_xz(cube: np.ndarray, y_index: int) -> np.ndarray:
    return cube[:, y_index, :].T


def add_row_annotations(fig: plt.Figure, row_axes: np.ndarray) -> None:
    ax0, ax1, ax2, ax3, ax4, ax5, ax6 = row_axes
    p0 = ax0.get_position()
    p1 = ax1.get_position()
    p2 = ax2.get_position()
    p3 = ax3.get_position()
    p4 = ax4.get_position()
    p5 = ax5.get_position()
    p6 = ax6.get_position()

    # Lift symbols slightly above center to avoid x-label overlap.
    y_mid = p0.y0 + 0.62 * (p0.y1 - p0.y0)
    gap = 0.007

    x_plus = 0.5 * (p0.x1 + p1.x0)
    fig.text(x_plus, y_mid, "+", ha="center", va="center", fontsize=24, fontweight="bold")

    arrow_12 = FancyArrowPatch(
        posA=(p1.x1 + gap, y_mid),
        posB=(p2.x0 - gap, y_mid),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="black",
    )
    fig.add_artist(arrow_12)

    x_minus = 0.5 * (p2.x1 + p3.x0)
    fig.text(x_minus, y_mid, "-", ha="center", va="center", fontsize=26, fontweight="bold")

    arrow_34 = FancyArrowPatch(
        posA=(p3.x1 + gap, y_mid),
        posB=(p4.x0 - gap, y_mid),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="black",
    )
    fig.add_artist(arrow_34)

    arrow_45 = FancyArrowPatch(
        posA=(p4.x1 + gap, y_mid),
        posB=(p5.x0 - gap, y_mid),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="black",
    )
    fig.add_artist(arrow_45)

    arrow_56 = FancyArrowPatch(
        posA=(p5.x1 + gap, y_mid),
        posB=(p6.x0 - gap, y_mid),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="black",
    )
    fig.add_artist(arrow_56)


def make_overview_plot(
    *,
    raw_cube: np.ndarray,
    noise_cube: np.ndarray,
    noisy_cube: np.ndarray,
    noisy_mean: float,
    mean_subtracted_noisy_cube: np.ndarray,
    wedge_filtered_ms_noisy_cube: np.ndarray,
    ps2d: np.ndarray,
    k_par_edges: np.ndarray,
    k_perp_edges: np.ndarray,
    box: float,
    wedge_c: float,
    out_path: str,
    dpi: int,
) -> None:
    ngrid = raw_cube.shape[0]
    mid_z = ngrid // 2
    mid_y = ngrid // 2

    mean_cube = np.full_like(noisy_cube, noisy_mean, dtype=np.float64)

    row_data = [
        (
            make_slice_xy(raw_cube, mid_z),
            make_slice_xy(noise_cube, mid_z),
            make_slice_xy(noisy_cube, mid_z),
            make_slice_xy(mean_cube, mid_z),
            make_slice_xy(mean_subtracted_noisy_cube, mid_z),
            make_slice_xy(wedge_filtered_ms_noisy_cube, mid_z),
            "y [cMpc/h]",
            "x-y at z-mid",
        ),
        (
            make_slice_xz(raw_cube, mid_y),
            make_slice_xz(noise_cube, mid_y),
            make_slice_xz(noisy_cube, mid_y),
            make_slice_xz(mean_cube, mid_y),
            make_slice_xz(mean_subtracted_noisy_cube, mid_y),
            make_slice_xz(wedge_filtered_ms_noisy_cube, mid_y),
            "z [cMpc/h]",
            "x-z at y-mid",
        ),
    ]

    fig, axes = plt.subplots(2, 7, figsize=(36.0, 10.0))
    extent_map = [0.0, box, 0.0, box]
    extent_ps = [
        float(k_perp_edges[0]),
        float(k_perp_edges[-1]),
        float(k_par_edges[0]),
        float(k_par_edges[-1]),
    ]

    positive_ps = ps2d[np.isfinite(ps2d) & (ps2d > 0.0)]
    has_positive_ps = positive_ps.size > 0
    ps2d_plot = np.where(ps2d > 0.0, ps2d, np.nan)

    cmap_ps = plt.get_cmap("viridis").copy()
    cmap_ps.set_bad(color="black")

    if has_positive_ps:
        ps_vmin = float(np.min(positive_ps))
        ps_vmax = float(np.max(positive_ps))
        if ps_vmax <= ps_vmin:
            ps_vmax = ps_vmin * 1.0001
        ps_norm = LogNorm(vmin=ps_vmin, vmax=ps_vmax)
    else:
        ps_norm = None

    for row_index, (
        raw_slice,
        noise_slice,
        noisy_slice,
        mean_slice,
        ms_slice,
        wedge_ms_slice,
        y_label,
        row_label,
    ) in enumerate(row_data):
        ax_raw = axes[row_index, 0]
        ax_noise = axes[row_index, 1]
        ax_noisy = axes[row_index, 2]
        ax_mean = axes[row_index, 3]
        ax_ms = axes[row_index, 4]
        ax_wedge = axes[row_index, 5]
        ax_ps = axes[row_index, 6]

        im_raw = ax_raw.imshow(
            raw_slice,
            origin="lower",
            cmap="viridis",
            extent=extent_map,
            aspect="equal",
            interpolation="bicubic",
        )
        ax_raw.set_title("Raw Tb map")
        ax_raw.set_xlabel("x [cMpc/h]")
        ax_raw.set_ylabel(y_label)
        ax_raw.text(
            0.02,
            1.03,
            row_label,
            transform=ax_raw.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
        )
        cb_raw = fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.02)
        cb_raw.set_label(r"$\delta T_b$ [mK]")

        im_noise = ax_noise.imshow(
            noise_slice,
            origin="lower",
            cmap="viridis",
            extent=extent_map,
            aspect="equal",
            interpolation="bicubic",
        )
        ax_noise.set_title("Noise map")
        ax_noise.set_xlabel("x [cMpc/h]")
        ax_noise.set_ylabel(y_label)
        cb_noise = fig.colorbar(im_noise, ax=ax_noise, fraction=0.046, pad=0.02)
        cb_noise.set_label(r"$\delta T_b$ [mK]")

        im_noisy = ax_noisy.imshow(
            noisy_slice,
            origin="lower",
            cmap="viridis",
            extent=extent_map,
            aspect="equal",
            interpolation="bicubic",
        )
        ax_noisy.set_title("Noisy Tb map")
        ax_noisy.set_xlabel("x [cMpc/h]")
        ax_noisy.set_ylabel(y_label)
        cb_noisy = fig.colorbar(im_noisy, ax=ax_noisy, fraction=0.046, pad=0.02)
        cb_noisy.set_label(r"$\delta T_b$ [mK]")

        mean_span = max(1.0, 0.01 * abs(noisy_mean))
        ax_mean.imshow(
            mean_slice,
            origin="lower",
            cmap="viridis",
            extent=extent_map,
            aspect="equal",
            vmin=noisy_mean - mean_span,
            vmax=noisy_mean + mean_span,
            interpolation="bicubic",
        )
        ax_mean.set_title("Mean(noisy map)")
        ax_mean.set_xlabel("x [cMpc/h]")
        ax_mean.set_ylabel(y_label)
        ax_mean.text(
            0.5,
            0.5,
            rf"$\langle \delta T_b \rangle = {noisy_mean:.3f}\ \mathrm{{mK}}$",
            transform=ax_mean.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round", fc="black", ec="white", lw=0.8, alpha=0.7),
        )

        im_ms = ax_ms.imshow(
            ms_slice,
            origin="lower",
            cmap="viridis",
            extent=extent_map,
            aspect="equal",
            interpolation="bicubic",
        )
        ax_ms.set_title("Mean-subtracted noisy map")
        ax_ms.set_xlabel("x [cMpc/h]")
        ax_ms.set_ylabel(y_label)
        cb_ms = fig.colorbar(im_ms, ax=ax_ms, fraction=0.046, pad=0.02)
        cb_ms.set_label(r"$\delta T_b$ [mK]")

        im_wedge = ax_wedge.imshow(
            wedge_ms_slice,
            origin="lower",
            cmap="viridis",
            extent=extent_map,
            aspect="equal",
            interpolation="bicubic",
        )
        ax_wedge.set_title("Wedge-filtered MS noisy map")
        ax_wedge.set_xlabel("x [cMpc/h]")
        ax_wedge.set_ylabel(y_label)
        cb_wedge = fig.colorbar(im_wedge, ax=ax_wedge, fraction=0.046, pad=0.02)
        cb_wedge.set_label(r"$\delta T_b$ [mK]")

        if ps_norm is not None:
            im_ps = ax_ps.imshow(
                ps2d_plot,
                origin="lower",
                cmap=cmap_ps,
                extent=extent_ps,
                aspect="auto",
                norm=ps_norm,
            )
        else:
            im_ps = ax_ps.imshow(ps2d_plot, origin="lower", cmap=cmap_ps, extent=extent_ps, aspect="auto")

        kperp_line = np.linspace(extent_ps[0], extent_ps[1], 256)
        kpar_line = wedge_c * kperp_line
        line_mask = (kpar_line >= extent_ps[2]) & (kpar_line <= extent_ps[3])
        if np.any(line_mask):
            ax_ps.plot(
                kperp_line[line_mask],
                kpar_line[line_mask],
                color="red",
                linestyle=":",
                linewidth=2.0,
            )

        ax_ps.set_title("Cylindrical 2DPS (10x10)")
        ax_ps.set_xlabel(r"$k_\perp$ [h/cMpc]")
        ax_ps.set_ylabel(r"$k_\parallel$ [h/cMpc]")
        cb_ps = fig.colorbar(im_ps, ax=ax_ps, fraction=0.046, pad=0.02)
        cb_ps.set_label(r"$P(k_\parallel, k_\perp)$")

    fig.suptitle("Wedge-filtered noisy Tb pipeline -> 2DPS (script, linear bins)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    for row_index in range(2):
        add_row_annotations(fig, axes[row_index, :])

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logger = setup_logging(os.path.join(args.outdir, args.logfile))
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("WEDGE-FILTERED TARGET TB -> 2DPS (SCRIPT, LINEAR) STARTED")
    logger.info("=" * 80)
    logger.info("Raw map: %s", args.raw_map)
    logger.info("Noise map npz: %s", args.noise_map_npz)
    logger.info("Noisy map: %s", args.noisy_map)
    logger.info("Mean-subtracted noisy map: %s", args.mean_subtracted_noisy_map)
    logger.info("Wedge-filtered MS noisy map: %s", args.wedge_filtered_ms_noisy_map)
    logger.info("Output directory: %s", args.outdir)
    logger.info("Wedge slope C: %.6f", args.wedge_C)

    for path in (
        args.raw_map,
        args.noise_map_npz,
        args.noisy_map,
        args.mean_subtracted_noisy_map,
        args.wedge_filtered_ms_noisy_map,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    raw_cube = np.asarray(np.load(args.raw_map), dtype=np.float64)
    noise_cube = load_noise_cube(args.noise_map_npz)
    noisy_cube = np.asarray(np.load(args.noisy_map), dtype=np.float64)
    ms_noisy_cube = np.asarray(np.load(args.mean_subtracted_noisy_map), dtype=np.float64)
    wedge_ms_noisy_cube = np.asarray(np.load(args.wedge_filtered_ms_noisy_map), dtype=np.float64)

    validate_cube(raw_cube, "raw_map")
    validate_cube(noise_cube, "noise_map")
    validate_cube(noisy_cube, "noisy_map")
    validate_cube(ms_noisy_cube, "mean_subtracted_noisy_map")
    validate_cube(wedge_ms_noisy_cube, "wedge_filtered_ms_noisy_map")

    if not (
        raw_cube.shape
        == noise_cube.shape
        == noisy_cube.shape
        == ms_noisy_cube.shape
        == wedge_ms_noisy_cube.shape
    ):
        raise ValueError(
            "Input cubes do not share the same shape: "
            f"raw={raw_cube.shape}, noise={noise_cube.shape}, "
            f"noisy={noisy_cube.shape}, ms_noisy={ms_noisy_cube.shape}, "
            f"wedge_ms_noisy={wedge_ms_noisy_cube.shape}"
        )

    ngrid = raw_cube.shape[0]
    noisy_mean = float(np.mean(noisy_cube, dtype=np.float64))

    logger.info("Cube shape: %s", raw_cube.shape)
    logger.info("Mean(noisy map): %.8f mK", noisy_mean)
    logger.info(
        "Mean(mean-subtracted noisy map): %.8e mK",
        float(np.mean(ms_noisy_cube, dtype=np.float64)),
    )
    logger.info(
        "Mean(wedge-filtered MS noisy map): %.8e mK",
        float(np.mean(wedge_ms_noisy_cube, dtype=np.float64)),
    )

    k_par_edges, k_perp_edges, k_par_bins, k_perp_bins = setup_linear_k_bins(
        ngrid,
        args.box,
        args.nbins_par,
        args.nbins_perp,
    )
    logger.info("k_par_edges: %s", np.array2string(k_par_edges, precision=6))
    logger.info("k_perp_edges: %s", np.array2string(k_perp_edges, precision=6))

    # Compute 2DPS from wedge-filtered mean-subtracted noisy map.
    ps2d, kount = compute_2dps_script_linear(
        wedge_ms_noisy_cube,
        box=args.box,
        z_target=args.z_target,
        omega_m=args.omega_m,
        hubble=args.hubble,
        k_par_edges=k_par_edges,
        k_perp_edges=k_perp_edges,
    )
    logger.info("2DPS shape: %s", ps2d.shape)
    logger.info("Empty 2DPS bins: %d / %d", int(np.count_nonzero(kount == 0)), int(kount.size))

    prefix = args.output_prefix
    mean_noisy_path = os.path.join(args.outdir, f"{prefix}_mean_noisy_value_mK.npy")
    ps2d_path = os.path.join(args.outdir, f"{prefix}_2dps_script_linear_10x10.npy")
    k_par_edges_path = os.path.join(args.outdir, f"{prefix}_k_par_edges_linear.npy")
    k_perp_edges_path = os.path.join(args.outdir, f"{prefix}_k_perp_edges_linear.npy")
    k_par_bins_path = os.path.join(args.outdir, f"{prefix}_k_par_bins_linear.npy")
    k_perp_bins_path = os.path.join(args.outdir, f"{prefix}_k_perp_bins_linear.npy")
    kount_path = os.path.join(args.outdir, f"{prefix}_kount_linear.npy")
    plot_path = os.path.join(args.outdir, f"{prefix}_pipeline_2dps_overview.png")

    np.save(mean_noisy_path, np.array(noisy_mean, dtype=np.float64))
    np.save(ps2d_path, ps2d)
    np.save(k_par_edges_path, k_par_edges)
    np.save(k_perp_edges_path, k_perp_edges)
    np.save(k_par_bins_path, k_par_bins)
    np.save(k_perp_bins_path, k_perp_bins)
    np.save(kount_path, kount)

    make_overview_plot(
        raw_cube=raw_cube,
        noise_cube=noise_cube,
        noisy_cube=noisy_cube,
        noisy_mean=noisy_mean,
        mean_subtracted_noisy_cube=ms_noisy_cube,
        wedge_filtered_ms_noisy_cube=wedge_ms_noisy_cube,
        ps2d=ps2d,
        k_par_edges=k_par_edges,
        k_perp_edges=k_perp_edges,
        box=args.box,
        wedge_c=args.wedge_C,
        out_path=plot_path,
        dpi=args.dpi,
    )

    logger.info("Saved noisy-map mean value: %s", mean_noisy_path)
    logger.info("Saved cylindrical 2DPS: %s", ps2d_path)
    logger.info("Saved kount: %s", kount_path)
    logger.info("Saved overview figure: %s", plot_path)

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("WEDGE-FILTERED TARGET TB -> 2DPS (SCRIPT, LINEAR) COMPLETE")
    logger.info("=" * 80)
    logger.info("Runtime: %.2f s (%.2f min)", total_time, total_time / 60.0)


if __name__ == "__main__":
    main()
