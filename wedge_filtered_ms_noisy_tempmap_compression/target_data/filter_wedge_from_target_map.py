#!/usr/bin/env python
# coding: utf-8

"""
Apply foreground wedge removal to one mean-subtracted noisy TARGET map.

Workflow:
  1. Load target real-space map T(x)
  2. FFT -> T(k)
  3. Zero out modes inside the wedge: k_par <= C * k_perp
  4. Plot the wedge-filtered cylindrical Fourier amplitude
  5. Inverse FFT -> T_filtered(x)
  6. Save the filtered map and two real-space comparison plots
"""

import argparse
import logging
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.integrate import quad


DEFAULT_INPUT_MAP = (
    "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/target_data/"
    "target_tempmap_w_noise_ms_seed1259935638.npy"
)
DEFAULT_OUTDIR = (
    "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/target_data"
)


def setup_logging(logpath):
    """Set up logging to both file and stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(logpath, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def compute_wedge_slope(z, omega_m, omega_l):
    """Compute C = x(z) * H(z) / (c * (1 + z))."""

    def efunc(zp):
        return np.sqrt(omega_m * (1.0 + zp) ** 3 + omega_l)

    integral_z, _ = quad(lambda zp: 1.0 / efunc(zp), 0.0, z)
    return integral_z * efunc(z) / (1.0 + z)


def get_3d_k_grids(ngrid, box):
    """Generate 3D k_parallel and k_perp grids for np.fft.fftn output ordering."""
    k1d = np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")

    k_par = np.abs(kz)
    k_perp = np.sqrt(kx ** 2 + ky ** 2)
    return k_par, k_perp


def setup_k_bins(ngrid, box, nbins_par, nbins_perp):
    """Set up linear cylindrical bin edges."""
    kmin = 2.0 * np.pi / box
    kmax = np.pi * ngrid / box

    k_par_edges = np.linspace(kmin, kmax, nbins_par + 1)
    k_perp_edges = np.linspace(kmin, kmax, nbins_perp + 1)
    return k_par_edges, k_perp_edges


def bin_cylindrical_stat(field_3d, k_par, k_perp, k_par_edges, k_perp_edges):
    """Bin a 3D scalar field into cylindrical (k_par, k_perp) bins by mean value."""
    nbins_par = len(k_par_edges) - 1
    nbins_perp = len(k_perp_edges) - 1
    binned = np.zeros((nbins_par, nbins_perp), dtype=np.float64)

    for j in range(nbins_perp):
        mask_perp = (k_perp >= k_perp_edges[j]) & (k_perp < k_perp_edges[j + 1])
        for i in range(nbins_par):
            mask_par = (k_par >= k_par_edges[i]) & (k_par < k_par_edges[i + 1])
            mask = mask_perp & mask_par
            count = np.count_nonzero(mask)
            binned[i, j] = np.mean(field_3d[mask]) if count > 0 else 0.0

    return binned


def infer_seed(input_map, seed):
    """Infer the seed from the filename if not supplied explicitly."""
    if seed is not None:
        return int(seed)

    match = re.search(r"_seed(\d+)", os.path.basename(input_map))
    if match is None:
        raise ValueError(
            "Could not infer seed from input filename. Please supply --seed explicitly."
        )
    return int(match.group(1))


def get_slice_index(idx_arg, ngrid):
    """Use requested slice index or the midpoint when idx_arg < 0."""
    idx = ngrid // 2 if idx_arg < 0 else int(idx_arg)
    if idx < 0 or idx >= ngrid:
        raise ValueError(f"Slice index {idx} is outside valid range [0, {ngrid - 1}]")
    return idx


def slice_xy(map3d, z_idx):
    """Return an xy slice at fixed z, transposed for imshow."""
    return map3d[:, :, z_idx].T


def slice_xz(map3d, y_idx):
    """Return an xz slice at fixed y, transposed for imshow."""
    return map3d[:, y_idx, :].T


def plot_fft_cylindrical(amp2d, k_par_edges, k_perp_edges, C, outdir, seed):
    """Plot cylindrical wedge-filtered Fourier amplitude."""
    positive = amp2d[amp2d > 0]
    if positive.size > 0:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        if vmin >= vmax:
            vmax = vmin * 1.001
    else:
        vmin, vmax = 1e-12, 1.0

    amp_plot = amp2d.astype(float).copy()
    amp_plot[amp_plot <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(6.3, 5.2), constrained_layout=True)

    im = ax.pcolormesh(
        k_perp_edges,
        k_par_edges,
        amp_plot,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
        shading="flat",
    )

    kperp_line = np.linspace(k_perp_edges[0], k_perp_edges[-1], 400)
    ax.plot(
        kperp_line,
        C * kperp_line,
        "w--",
        linewidth=1.2,
        label=f"Wedge (C={C:.3f})",
    )

    ax.set_xlabel(r"$k_\perp\ [h/\mathrm{cMpc}]$", fontsize=10)
    ax.set_ylabel(r"$k_\parallel\ [h/\mathrm{cMpc}]$", fontsize=10)
    ax.set_xlim(k_perp_edges[0], k_perp_edges[-1])
    ax.set_ylim(k_par_edges[0], k_par_edges[-1])
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Target seed {seed}", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        r"$|\tilde{\delta T_b}(k_\parallel, k_\perp)|\ [\mathrm{mK}\ (\mathrm{cMpc}/h)^{3/2}]$",
        fontsize=10,
    )
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Wedge-Removed Cylindrical Fourier Amplitude", fontsize=12, weight="bold")

    outfile = os.path.join(outdir, f"target_fft_wedge_removed_seed{seed}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_map_comparison(noisy_map, filtered_map, plane, slice_idx, box, outdir, seed):
    """Plot noisy vs wedge-filtered map slices for one plane."""
    if plane == "xy":
        noisy_slice = slice_xy(noisy_map, slice_idx)
        filtered_slice = slice_xy(filtered_map, slice_idx)
        vertical_label = "y"
        plane_label = "x-y"
        slice_label = f"z={slice_idx}"
        outfile = os.path.join(
            outdir,
            f"target_noisy_vs_wedge_filtered_xy_seed{seed}_z{slice_idx}.png",
        )
    elif plane == "xz":
        noisy_slice = slice_xz(noisy_map, slice_idx)
        filtered_slice = slice_xz(filtered_map, slice_idx)
        vertical_label = "z"
        plane_label = "x-z"
        slice_label = f"y={slice_idx}"
        outfile = os.path.join(
            outdir,
            f"target_noisy_vs_wedge_filtered_xz_seed{seed}_y{slice_idx}.png",
        )
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)

    noisy_vmin = float(np.min(noisy_slice))
    noisy_vmax = float(np.max(noisy_slice))
    if noisy_vmin == noisy_vmax:
        noisy_vmin -= 1e-6
        noisy_vmax += 1e-6

    filtered_vmin = float(np.min(filtered_slice))
    filtered_vmax = float(np.max(filtered_slice))
    if filtered_vmin == filtered_vmax:
        filtered_vmin -= 1e-6
        filtered_vmax += 1e-6

    im_left = axes[0].imshow(
        noisy_slice,
        origin="lower",
        extent=[0.0, box, 0.0, box],
        cmap="viridis",
        interpolation="bicubic",
        aspect="auto",
        vmin=noisy_vmin,
        vmax=noisy_vmax,
    )
    axes[0].set_title("Mean-subtracted noisy", fontsize=11)

    im_right = axes[1].imshow(
        filtered_slice,
        origin="lower",
        extent=[0.0, box, 0.0, box],
        cmap="viridis",
        interpolation="bicubic",
        aspect="auto",
        vmin=filtered_vmin,
        vmax=filtered_vmax,
    )
    axes[1].set_title("Wedge filtered", fontsize=11)

    for ax in axes:
        ax.set_xlabel(r"$x\ [\mathrm{cMpc}/h]$", fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(rf"${vertical_label}\ [\mathrm{{cMpc}}/h]$", fontsize=10)

    cbar_left = fig.colorbar(im_left, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_left.set_label(r"$\delta T_b\ [\mathrm{mK}]$", fontsize=10)
    cbar_left.ax.tick_params(labelsize=8)

    cbar_right = fig.colorbar(im_right, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_right.set_label(r"$\delta T_b\ [\mathrm{mK}]$", fontsize=10)
    cbar_right.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Target seed {seed}: {plane_label} slice at {slice_label}",
        fontsize=12,
        weight="bold",
    )

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply wedge filtering to a mean-subtracted noisy TARGET map."
    )
    parser.add_argument(
        "--input_map",
        default=DEFAULT_INPUT_MAP,
        help="Path to the mean-subtracted noisy target map (.npy).",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory where filtered target data and plots will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Target seed. If omitted, it is inferred from the input filename.",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=128.0,
        help="Simulation box size in cMpc/h.",
    )
    parser.add_argument(
        "--ngrid",
        type=int,
        default=64,
        help="Expected grid size of the cube.",
    )
    parser.add_argument(
        "--nbins_par",
        type=int,
        default=64,
        help="Number of k_parallel bins for the cylindrical FFT plot.",
    )
    parser.add_argument(
        "--nbins_perp",
        type=int,
        default=64,
        help="Number of k_perp bins for the cylindrical FFT plot.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=7.0,
        help="Redshift used to compute the wedge slope when --C is not supplied.",
    )
    parser.add_argument(
        "--omega_m",
        type=float,
        default=0.308,
        help="Matter density parameter.",
    )
    parser.add_argument(
        "--omega_l",
        type=float,
        default=0.692,
        help="Dark-energy density parameter.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=None,
        help="Explicit wedge slope. If supplied, this overrides the redshift-based value.",
    )
    parser.add_argument(
        "--z_index",
        type=int,
        default=-1,
        help="z index for the xy slice plot. Use midpoint if < 0.",
    )
    parser.add_argument(
        "--y_index",
        type=int,
        default=-1,
        help="y index for the xz slice plot. Use midpoint if < 0.",
    )
    parser.add_argument(
        "--logfile",
        default="filter_wedge_from_target_map.log",
        help="Log filename written inside outdir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logpath = os.path.join(args.outdir, args.logfile)
    logger = setup_logging(logpath)

    logger.info("=" * 72)
    logger.info("Wedge Filtering for Mean-Subtracted Noisy TARGET Map")
    logger.info("=" * 72)
    logger.info("input_map: %s", args.input_map)
    logger.info("outdir: %s", args.outdir)
    logger.info("box: %.3f cMpc/h", args.box)
    logger.info("ngrid: %d", args.ngrid)
    logger.info("nbins_par: %d | nbins_perp: %d", args.nbins_par, args.nbins_perp)

    if not os.path.exists(args.input_map):
        raise FileNotFoundError(f"Input map not found: {args.input_map}")

    seed = infer_seed(args.input_map, args.seed)
    logger.info("seed: %d", seed)

    noisy_map = np.load(args.input_map)
    logger.info(
        "Loaded noisy map: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6e std=%.6f",
        noisy_map.shape,
        noisy_map.dtype,
        float(np.min(noisy_map)),
        float(np.max(noisy_map)),
        float(np.mean(noisy_map)),
        float(np.std(noisy_map)),
    )

    if noisy_map.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {noisy_map.shape}")
    if noisy_map.shape[0] != noisy_map.shape[1] or noisy_map.shape[1] != noisy_map.shape[2]:
        raise ValueError(f"Expected a cubic grid, got shape {noisy_map.shape}")
    if noisy_map.shape != (args.ngrid, args.ngrid, args.ngrid):
        raise ValueError(
            f"Input map has shape {noisy_map.shape}, expected {(args.ngrid, args.ngrid, args.ngrid)}"
        )

    z_idx = get_slice_index(args.z_index, args.ngrid)
    y_idx = get_slice_index(args.y_index, args.ngrid)
    logger.info("xy plot will use z index: %d", z_idx)
    logger.info("xz plot will use y index: %d", y_idx)

    if args.C is not None:
        C = float(args.C)
        logger.info("Using user-supplied wedge slope C = %.6f", C)
    else:
        C = compute_wedge_slope(args.z, args.omega_m, args.omega_l)
        logger.info(
            "Computed wedge slope from z=%.3f, omega_m=%.3f, omega_l=%.3f: C = %.6f",
            args.z,
            args.omega_m,
            args.omega_l,
            C,
        )

    k_par, k_perp = get_3d_k_grids(args.ngrid, args.box)
    k_par_edges, k_perp_edges = setup_k_bins(
        args.ngrid,
        args.box,
        args.nbins_par,
        args.nbins_perp,
    )

    mask = k_par > C * k_perp
    kept_fraction = np.count_nonzero(mask) / mask.size
    logger.info("Wedge mask created. Fraction of modes kept: %.2f%%", 100.0 * kept_fraction)

    ft_before = np.fft.fftn(noisy_map)
    ft_filtered = ft_before.copy()
    ft_filtered[~mask] = 0.0

    deltak = ft_filtered * (np.sqrt(args.box ** 3) / args.ngrid ** 3)
    amp3d = np.abs(deltak)
    amp2d = bin_cylindrical_stat(amp3d, k_par, k_perp, k_par_edges, k_perp_edges)

    filtered_map = np.fft.ifftn(ft_filtered).real.astype(np.float32)
    logger.info(
        "Filtered map stats: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6e std=%.6f",
        filtered_map.shape,
        filtered_map.dtype,
        float(np.min(filtered_map)),
        float(np.max(filtered_map)),
        float(np.mean(filtered_map)),
        float(np.std(filtered_map)),
    )

    filtered_map_path = os.path.join(
        args.outdir,
        f"target_tempmap_w_noise_ms_wedge_filtered_seed{seed}.npy",
    )
    np.save(filtered_map_path, filtered_map)
    logger.info("Saved filtered target map: %s", filtered_map_path)

    fft_plot_path = plot_fft_cylindrical(
        amp2d=amp2d,
        k_par_edges=k_par_edges,
        k_perp_edges=k_perp_edges,
        C=C,
        outdir=args.outdir,
        seed=seed,
    )
    logger.info("Saved FFT plot: %s", fft_plot_path)

    xy_plot_path = plot_map_comparison(
        noisy_map=noisy_map,
        filtered_map=filtered_map,
        plane="xy",
        slice_idx=z_idx,
        box=args.box,
        outdir=args.outdir,
        seed=seed,
    )
    logger.info("Saved x-y comparison plot: %s", xy_plot_path)

    xz_plot_path = plot_map_comparison(
        noisy_map=noisy_map,
        filtered_map=filtered_map,
        plane="xz",
        slice_idx=y_idx,
        box=args.box,
        outdir=args.outdir,
        seed=seed,
    )
    logger.info("Saved x-z comparison plot: %s", xz_plot_path)

    logger.info("=" * 72)
    logger.info("Wedge filtering complete")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

