#!/usr/bin/env python
# coding: utf-8

"""Compute PS and NN-subtracted PS for wedge-filtered noisy temperature maps.

Tasks:
1) Load training/validation wedge-filtered noisy maps.
2) Compute dimensionless power spectra Delta^2(k) for each map.
3) Save training/validation PS arrays separately.
4) Plot all (training+validation) PS curves on:
   - log-log
   - linear-linear
   - x linear, y log
5) Subtract fixed mean wedge-noise PS from each map PS.
6) Save training/validation NN-sub PS arrays separately.
7) Plot all (training+validation) NN-sub PS curves on the same 3 scale styles.
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import script


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute PS and NN-sub PS for wedge-filtered training/validation maps"
    )

    p.add_argument(
        "--project_dir",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge",
        help="Project directory",
    )
    p.add_argument(
        "--training_maps",
        type=str,
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "maps_wedge_filtered/training_wedge_filtered_main.npy"
        ),
        help="Input training wedge-filtered maps (.npy)",
    )
    p.add_argument(
        "--validation_maps",
        type=str,
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "maps_wedge_filtered/validation_wedge_filtered_main.npy"
        ),
        help="Input validation wedge-filtered maps (.npy)",
    )
    p.add_argument(
        "--noise_ps_mean",
        type=str,
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "noise_maps/PS_noise_maps_wedge_filtered_mean.npy"
        ),
        help="Fixed mean wedge-noise power spectrum (.npy)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/cov_matrix",
        help="Output directory for PS arrays and plots",
    )

    p.add_argument("--nbins", type=int, default=10, help="Number of k bins")
    p.add_argument("--box", type=float, default=128.0, help="Box size (Mpc)")
    p.add_argument("--z_target", type=float, default=7.0, help="Target redshift")
    p.add_argument("--omega_m", type=float, default=0.308, help="Omega_m")
    p.add_argument("--hubble", type=float, default=0.678, help="h")

    return p.parse_args()


def setup_logging(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logpath, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def build_k_edges_kbins(ngrid, box, nbins, log_bins=True):
    kmin = 2.0 * np.pi / box
    kmax = np.pi * ngrid / box

    if log_bins:
        lnk_edges = np.linspace(np.log(kmin), np.log(kmax), num=nbins + 1, endpoint=True)
        lnk_bins = 0.5 * (lnk_edges[:-1] + lnk_edges[1:])
        k_edges = np.exp(lnk_edges)
        k_bins = np.exp(lnk_bins)
    else:
        k_edges = np.linspace(kmin, kmax, num=nbins + 1, endpoint=True)
        k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

    return k_edges.astype(np.float64), k_bins.astype(np.float64)


def ensure_map_stack(arr, name):
    if arr.ndim == 3:
        return arr[np.newaxis, ...]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"{name} must be 3D or 4D, got shape {arr.shape}")


def load_maps_any_format(path, name, logger):
    """Load maps from either numeric npy stack or dict-style object npy."""
    try:
        arr = np.load(path)
        arr = ensure_map_stack(np.asarray(arr), name)
        logger.info("Loaded %s as numeric ndarray from %s", name, path)
        return arr.astype(np.float32, copy=False)
    except ValueError as exc:
        # Common case for np.save(dict): object array requiring allow_pickle=True.
        if "Object arrays cannot be loaded" not in str(exc):
            raise

    arr_obj = np.load(path, allow_pickle=True)

    # Case: scalar object containing dict saved via np.save(dict_obj)
    if isinstance(arr_obj, np.ndarray) and arr_obj.dtype == object and arr_obj.shape == ():
        payload = arr_obj.item()
        if isinstance(payload, dict):
            if len(payload) == 0:
                raise ValueError(f"{name} dictionary in {path} is empty")

            # Stable ordering by seed/key when possible.
            try:
                keys = sorted(payload.keys())
            except TypeError:
                keys = list(payload.keys())

            maps = [np.asarray(payload[k], dtype=np.float32) for k in keys]
            stack = np.stack(maps, axis=0)
            stack = ensure_map_stack(stack, name)
            logger.info(
                "Loaded %s as dict-object npy from %s (%d maps)",
                name,
                path,
                stack.shape[0],
            )
            return stack

    # Case: object array/list-like payload that can be stacked directly.
    if isinstance(arr_obj, np.ndarray) and arr_obj.dtype == object:
        maps = [np.asarray(x, dtype=np.float32) for x in arr_obj]
        stack = np.stack(maps, axis=0)
        stack = ensure_map_stack(stack, name)
        logger.info(
            "Loaded %s as object-array npy from %s (%d maps)",
            name,
            path,
            stack.shape[0],
        )
        return stack

    # Fallback: treat as numeric ndarray.
    stack = ensure_map_stack(np.asarray(arr_obj), name)
    logger.info("Loaded %s from %s using pickle-enabled fallback", name, path)
    return stack.astype(np.float32, copy=False)


def compute_delta2_for_maps(maps_4d, ps_helper, k_edges, k_bins, desc):
    n_maps = maps_4d.shape[0]
    nbins = len(k_bins)
    out = np.zeros((n_maps, nbins), dtype=np.float32)

    for i in tqdm(range(n_maps), desc=desc):
        m = maps_4d[i].astype(np.float32, copy=False)
        powspec_binned, kount = ps_helper.get_binned_powspec(
            m,
            k_edges,
            convolve=False,
            units="",
            bin_weighted=False,
        )

        delta2 = np.zeros(nbins, dtype=np.float64)
        valid = kount > 0
        delta2[valid] = (k_bins[valid] ** 3) * powspec_binned[valid] / (2.0 * np.pi ** 2)
        out[i] = delta2.astype(np.float32)

    return out


def plot_ps_bundle(k_bins, ps_stack, out_png, scale_mode, title):
    fig, ax = plt.subplots(figsize=(10, 7))

    if scale_mode == "loglog":
        y_stack = np.where(ps_stack > 0.0, ps_stack, np.nan)
        for i in range(y_stack.shape[0]):
            ax.loglog(k_bins, y_stack[i], color="C0", alpha=0.05, linewidth=0.6)
    elif scale_mode == "semilogy":
        y_stack = np.where(ps_stack > 0.0, ps_stack, np.nan)
        for i in range(y_stack.shape[0]):
            ax.semilogy(k_bins, y_stack[i], color="C0", alpha=0.05, linewidth=0.6)
    elif scale_mode == "linear":
        y_stack = ps_stack
        for i in range(y_stack.shape[0]):
            ax.plot(k_bins, y_stack[i], color="C0", alpha=0.05, linewidth=0.6)
    else:
        raise ValueError(f"Unknown scale mode: {scale_mode}")

    p16 = np.percentile(ps_stack, 16.0, axis=0)
    p50 = np.percentile(ps_stack, 50.0, axis=0)
    p84 = np.percentile(ps_stack, 84.0, axis=0)

    if scale_mode in ("loglog", "semilogy"):
        p16_plot = np.where(p16 > 0.0, p16, np.nan)
        p50_plot = np.where(p50 > 0.0, p50, np.nan)
        p84_plot = np.where(p84 > 0.0, p84, np.nan)
    else:
        p16_plot = p16
        p50_plot = p50
        p84_plot = p84

    ax.plot(k_bins, p50_plot, color="k", linewidth=2.0, label="Median")
    ax.fill_between(k_bins, p16_plot, p84_plot, color="gray", alpha=0.25, label="16-84%")

    ax.set_xlabel("k [1/Mpc]", fontsize=13)
    ax.set_ylabel(r"$\Delta^2(k)$", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logger = setup_logging(os.path.join(args.outdir, "compute_wedge_filtered_map_ps.log"))
    logger.info("=" * 80)
    logger.info("COMPUTE WEDGE-FILTERED MAP POWER SPECTRA - STARTED")
    logger.info("=" * 80)

    required = [args.training_maps, args.validation_maps, args.noise_ps_mean]
    for path in required:
        if not os.path.exists(path):
            logger.error("Missing required file: %s", path)
            sys.exit(1)

    training_maps = load_maps_any_format(args.training_maps, "training_maps", logger)
    validation_maps = load_maps_any_format(args.validation_maps, "validation_maps", logger)

    logger.info("Training maps shape: %s", str(training_maps.shape))
    logger.info("Validation maps shape: %s", str(validation_maps.shape))

    if training_maps.shape[1:] != validation_maps.shape[1:]:
        logger.error(
            "Training/validation spatial shape mismatch: %s vs %s",
            str(training_maps.shape[1:]),
            str(validation_maps.shape[1:]),
        )
        sys.exit(1)

    ngrid = training_maps.shape[-1]
    if training_maps.shape[-3:] != (ngrid, ngrid, ngrid):
        logger.error(
            "Expected cubic maps (N, ngrid, ngrid, ngrid), got %s",
            str(training_maps.shape),
        )
        sys.exit(1)

    k_edges, k_bins = build_k_edges_kbins(ngrid, args.box, args.nbins, log_bins=True)
    np.save(os.path.join(args.outdir, "k_bins_ps_wedge.npy"), k_bins.astype(np.float32))
    np.save(os.path.join(args.outdir, "k_edges_ps_wedge.npy"), k_edges.astype(np.float32))

    dummy_density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    ps_helper = script.ionization_map(
        matter_fields=None,
        densitycontr_arr=dummy_density,
        box=args.box,
        z=args.z_target,
        omega_m=args.omega_m,
        h=args.hubble,
    )

    logger.info("Computing training PS...")
    train_ps = compute_delta2_for_maps(
        training_maps,
        ps_helper,
        k_edges,
        k_bins,
        desc="Training PS",
    )

    logger.info("Computing validation PS...")
    val_ps = compute_delta2_for_maps(
        validation_maps,
        ps_helper,
        k_edges,
        k_bins,
        desc="Validation PS",
    )

    train_ps_path = os.path.join(args.outdir, "training_wedge_filtered_noisy_temp_map_ps.npy")
    val_ps_path = os.path.join(args.outdir, "validation_wedge_filtered_noisy_temp_map_ps.npy")
    np.save(train_ps_path, train_ps)
    np.save(val_ps_path, val_ps)
    logger.info("Saved training PS: %s", train_ps_path)
    logger.info("Saved validation PS: %s", val_ps_path)

    combined_ps = np.vstack([train_ps, val_ps])

    plot_ps_bundle(
        k_bins,
        combined_ps,
        os.path.join(args.outdir, "wedge_filtered_noisy_temp_map_ps_loglog.png"),
        scale_mode="loglog",
        title="Wedge-filtered noisy map PS (train+val): log-log",
    )
    plot_ps_bundle(
        k_bins,
        combined_ps,
        os.path.join(args.outdir, "wedge_filtered_noisy_temp_map_ps_linear.png"),
        scale_mode="linear",
        title="Wedge-filtered noisy map PS (train+val): linear-linear",
    )
    plot_ps_bundle(
        k_bins,
        combined_ps,
        os.path.join(args.outdir, "wedge_filtered_noisy_temp_map_ps_ylog_xlinear.png"),
        scale_mode="semilogy",
        title="Wedge-filtered noisy map PS (train+val): y-log, x-linear",
    )
    logger.info("Saved 3 plots for raw PS (train+val).")

    noise_ps_mean = np.load(args.noise_ps_mean).astype(np.float32)
    if noise_ps_mean.ndim != 1 or noise_ps_mean.shape[0] != args.nbins:
        logger.error(
            "Mean noise PS shape mismatch: expected (%d,), got %s",
            args.nbins,
            str(noise_ps_mean.shape),
        )
        sys.exit(1)

    train_ps_nnsub = train_ps - noise_ps_mean[None, :]
    val_ps_nnsub = val_ps - noise_ps_mean[None, :]

    train_ps_nnsub_path = os.path.join(
        args.outdir,
        "training_wedge_filtered_noisy_temp_map_ps_nnsub.npy",
    )
    val_ps_nnsub_path = os.path.join(
        args.outdir,
        "validation_wedge_filtered_noisy_temp_map_ps_nnsub.npy",
    )
    np.save(train_ps_nnsub_path, train_ps_nnsub.astype(np.float32))
    np.save(val_ps_nnsub_path, val_ps_nnsub.astype(np.float32))

    logger.info("Saved training NN-sub PS: %s", train_ps_nnsub_path)
    logger.info("Saved validation NN-sub PS: %s", val_ps_nnsub_path)

    combined_ps_nnsub = np.vstack([train_ps_nnsub, val_ps_nnsub])

    plot_ps_bundle(
        k_bins,
        combined_ps_nnsub,
        os.path.join(args.outdir, "wedge_filtered_noisy_temp_map_ps_nnsub_loglog.png"),
        scale_mode="loglog",
        title="Wedge-filtered noisy map PS NN-sub (train+val): log-log",
    )
    plot_ps_bundle(
        k_bins,
        combined_ps_nnsub,
        os.path.join(args.outdir, "wedge_filtered_noisy_temp_map_ps_nnsub_linear.png"),
        scale_mode="linear",
        title="Wedge-filtered noisy map PS NN-sub (train+val): linear-linear",
    )
    plot_ps_bundle(
        k_bins,
        combined_ps_nnsub,
        os.path.join(args.outdir, "wedge_filtered_noisy_temp_map_ps_nnsub_ylog_xlinear.png"),
        scale_mode="semilogy",
        title="Wedge-filtered noisy map PS NN-sub (train+val): y-log, x-linear",
    )
    logger.info("Saved 3 plots for NN-sub PS (train+val).")

    logger.info("=" * 80)
    logger.info("COMPUTE WEDGE-FILTERED MAP POWER SPECTRA - COMPLETED SUCCESSFULLY")
    logger.info("Output directory: %s", args.outdir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

