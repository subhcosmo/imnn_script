#!/usr/bin/env python
# coding: utf-8

"""Compute covariance/correlation from wedge NN-subtracted PS samples.

Inputs:
- training_wedge_filtered_noisy_temp_map_ps_nnsub.npy
- validation_wedge_filtered_noisy_temp_map_ps_nnsub.npy

Outputs (in outdir):
- covariance_matrix_ps_wedge.npy
- correlation_matrix_ps_wedge.npy
- covariance_correlation_matrix_ps_wedge.png
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute covariance and correlation matrices from wedge NN-sub PS"
    )

    p.add_argument(
        "--training_nnsub",
        type=str,
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "cov_matrix/training_wedge_filtered_noisy_temp_map_ps_nnsub.npy"
        ),
        help="Training NN-sub PS file",
    )
    p.add_argument(
        "--validation_nnsub",
        type=str,
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "cov_matrix/validation_wedge_filtered_noisy_temp_map_ps_nnsub.npy"
        ),
        help="Validation NN-sub PS file",
    )
    p.add_argument(
        "--k_bins",
        type=str,
        default=(
            "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/"
            "cov_matrix/k_bins_ps_wedge.npy"
        ),
        help="k-bin centers file used for axis labeling",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/cov_matrix",
        help="Output directory",
    )

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


def load_ps_stack(path, name):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (N, nbins). Got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logger = setup_logging(os.path.join(args.outdir, "compute_covariance_from_wedge_nnsub_ps.log"))
    logger.info("=" * 80)
    logger.info("COMPUTE COVARIANCE FROM WEDGE NN-SUB POWER SPECTRA - STARTED")
    logger.info("=" * 80)

    for req in (args.training_nnsub, args.validation_nnsub, args.k_bins):
        if not os.path.exists(req):
            logger.error("Missing required file: %s", req)
            sys.exit(1)

    train_ps = load_ps_stack(args.training_nnsub, "training_nnsub")
    val_ps = load_ps_stack(args.validation_nnsub, "validation_nnsub")
    k_bins = np.load(args.k_bins).astype(np.float64)

    logger.info("Training NN-sub PS shape: %s", str(train_ps.shape))
    logger.info("Validation NN-sub PS shape: %s", str(val_ps.shape))
    logger.info("k_bins shape: %s", str(k_bins.shape))

    if train_ps.shape[1] != val_ps.shape[1]:
        logger.error(
            "Training/validation nbins mismatch: %d vs %d",
            train_ps.shape[1],
            val_ps.shape[1],
        )
        sys.exit(1)

    nbins = train_ps.shape[1]
    if k_bins.ndim != 1 or k_bins.shape[0] != nbins:
        logger.error(
            "k_bins length mismatch: expected %d, got %s",
            nbins,
            str(k_bins.shape),
        )
        sys.exit(1)

    samples = np.vstack([train_ps, val_ps])
    nsamples = samples.shape[0]

    if nsamples < 2:
        logger.error("Need at least 2 total samples, got %d", nsamples)
        sys.exit(1)

    logger.info("Total combined samples: %d", nsamples)

    # Covariance over features (k-bins), so rowvar=False.
    cov = np.cov(samples, rowvar=False, ddof=1)

    # Correlation from covariance with safe zero-std handling.
    std = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    # Keep exact 1 on valid diagonal entries.
    valid_diag = std > 0
    corr[np.diag_indices_from(corr)] = np.where(valid_diag, 1.0, 0.0)

    cov_path = os.path.join(args.outdir, "covariance_matrix_ps_wedge.npy")
    corr_path = os.path.join(args.outdir, "correlation_matrix_ps_wedge.npy")
    np.save(cov_path, cov.astype(np.float64))
    np.save(corr_path, corr.astype(np.float64))

    logger.info("Saved covariance matrix: %s", cov_path)
    logger.info("Saved correlation matrix: %s", corr_path)

    # Plot side-by-side with equal-size matrix cells and k labels on axes.
    tick_labels = [f"{k:.2f}" for k in k_bins]
    tick_pos = np.arange(nbins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    vmax_cov = np.max(np.abs(cov))
    cov_norm = Normalize(vmin=-vmax_cov, vmax=vmax_cov) if vmax_cov > 0 else None

    im0 = axes[0].imshow(
        cov,
        origin="lower",
        cmap="RdBu_r",
        norm=cov_norm,
        aspect="equal",
    )
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.9)
    cbar0.set_label("Covariance", fontsize=11)

    im1 = axes[1].imshow(
        corr,
        origin="lower",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0),
        aspect="equal",
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.9)
    cbar1.set_label("Correlation", fontsize=11)

    axes[0].set_title("Covariance Matrix", fontsize=12)
    axes[1].set_title("Correlation Matrix", fontsize=12)

    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel(r"$k\ [1/\mathrm{Mpc}]$", fontsize=11)
        ax.set_ylabel(r"$k\ [1/\mathrm{Mpc}]$", fontsize=11)

    plot_path = os.path.join(args.outdir, "covariance_correlation_matrix_ps_wedge.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved covariance/correlation plot: %s", plot_path)
    logger.info("=" * 80)
    logger.info("COMPUTE COVARIANCE FROM WEDGE NN-SUB POWER SPECTRA - COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

