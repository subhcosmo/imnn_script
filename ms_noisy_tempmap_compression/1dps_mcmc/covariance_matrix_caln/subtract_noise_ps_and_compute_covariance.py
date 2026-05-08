#!/usr/bin/env python
# coding: utf-8

"""Subtract mean noise power spectrum from signal+noise power spectra and compute covariance.

This script:
1. Loads PS_noise_maps_mean.npy (mean noise power spectrum)
2. Loads training_PS.npy and validation_PS.npy (signal+noise power spectra)
3. Subtracts mean noise PS from each power spectrum
4. Saves as training_PS_nnsub.npy and validation_PS_nnsub.npy
5. Plots all noise-subtracted power spectra
6. Computes and saves covariance matrix from the combined dataset
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import logging
import sys
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==============================================================
# ARGUMENTS
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Subtract Noise PS and Compute Covariance")
    p.add_argument(
        "--training_ps_file",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/training_PS.npy",
        help="Path to training_PS.npy"
    )
    p.add_argument(
        "--validation_ps_file",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/validation_PS.npy",
        help="Path to validation_PS.npy"
    )
    p.add_argument(
        "--noise_ps_mean_file",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/noise_maps/power_spectra/PS_noise_maps_mean.npy",
        help="Path to PS_noise_maps_mean.npy"
    )
    p.add_argument(
        "--k_bins_file",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/noise_maps/power_spectra/k_bins_noise_ps.npy",
        help="Path to k_bins file"
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/power_spectra_nn_sub",
        help="Output directory"
    )
    p.add_argument(
        "--logfile",
        default="subtract_noise_ps_and_compute_covariance.log",
        help="Log filename"
    )
    return p.parse_args()


# ==============================================================
# LOGGING
# ==============================================================

def setup_logging(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logpath, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ==============================================================
# MAIN
# ==============================================================

def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    logger = setup_logging(os.path.join(args.outdir, args.logfile))

    logger.info("=" * 80)
    logger.info("SUBTRACT NOISE PS AND COMPUTE COVARIANCE")
    logger.info("=" * 80)

    # ==============================================================
    # LOAD INPUT FILES
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("LOADING INPUT FILES")
    logger.info("=" * 60)

    # Check files exist
    for fpath, fname in [
        (args.training_ps_file, "Training PS"),
        (args.validation_ps_file, "Validation PS"),
        (args.noise_ps_mean_file, "Noise PS Mean"),
        (args.k_bins_file, "k-bins"),
    ]:
        if not os.path.exists(fpath):
            logger.error("%s file not found: %s", fname, fpath)
            sys.exit(1)
        logger.info("Found %s: %s", fname, fpath)

    # Load training PS
    training_ps_raw = np.load(args.training_ps_file, allow_pickle=True)
    training_ps_dict = training_ps_raw.item() if isinstance(training_ps_raw, np.ndarray) and training_ps_raw.shape == () else training_ps_raw
    training_seeds = list(training_ps_dict.keys())
    logger.info("Loaded training PS: %d seeds", len(training_seeds))

    # Load validation PS
    validation_ps_raw = np.load(args.validation_ps_file, allow_pickle=True)
    validation_ps_dict = validation_ps_raw.item() if isinstance(validation_ps_raw, np.ndarray) and validation_ps_raw.shape == () else validation_ps_raw
    validation_seeds = list(validation_ps_dict.keys())
    logger.info("Loaded validation PS: %d seeds", len(validation_seeds))

    # Load mean noise PS
    noise_ps_mean = np.load(args.noise_ps_mean_file)
    logger.info("Loaded mean noise PS: shape=%s", noise_ps_mean.shape)
    logger.info("Mean noise PS values: %s", noise_ps_mean)

    # Load k-bins
    k_bins = np.load(args.k_bins_file)
    logger.info("Loaded k-bins: %s", k_bins)

    # Verify shapes match
    sample_ps = training_ps_dict[training_seeds[0]]
    if sample_ps.shape != noise_ps_mean.shape:
        logger.error("Shape mismatch! Training PS shape: %s, Noise PS mean shape: %s", 
                     sample_ps.shape, noise_ps_mean.shape)
        sys.exit(1)
    logger.info("Shape verification passed: %s", sample_ps.shape)

    # ==============================================================
    # SUBTRACT MEAN NOISE PS
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUBTRACTING MEAN NOISE POWER SPECTRUM")
    logger.info("=" * 60)

    # Subtract from training PS
    training_ps_nnsub_dict = {}
    for seed in tqdm(training_seeds, desc="Training PS noise subtraction", ncols=100):
        ps_original = training_ps_dict[seed]
        ps_nnsub = ps_original - noise_ps_mean
        training_ps_nnsub_dict[seed] = ps_nnsub

    logger.info("Noise-subtracted %d training power spectra", len(training_ps_nnsub_dict))

    # Subtract from validation PS
    validation_ps_nnsub_dict = {}
    for seed in tqdm(validation_seeds, desc="Validation PS noise subtraction", ncols=100):
        ps_original = validation_ps_dict[seed]
        ps_nnsub = ps_original - noise_ps_mean
        validation_ps_nnsub_dict[seed] = ps_nnsub

    logger.info("Noise-subtracted %d validation power spectra", len(validation_ps_nnsub_dict))

    # ==============================================================
    # SAVE NOISE-SUBTRACTED PS
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING NOISE-SUBTRACTED POWER SPECTRA")
    logger.info("=" * 60)

    training_nnsub_file = os.path.join(args.outdir, "training_PS_nnsub.npy")
    np.save(training_nnsub_file, training_ps_nnsub_dict)
    logger.info("Saved training noise-subtracted PS to: %s", training_nnsub_file)

    validation_nnsub_file = os.path.join(args.outdir, "validation_PS_nnsub.npy")
    np.save(validation_nnsub_file, validation_ps_nnsub_dict)
    logger.info("Saved validation noise-subtracted PS to: %s", validation_nnsub_file)

    # Save k-bins for reference
    k_bins_file = os.path.join(args.outdir, "k_bins_ps.npy")
    np.save(k_bins_file, k_bins)
    logger.info("Saved k-bins to: %s", k_bins_file)

    # ==============================================================
    # CONVERT TO ARRAYS FOR STATISTICS AND PLOTTING
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPUTING STATISTICS")
    logger.info("=" * 60)

    # Convert to arrays
    training_arr = np.array([training_ps_nnsub_dict[s] for s in sorted(training_seeds)], dtype=np.float64)
    validation_arr = np.array([validation_ps_nnsub_dict[s] for s in sorted(validation_seeds)], dtype=np.float64)

    logger.info("Training array shape: %s", training_arr.shape)
    logger.info("Validation array shape: %s", validation_arr.shape)

    # Combine for covariance
    combined = np.vstack((training_arr, validation_arr))
    logger.info("Combined dataset shape: %s", combined.shape)

    # Compute statistics
    mean_ps = np.mean(combined, axis=0)
    std_ps = np.std(combined, axis=0)

    logger.info("Mean noise-subtracted PS: %s", mean_ps)
    logger.info("Std noise-subtracted PS: %s", std_ps)

    # Save mean PS
    mean_ps_file = os.path.join(args.outdir, "mean_power_spectrum_nnsub.npy")
    np.save(mean_ps_file, mean_ps)
    logger.info("Saved mean PS to: %s", mean_ps_file)

    # ==============================================================
    # PLOTTING - ALL POWER SPECTRA
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 60)

    # --- Plot 1: All power spectra with mean ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all realizations in black
    for row in combined:
        ax.loglog(k_bins, row, color="black", alpha=0.05, lw=0.5)

    # Plot mean in red
    ax.loglog(k_bins, mean_ps, color="red", lw=2.0, label="Mean")

    ax.set_xlabel(r"$k\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=12)
    ax.set_ylabel(r"$\Delta^2_{21}(k)\ [\mathrm{mK}^2]$", fontsize=12)
    ax.set_title("Noise-Subtracted Power Spectra (Training + Validation)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plot_file = os.path.join(args.outdir, "power_spectra_nnsub_all.png")
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved all PS plot to: %s", plot_file)

    # --- Plot 2: Mean with 1-sigma shading ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(k_bins, mean_ps, 'b-', lw=2, label='Mean')
    ax.fill_between(k_bins, mean_ps - std_ps, mean_ps + std_ps, alpha=0.3, color='blue', label='±1σ')

    ax.set_xlabel(r"$k\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=12)
    ax.set_ylabel(r"$\Delta^2_{21}(k)\ [\mathrm{mK}^2]$", fontsize=12)
    ax.set_title("Mean Noise-Subtracted Power Spectrum with 1σ Band", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    mean_plot_file = os.path.join(args.outdir, "power_spectrum_nnsub_mean.png")
    fig.savefig(mean_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved mean PS plot to: %s", mean_plot_file)

    # --- Plot 3: Training vs Validation comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training in blue
    for row in training_arr:
        ax.loglog(k_bins, row, color="blue", alpha=0.05, lw=0.5)

    # Plot validation in red
    for row in validation_arr:
        ax.loglog(k_bins, row, color="red", alpha=0.05, lw=0.5)

    # Plot means
    training_mean = np.mean(training_arr, axis=0)
    validation_mean = np.mean(validation_arr, axis=0)
    ax.loglog(k_bins, training_mean, 'b-', lw=2, label=f'Training Mean (N={len(training_seeds)})')
    ax.loglog(k_bins, validation_mean, 'r-', lw=2, label=f'Validation Mean (N={len(validation_seeds)})')

    ax.set_xlabel(r"$k\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=12)
    ax.set_ylabel(r"$\Delta^2_{21}(k)\ [\mathrm{mK}^2]$", fontsize=12)
    ax.set_title("Training vs Validation Noise-Subtracted Power Spectra", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    compare_plot_file = os.path.join(args.outdir, "power_spectra_nnsub_train_vs_val.png")
    fig.savefig(compare_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved train vs val plot to: %s", compare_plot_file)

    # --- Plot 4: Fractional standard deviation ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogx(k_bins, std_ps / mean_ps, 'bo-', lw=2, markersize=8)
    ax.set_xlabel(r"$k\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=12)
    ax.set_ylabel(r"$\sigma_{\Delta^2} / \langle\Delta^2\rangle$", fontsize=12)
    ax.set_title("Fractional Standard Deviation of Noise-Subtracted Power Spectrum", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    frac_std_file = os.path.join(args.outdir, "fractional_std_nnsub.png")
    fig.savefig(frac_std_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved fractional std plot to: %s", frac_std_file)

    # ==============================================================
    # COMPUTE COVARIANCE MATRIX
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPUTING COVARIANCE MATRIX")
    logger.info("=" * 60)

    # Compute covariance matrix
    # rowvar=False means each column is a variable (k-bin) and each row is an observation
    cov_matrix = np.cov(combined, rowvar=False)  # shape (n_kbins, n_kbins)

    logger.info("Covariance matrix shape: %s", cov_matrix.shape)
    logger.info("Covariance matrix diagonal (variances): %s", np.diag(cov_matrix))

    # Print full covariance matrix
    logger.info("")
    logger.info("Full Covariance Matrix:")
    logger.info("k_bins used: %s", k_bins)
    for i in range(cov_matrix.shape[0]):
        logger.info("  Row %d: %s", i, cov_matrix[i, :])

    # Verify the covariance computation manually
    N_total = combined.shape[0]
    fluctuations = combined - mean_ps
    cov_manual = fluctuations.T @ fluctuations / (N_total - 1)

    max_diff = np.max(np.abs(cov_matrix - cov_manual))
    logger.info("Max difference between np.cov and manual computation: %.2e", max_diff)
    if max_diff < 1e-10:
        logger.info("✅ Covariance computation verified!")
    else:
        logger.warning("⚠️ Small numerical difference in covariance computation")

    # ==============================================================
    # SAVE COVARIANCE MATRIX
    # ==============================================================

    cov_file = os.path.join(args.outdir, "cov_matrix_nnsub.npy")
    np.save(cov_file, cov_matrix)
    logger.info("Saved covariance matrix to: %s", cov_file)

    # ==============================================================
    # COMPUTE CORRELATION MATRIX
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPUTING CORRELATION MATRIX")
    logger.info("=" * 60)

    # Correlation = Cov[i,j] / sqrt(Cov[i,i] * Cov[j,j])
    corr_matrix = cov_matrix / np.outer(std_ps, std_ps)

    logger.info("Correlation matrix diagonal (should be 1): %s", np.diag(corr_matrix))

    # Print full correlation matrix
    logger.info("")
    logger.info("Full Correlation Matrix:")
    for i in range(corr_matrix.shape[0]):
        logger.info("  Row %d: %s", i, np.array2string(corr_matrix[i, :], precision=4, suppress_small=True))

    # Save correlation matrix
    corr_file = os.path.join(args.outdir, "corr_matrix_nnsub.npy")
    np.save(corr_file, corr_matrix)
    logger.info("Saved correlation matrix to: %s", corr_file)

    # ==============================================================
    # PLOT COVARIANCE AND CORRELATION MATRICES
    # ==============================================================

    logger.info("")
    logger.info("Generating covariance/correlation plots...")

    # --- Plot: Covariance matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cov_matrix, origin='lower', cmap='viridis')
    ax.set_xlabel("k-bin index", fontsize=12)
    ax.set_ylabel("k-bin index", fontsize=12)
    ax.set_title("Covariance Matrix (Noise-Subtracted PS)", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Cov$[\Delta^2_i, \Delta^2_j]$", fontsize=10)

    # Add k-bin labels
    ax.set_xticks(range(len(k_bins)))
    ax.set_yticks(range(len(k_bins)))
    ax.set_xticklabels([f"{k:.2f}" for k in k_bins], rotation=45, fontsize=8)
    ax.set_yticklabels([f"{k:.2f}" for k in k_bins], fontsize=8)

    cov_plot_file = os.path.join(args.outdir, "covariance_matrix_nnsub.png")
    plt.savefig(cov_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved covariance matrix plot to: %s", cov_plot_file)

    # --- Plot: Correlation matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel("k-bin index", fontsize=12)
    ax.set_ylabel("k-bin index", fontsize=12)
    ax.set_title("Correlation Matrix (Noise-Subtracted PS)", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation coefficient", fontsize=10)

    # Add k-bin labels
    ax.set_xticks(range(len(k_bins)))
    ax.set_yticks(range(len(k_bins)))
    ax.set_xticklabels([f"{k:.2f}" for k in k_bins], rotation=45, fontsize=8)
    ax.set_yticklabels([f"{k:.2f}" for k in k_bins], fontsize=8)

    corr_plot_file = os.path.join(args.outdir, "correlation_matrix_nnsub.png")
    plt.savefig(corr_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation matrix plot to: %s", corr_plot_file)

    # ==============================================================
    # STATISTICS SUMMARY
    # ==============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("STATISTICS SUMMARY")
    logger.info("=" * 60)

    logger.info("Number of training samples: %d", len(training_seeds))
    logger.info("Number of validation samples: %d", len(validation_seeds))
    logger.info("Total samples used for covariance: %d", N_total)
    logger.info("Number of k-bins: %d", len(k_bins))
    logger.info("")
    logger.info("Mean noise PS that was subtracted: %s", noise_ps_mean)
    logger.info("")
    logger.info("Noise-subtracted power spectrum statistics:")
    logger.info("  Mean: %s", mean_ps)
    logger.info("  Std:  %s", std_ps)
    logger.info("  Fractional std: %s", std_ps / mean_ps)
    logger.info("")
    logger.info("Covariance matrix condition number: %.2e", np.linalg.cond(cov_matrix))

    # ==============================================================
    # COMPLETION
    # ==============================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPUTATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("Output directory: %s", args.outdir)
    logger.info("")
    logger.info("Output files:")
    logger.info("  Training PS (noise-subtracted): %s", training_nnsub_file)
    logger.info("  Validation PS (noise-subtracted): %s", validation_nnsub_file)
    logger.info("  Mean PS: %s", mean_ps_file)
    logger.info("  k-bins: %s", k_bins_file)
    logger.info("  Covariance matrix: %s", cov_file)
    logger.info("  Correlation matrix: %s", corr_file)
    logger.info("")
    logger.info("Plots:")
    logger.info("  All PS: %s", plot_file)
    logger.info("  Mean PS: %s", mean_plot_file)
    logger.info("  Train vs Val: %s", compare_plot_file)
    logger.info("  Fractional std: %s", frac_std_file)
    logger.info("  Covariance matrix: %s", cov_plot_file)
    logger.info("  Correlation matrix: %s", corr_plot_file)


if __name__ == "__main__":
    main()
