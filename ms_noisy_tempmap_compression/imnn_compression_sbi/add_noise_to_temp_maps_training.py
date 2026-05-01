#!/usr/bin/env python
# coding: utf-8

"""HPC-friendly script to add noise to temperature maps (Training data).

This script:
1. Reads raw temperature maps (stored as dictionary with seeds)
2. Reads corresponding noise maps (stored as individual .npz files)
3. Adds noise to temperature maps for each seed
4. Stores the combined maps in the same format as temperature maps
5. Plots the first 10 combined maps
"""

import argparse
import logging
import sys
import os
import numpy as np

# Set non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def setup_logging(logpath):
    """Setup logging to both file and stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(logpath, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Add noise to temperature maps (Training data)"
    )
    p.add_argument(
        "--tempmap_dir",
        default="./temp_maps/temp_maps_training",
        help="Directory containing temperature map .npy files"
    )
    p.add_argument(
        "--noisemap_dir",
        default="./noise_maps/noise_maps_training",
        help="Directory containing noise map .npz files"
    )
    p.add_argument(
        "--outdir",
        default="./temp_plus_noise_maps_raw/temp_plus_noise_maps_raw_training",
        help="Directory to save output files"
    )
    p.add_argument(
        "--logfile",
        default="add_noise_training_run.log",
        help="Log filename"
    )
    return p.parse_args()


# ==============================================================
# MAIN SETUP
# ==============================================================

args = parse_args()
os.makedirs(args.outdir, exist_ok=True)
logpath = os.path.join(args.outdir, args.logfile)
logger = setup_logging(logpath)

logger.info("=" * 70)
logger.info("Add Noise to Temperature Maps Script (Training Data)")
logger.info("=" * 70)
logger.info("Temperature map directory: %s", args.tempmap_dir)
logger.info("Noise map directory: %s", args.noisemap_dir)
logger.info("Output directory: %s", args.outdir)


# ==============================================================
# LOAD TEMPERATURE MAPS
# ==============================================================

logger.info("")
logger.info("Loading temperature maps...")

tempmap_files = {
    'main': os.path.join(args.tempmap_dir, "training_tempmap_main.npy"),
    'QHII_plus': os.path.join(args.tempmap_dir, "training_tempmap_QHII_plus.npy"),
    'QHII_minus': os.path.join(args.tempmap_dir, "training_tempmap_QHII_minus.npy"),
    'Mmin_plus': os.path.join(args.tempmap_dir, "training_tempmap_Mmin_plus.npy"),
    'Mmin_minus': os.path.join(args.tempmap_dir, "training_tempmap_Mmin_minus.npy"),
}

tempmaps = {}
for key, filepath in tempmap_files.items():
    if os.path.exists(filepath):
        tempmaps[key] = np.load(filepath, allow_pickle=True).item()
        logger.info("  Loaded %s: %d seeds", key, len(tempmaps[key]))
    else:
        logger.error("  File not found: %s", filepath)
        sys.exit(1)

# Get list of seeds from main temperature map
seeds = list(tempmaps['main'].keys())
logger.info("Total seeds to process: %d", len(seeds))


# ==============================================================
# ADD NOISE TO TEMPERATURE MAPS
# ==============================================================

logger.info("")
logger.info("=" * 70)
logger.info("Adding noise to temperature maps...")
logger.info("=" * 70)

from tqdm import tqdm

# Storage for combined maps
combined_maps = {
    'main': {},
    'QHII_plus': {},
    'QHII_minus': {},
    'Mmin_plus': {},
    'Mmin_minus': {},
}

missing_noise = []
processed_count = 0

for seed in tqdm(seeds, desc="Processing seeds", ncols=100):
    # Load corresponding noise map
    noise_file = os.path.join(args.noisemap_dir, f"AAstar_noise21cm_cube_seed{seed}.npz")
    
    if not os.path.exists(noise_file):
        missing_noise.append(seed)
        continue
    
    # Load noise cube
    noise_data = np.load(noise_file)
    noise_cube = noise_data['noisecube_21cm']
    
    # Add noise to all temperature map variants
    for key in combined_maps.keys():
        temp_map = tempmaps[key][seed]
        combined_maps[key][seed] = temp_map + noise_cube
    
    processed_count += 1

logger.info("")
logger.info("Processed %d seeds successfully", processed_count)
if missing_noise:
    logger.warning("Missing noise files for %d seeds: %s", len(missing_noise), missing_noise[:10])


# ==============================================================
# SAVE RESULTS
# ==============================================================

logger.info("")
logger.info("Saving combined maps...")

output_files = {
    'main': os.path.join(args.outdir, "training_temp_plus_noise_main.npy"),
    'QHII_plus': os.path.join(args.outdir, "training_temp_plus_noise_QHII_plus.npy"),
    'QHII_minus': os.path.join(args.outdir, "training_temp_plus_noise_QHII_minus.npy"),
    'Mmin_plus': os.path.join(args.outdir, "training_temp_plus_noise_Mmin_plus.npy"),
    'Mmin_minus': os.path.join(args.outdir, "training_temp_plus_noise_Mmin_minus.npy"),
}

for key, filepath in output_files.items():
    np.save(filepath, combined_maps[key])
    logger.info("  Saved %s: %d seeds", filepath, len(combined_maps[key]))


# ==============================================================
# CHECK FILE SIZES
# ==============================================================

logger.info("")
logger.info("File sizes:")
total_size = 0
for key, filepath in output_files.items():
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024**2)
        total_size += size_mb
        logger.info("  %s: %.2f MB", filepath, size_mb)
logger.info("Total storage used: %.2f MB", total_size)


# ==============================================================
# VERIFICATION: Sample statistics
# ==============================================================

logger.info("")
logger.info("=" * 70)
logger.info("Verification: Sample statistics from first 10 seeds")
logger.info("=" * 70)

seeds_list = list(combined_maps['main'].keys())
for i, seed in enumerate(seeds_list[:10]):
    combined = combined_maps['main'][seed]
    temp_orig = tempmaps['main'][seed]
    
    logger.info("Seed %s:", seed)
    logger.info("  Original temp - Min: %.2f, Max: %.2f, Mean: %.2f mK", 
                temp_orig.min(), temp_orig.max(), temp_orig.mean())
    logger.info("  Combined      - Min: %.2f, Max: %.2f, Mean: %.2f mK",
                combined.min(), combined.max(), combined.mean())


# ==============================================================
# PLOTTING: First 10 seeds for ALL 5 CASES
# ==============================================================

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger.info("")
logger.info("=" * 70)
logger.info("Generating plots for all 5 cases...")
logger.info("=" * 70)

# Configuration
box_plot = 128.0
ngrid_plot = 64
ygrid_plot = int(ngrid_plot * 0.54)
fid_QHII = 0.54
fid_log10Mmin = 9.0
delta_QHII = 0.005
delta_log10Mmin = 0.01

# Select first 10 seeds
n_show = min(10, len(seeds_list))
seeds_plot = seeds_list[:n_show]

logger.info("Plotting first %d seeds", n_show)
logger.info("Using y-slice index = %d (out of total %d)", ygrid_plot, ngrid_plot)

# Define cases to plot
cases_info = {
    'main': {'title': f'Fiducial ($Q^M_{{HII}}={fid_QHII}$, $\\log_{{10}}M_{{min}}={fid_log10Mmin}$)', 'filename': 'training_temp_plus_noise_main_first10.png'},
    'QHII_plus': {'title': f'$Q^M_{{HII}}+$ ({fid_QHII + delta_QHII})', 'filename': 'training_temp_plus_noise_QHII_plus_first10.png'},
    'QHII_minus': {'title': f'$Q^M_{{HII}}-$ ({fid_QHII - delta_QHII})', 'filename': 'training_temp_plus_noise_QHII_minus_first10.png'},
    'Mmin_plus': {'title': f'$\\log_{{10}}M_{{min}}+$ ({fid_log10Mmin + delta_log10Mmin})', 'filename': 'training_temp_plus_noise_Mmin_plus_first10.png'},
    'Mmin_minus': {'title': f'$\\log_{{10}}M_{{min}}-$ ({fid_log10Mmin - delta_log10Mmin})', 'filename': 'training_temp_plus_noise_Mmin_minus_first10.png'},
}

for case_key, case_info in cases_info.items():
    logger.info("Plotting case: %s", case_key)
    
    # Create 2×5 subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    axes = axes.flatten()

    # Determine global color scale for this case
    vmin_global, vmax_global = np.inf, -np.inf
    for seed in seeds_plot:
        arr = combined_maps[case_key][seed]
        slice_data = arr[:, ygrid_plot, :]
        vmin_global = min(vmin_global, np.min(slice_data))
        vmax_global = max(vmax_global, np.max(slice_data))

    logger.info("  Color scale: [%.2f, %.2f] mK", vmin_global, vmax_global)

    # Plot
    for idx, seed in enumerate(seeds_plot):
        ax = axes[idx]
        arr = combined_maps[case_key][seed]
        slice_data = arr[:, ygrid_plot, :].T

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)

        im = ax.imshow(
            slice_data,
            origin="lower",
            extent=[0, box_plot, 0, box_plot],
            interpolation="bicubic",
            cmap="viridis",
            vmin=vmin_global,
            vmax=vmax_global,
        )

        ax.set_title(rf"Seed {seed}", fontsize=9)

        if idx % 5 == 0:
            ax.set_ylabel(r"$z\ (\mathrm{cMpc}/h)$", fontsize=9)
        if idx >= 5:
            ax.set_xlabel(r"$x\ (\mathrm{cMpc}/h)$", fontsize=9)

        cbar = plt.colorbar(im, cax=cax)
        if idx % 5 == 4:
            cbar.set_label(r"$T_b + \mathrm{Noise}\ \mathrm{[mK]}$", fontsize=9)
        cbar.ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        rf"Training Data: Temperature + Noise Maps (RAW) — {case_info['title']}"
        rf" | $y$-slice = {ygrid_plot}/{ngrid_plot}",
        fontsize=11,
        y=0.995,
        weight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    png_out = os.path.join(args.outdir, case_info['filename'])
    plt.savefig(png_out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("  Saved plot to %s", png_out)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

logger.info("")
logger.info("=" * 70)
logger.info("SCRIPT COMPLETED SUCCESSFULLY")
logger.info("=" * 70)
logger.info("Output directory: %s", args.outdir)
logger.info("Total seeds processed: %d", processed_count)
logger.info("NOTE: These are RAW temperature + noise maps (no mean subtraction)")
