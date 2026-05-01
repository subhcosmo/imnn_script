#!/usr/bin/env python
# coding: utf-8

"""HPC-friendly runner for generating 21cm brightness temperature maps (Training data).

Features added for HPC runs:
- Command-line args for output directory and seeds file
- File and stdout logging (log file is tailable for live monitoring)
- Non-interactive Matplotlib backend (Agg)
- Plot saved to files instead of displayed

This script generates raw (non mean-subtracted) brightness temperature maps
using the semi-numerical approach with MUSIC 2LPT initial conditions.
"""

import argparse
import logging
import sys
import os
import numpy as np

# Set non-interactive backend before importing pyplot later
import matplotlib
matplotlib.use("Agg")


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def check_duplicates(seeds_arr):
    """Check for duplicate seeds in the array."""
    unique_seeds, counts = np.unique(seeds_arr, return_counts=True)
    duplicates = unique_seeds[counts > 1]
    return duplicates


def setup_logging(logpath):
    """Setup logging to both file and stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # remove default handlers if any
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
        description="Generate and save 21cm brightness temperature maps (HPC-friendly)"
    )
    p.add_argument(
        "--seeds", 
        default="random_seeds_training.npy", 
        help=".npy file with seeds array"
    )
    p.add_argument(
        "--outdir", 
        default=".", 
        help="Directory to save outputs and logs"
    )
    p.add_argument(
        "--logfile", 
        default="training_temp_map_run.log", 
        help="Log filename to write progress to"
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
logger.info("21cm Brightness Temperature Map Generation Script (Training Data)")
logger.info("=" * 70)

# Load seeds and check duplicates
seeds = np.load(args.seeds)
dups = check_duplicates(seeds)
if dups.size > 0:
    logger.warning("Found %d duplicate seed values: %s", len(dups), dups)
else:
    logger.info("No duplicates found in seeds file: %s", args.seeds)


# ==============================================================
# IMPORT REQUIRED MODULES
# ==============================================================

import shutil
from tqdm import tqdm
import script
from script import two_lpt


# ==============================================================
# CONFIGURATION
# ==============================================================

music_exec = "/scratch/subhankar/software/music/build/MUSIC"
base_path = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise"

# Cosmology
box = 128.0         # Mpc/h
dx = 1.0            # Mpc/h
z_target = 7.0
zlist = np.array([z_target])
omega_m = 0.308
omega_l = 1.0 - omega_m
omega_b = 0.0482
h = 0.678
sigma_8 = 0.829
ns = 0.961

# Simulation grid parameters
ngrid = 64
scaledist = 1e-3
ygrid = int(ngrid * 0.54)

# Fiducial & perturbations
fid_log10Mmin = 9.0
fid_QHII = 0.54
delta_QHII = 0.005
delta_log10Mmin = 0.01

logger.info("Configuration:")
logger.info("  MUSIC executable: %s", music_exec)
logger.info("  Base path: %s", base_path)
logger.info("  Box size: %.1f Mpc/h, dx: %.1f Mpc/h", box, dx)
logger.info("  z_target: %.1f", z_target)
logger.info("  Cosmology: Omega_m=%.3f, Omega_L=%.3f, Omega_b=%.4f, h=%.3f, sigma_8=%.3f, n_s=%.3f",
            omega_m, omega_l, omega_b, h, sigma_8, ns)
logger.info("  ngrid: %d, scaledist: %.0e", ngrid, scaledist)
logger.info("  Fiducial: log10Mmin=%.1f, QHII=%.2f", fid_log10Mmin, fid_QHII)
logger.info("  Perturbations: delta_QHII=%.3f, delta_log10Mmin=%.2f", delta_QHII, delta_log10Mmin)


# ==============================================================
# LOAD ALL SEEDS
# ==============================================================

seeds = np.load(args.seeds).tolist()
logger.info("Loaded %d training seeds for data generation from %s.", len(seeds), args.seeds)


# ==============================================================
# STORAGE DICTIONARIES
# ==============================================================

tempmap_base = {}
tempmap_QHII_plus = {}
tempmap_QHII_minus = {}
tempmap_Mmin_plus = {}
tempmap_Mmin_minus = {}


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def mean_brightness_temperature(z, omega_b, omega_m, h):
    """
    Mean 21cm brightness temperature in mK.
    
    T̄_b(z) = 27 * (Ω_b h² / 0.023) * sqrt(0.15 / (Ω_m h² * (1+z)/10))
    """
    Tb = (27.0 * (omega_b * h**2 / 0.023) * 
          np.sqrt(0.15 / (omega_m * h**2) * (1.0 + z) / 10.0))
    return Tb  # mK


def compute_mass_weighted_fcoll(matter_data, log10Mmin_value):
    """Compute mass-weighted collapsed fraction."""
    fcoll_arr = matter_data.get_fcoll_for_Mmin(log10Mmin_value)
    return np.mean(fcoll_arr * (1.0 + matter_data.densitycontr_arr)), fcoll_arr


def compute_zeta_from_QHII(target_QHII, mass_weighted_fcoll):
    """Compute ionizing efficiency zeta from target QHII."""
    if mass_weighted_fcoll <= 0:
        raise RuntimeError("Mass-weighted fcoll is non-positive!")
    return target_QHII / mass_weighted_fcoll


def compute_brightness_temperature(matter_data, ion_map, zeta_value, log10Mmin_value, 
                                    z, omega_b, omega_m, h):
    """
    Compute raw brightness temperature map (no mean subtraction).
    
    δT_b(x) = T̄_b(z) * (1 - q_i) * (1 + δ)
    
    Parameters
    ----------
    matter_data : script.matter_fields
        Matter fields object containing density contrast.
    ion_map : script.ionization_map
        Ionization map object.
    zeta_value : float
        Ionizing efficiency.
    log10Mmin_value : float
        Log10 of minimum halo mass for star formation.
    z : float
        Redshift.
    omega_b, omega_m, h : float
        Cosmological parameters.
    
    Returns
    -------
    Tb_arr : np.ndarray
        Brightness temperature field in mK.
    """
    # Collapsed fraction
    fcoll_arr = matter_data.get_fcoll_for_Mmin(log10Mmin_value)
    
    # Ionized fraction
    qi_arr = ion_map.get_qi(zeta_value * fcoll_arr)
    
    # Neutral fraction
    xHI_arr = 1.0 - qi_arr
    
    # Mean brightness temperature
    Tb_bar = mean_brightness_temperature(z, omega_b, omega_m, h)
    
    # Brightness temperature field (mK) - RAW, no mean subtraction
    Tb_arr = Tb_bar * xHI_arr * (1.0 + matter_data.densitycontr_arr)
    
    return Tb_arr


# ==============================================================
# MAIN LOOP OVER ALL SEEDS
# ==============================================================

logger.info("")
logger.info("=" * 70)
logger.info("Starting brightness temperature map generation...")
logger.info("=" * 70)

for seed in tqdm(seeds, desc="Processing all seeds", ncols=100):

    outpath = f"{base_path}/output_seed_{seed}"
    snap_path = f"{outpath}/snap_000"
    os.makedirs(outpath, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(outpath)

    # Run MUSIC to generate initial conditions
    two_lpt.run_music(
        music_exec, box, zlist, seed, outpath, "snap", dx,
        omega_m, omega_l, omega_b, h, sigma_8, ns
    )

    os.chdir(cwd)

    if not os.path.exists(snap_path):
        raise RuntimeError(
            f"MUSIC failed: snapshot not found for seed {seed} at {snap_path}"
        )

    # Load simulation data
    sim_data = script.default_simulation_data(
        snap_path, outpath, sigma_8=sigma_8, ns=ns,
        omega_b=omega_b, scaledist=scaledist
    )
    matter_data = script.matter_fields(sim_data, ngrid, outpath, overwrite_files=False)
    ion_map = script.ionization_map(matter_data)

    # ==========================================================
    # Step 1: Fiducial case (base)
    # ==========================================================
    mass_weighted_fcoll_fid, fcoll_arr_fid = compute_mass_weighted_fcoll(
        matter_data, fid_log10Mmin
    )
    zeta_fid = compute_zeta_from_QHII(fid_QHII, mass_weighted_fcoll_fid)
    
    Tb_base = compute_brightness_temperature(
        matter_data, ion_map, zeta_fid, fid_log10Mmin,
        z_target, omega_b, omega_m, h
    )
    tempmap_base[str(seed)] = Tb_base  # RAW, no mean subtraction

    # ==========================================================
    # Step 2: QHII ± δ
    # ==========================================================

    # QHII+
    QHII_p = fid_QHII + delta_QHII
    zeta_plus = compute_zeta_from_QHII(QHII_p, mass_weighted_fcoll_fid)
    Tb_plus = compute_brightness_temperature(
        matter_data, ion_map, zeta_plus, fid_log10Mmin,
        z_target, omega_b, omega_m, h
    )
    tempmap_QHII_plus[str(seed)] = Tb_plus  # RAW

    # QHII-
    QHII_m = fid_QHII - delta_QHII
    zeta_minus = compute_zeta_from_QHII(QHII_m, mass_weighted_fcoll_fid)
    Tb_minus = compute_brightness_temperature(
        matter_data, ion_map, zeta_minus, fid_log10Mmin,
        z_target, omega_b, omega_m, h
    )
    tempmap_QHII_minus[str(seed)] = Tb_minus  # RAW

    # ==========================================================
    # Step 3: log10Mmin ± δ (QHII fixed)
    # ==========================================================

    for delta, label in [(+delta_log10Mmin, "plus"), (-delta_log10Mmin, "minus")]:
        log10Mmin_val = fid_log10Mmin + delta

        mass_weighted_fcoll_new, _ = compute_mass_weighted_fcoll(
            matter_data, log10Mmin_val
        )
        zeta_adjusted = compute_zeta_from_QHII(fid_QHII, mass_weighted_fcoll_new)

        Tb_new = compute_brightness_temperature(
            matter_data, ion_map, zeta_adjusted, log10Mmin_val,
            z_target, omega_b, omega_m, h
        )

        if label == "plus":
            tempmap_Mmin_plus[str(seed)] = Tb_new  # RAW
        else:
            tempmap_Mmin_minus[str(seed)] = Tb_new  # RAW

    # ==========================================================
    # CLEANUP: Delete snapshot folder to save disk space
    # ==========================================================
    if os.path.exists(outpath):
        shutil.rmtree(outpath)


# ==============================================================
# SAVE RESULTS
# ==============================================================

out_main = os.path.join(args.outdir, "training_tempmap_main.npy")
out_qhii_p = os.path.join(args.outdir, "training_tempmap_QHII_plus.npy")
out_qhii_m = os.path.join(args.outdir, "training_tempmap_QHII_minus.npy")
out_mmin_p = os.path.join(args.outdir, "training_tempmap_Mmin_plus.npy")
out_mmin_m = os.path.join(args.outdir, "training_tempmap_Mmin_minus.npy")

np.save(out_main, tempmap_base)
np.save(out_qhii_p, tempmap_QHII_plus)
np.save(out_qhii_m, tempmap_QHII_minus)
np.save(out_mmin_p, tempmap_Mmin_plus)
np.save(out_mmin_m, tempmap_Mmin_minus)

logger.info("")
logger.info("Saved numpy outputs to %s", args.outdir)


# ==============================================================
# CHECK FILE SIZES
# ==============================================================

files_to_check = [out_main, out_qhii_p, out_qhii_m, out_mmin_p, out_mmin_m]

logger.info("")
logger.info("All brightness temperature maps saved successfully:")
total_size = 0
for fname in files_to_check:
    if os.path.exists(fname):
        size_mb = os.path.getsize(fname) / (1024**2)
        total_size += size_mb
        logger.info(" - %s: %.2f MB", fname, size_mb)

logger.info("Total storage used: %.2f MB", total_size)
logger.info("Grid shape per map: %s", getattr(Tb_base, "shape", "unknown"))


# ==============================================================
# VERIFICATION: Print sample statistics
# ==============================================================

logger.info("")
logger.info("=" * 70)
logger.info("Verification: Sample statistics from first 10 seeds")
logger.info("=" * 70)

# Load the saved dictionary
tempmap_base_loaded = np.load(out_main, allow_pickle=True).item()

# Get all available seeds (as strings)
seeds_list = list(tempmap_base_loaded.keys())
logger.info("Total available seeds: %d", len(seeds_list))
logger.info("First 10 seeds: %s", seeds_list[:10])

# Loop over the first 10 seeds
for i, seed in enumerate(seeds_list[:10]):
    logger.info("==================== Seed %d: %s ====================", i+1, seed)
    
    Tb = tempmap_base_loaded[seed]
    
    logger.info("Array shape: %s", Tb.shape)
    logger.info("Array dtype: %s", Tb.dtype)
    logger.info("Min: %.4f mK", np.min(Tb))
    logger.info("Max: %.4f mK", np.max(Tb))
    logger.info("Mean: %.4f mK", np.mean(Tb))
    logger.info("Std: %.4f mK", np.std(Tb))


# ==============================================================
# PLOTTING: First 10 seeds
# ==============================================================

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger.info("")
logger.info("=" * 70)
logger.info("Generating plots...")
logger.info("=" * 70)

# Configuration for plotting
box_plot = 128.0
ngrid_plot = 64
ygrid_plot = int(ngrid_plot * 0.54)

# Select first 10 seeds
all_seeds = list(tempmap_base_loaded.keys())
n_show = min(10, len(all_seeds))
seeds_plot = all_seeds[:n_show]

logger.info("Plotting first %d seeds", n_show)
logger.info("Using y-slice index = %d (out of total %d)", ygrid_plot, ngrid_plot)

# Create 2×5 subplot grid
fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
axes = axes.flatten()

# Determine global color scale across all selected slices
vmin_global, vmax_global = np.inf, -np.inf

for seed in seeds_plot:
    Tb_arr = tempmap_base_loaded[seed]
    slice_data = Tb_arr[:, ygrid_plot, :]
    vmin_global = min(vmin_global, np.min(slice_data))
    vmax_global = max(vmax_global, np.max(slice_data))

logger.info("Global color scale across first %d seeds: [%.2f, %.2f] mK", 
            n_show, vmin_global, vmax_global)

# Plot each seed's y-slice
for idx, seed in enumerate(seeds_plot):
    ax = axes[idx]

    Tb_arr = tempmap_base_loaded[seed]
    slice_data = Tb_arr[:, ygrid_plot, :].T  # transpose for (x,z) view

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    im = ax.imshow(
        slice_data,
        origin="lower",
        extent=[0, box_plot, 0, box_plot],
        interpolation="bicubic",
        cmap="RdBu_r",
        vmin=vmin_global,
        vmax=vmax_global,
    )

    ax.set_title(rf"Seed {seed}", fontsize=9)

    # Axis labels
    if idx % 5 == 0:
        ax.set_ylabel(r"$z\ (\mathrm{cMpc}/h)$", fontsize=9)
    if idx >= 5:
        ax.set_xlabel(r"$x\ (\mathrm{cMpc}/h)$", fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, cax=cax)
    if idx % 5 == 4:
        cbar.set_label(r"$T_b\ \mathrm{[mK]}$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Tick formatting
    ax.tick_params(labelsize=8)

# Suptitle and layout
fig.suptitle(
    rf"Training Data: 21cm Brightness Temperature Maps — $y$-slice = {ygrid_plot}/{ngrid_plot}"
    rf" | Target $Q^M_{{\mathrm{{HII}}}} = {fid_QHII}$",
    fontsize=12,
    y=0.995,
    weight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save plots to outdir (HPC-friendly)
png_out = os.path.join(args.outdir, "training_tempmaps_first10_fiducial.png")
pdf_out = os.path.join(args.outdir, "training_tempmaps_first10_fiducial.pdf")
plt.savefig(pdf_out, bbox_inches="tight", dpi=300)
plt.savefig(png_out, bbox_inches="tight", dpi=300)
logger.info("Saved plots to %s and %s", png_out, pdf_out)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

logger.info("")
logger.info("=" * 70)
logger.info("SCRIPT COMPLETED SUCCESSFULLY")
logger.info("=" * 70)
logger.info("Output directory: %s", args.outdir)
logger.info("Total seeds processed: %d", len(seeds_list))
logger.info("NOTE: These are RAW temperature maps (no mean subtraction)")

