#!/usr/bin/env python
# coding: utf-8

"""SBI with Noise-Subtracted Power Spectrum for mean-subtracted noisy temperature maps.

This script performs Sequential Neural Posterior Estimation (SNPE) using
the noise-subtracted dimensionless 21cm power spectrum as the summary statistic.

Workflow for each simulation:
1. Generate temperature map using MUSIC + script
2. Add noise (randomly chosen from noise folder)
3. Mean subtract the signal+noise
4. Compute dimensionless power spectrum
5. **Subtract mean noise power spectrum** (to remove noise bias)
6. Use as summary statistic for SBI

Key difference from sbi_with_power_spectrum.py:
- After computing Delta^2(k), we subtract the pre-computed mean noise PS
- Target PS is also the noise-subtracted version (dimless_nnsub_ps.npy)
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import logging
import sys
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import corner

from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn

import script
from script import two_lpt


# ==============================================================
# ARGUMENTS
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SBI with Noise-Subtracted Power Spectrum")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--rounds", type=int, default=4, help="Number of SNPE rounds")
    p.add_argument("--sims_per_round", type=int, default=800, help="Simulations per round")
    p.add_argument("--nbins", type=int, default=10, help="Number of k-bins for power spectrum")
    return p.parse_args()


args = parse_args()
os.makedirs(args.outdir, exist_ok=True)


# ==============================================================
# FIXED PATHS
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise"

# Target power spectrum (observed data) - NOISE SUBTRACTED VERSION
TARGET_PS_FILE = f"{PROJECT_DIR}/target_data/power_spectra/dimless_nnsub_ps.npy"

# Mean noise power spectrum (to subtract from each simulation)
NOISE_PS_MEAN_FILE = f"{PROJECT_DIR}/noise_maps/power_spectra/PS_noise_maps_mean.npy"

# Noise files directory
NOISE_DIR = f"{PROJECT_DIR}/noise_maps/noise_maps_sbi"

# Seeds to exclude (training/validation seeds)
SEED_DIR = PROJECT_DIR

# MUSIC executable
MUSIC_EXEC = "/scratch/subhankar/software/music/build/MUSIC"

# Target seed (to exclude)
TARGET_SEED = 1259935638


# ==============================================================
# COSMOLOGY & SIMULATION PARAMETERS
# ==============================================================

box = 128.0
dx = 1.0
z_target = 7.0
omega_m = 0.308
omega_l = 1 - omega_m
omega_b = 0.0482
h = 0.678
sigma_8 = 0.829
ns = 0.961
ngrid = 64
scaledist = 1e-3


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


logger = setup_logging(os.path.join(args.outdir, "sbi_ps_nnsub_run.log"))

logger.info("=" * 80)
logger.info("SBI WITH NOISE-SUBTRACTED POWER SPECTRUM - PIPELINE STARTED")
logger.info("=" * 80)


# ==============================================================
# LOAD MEAN NOISE POWER SPECTRUM (FOR SUBTRACTION)
# ==============================================================

if not os.path.exists(NOISE_PS_MEAN_FILE):
    logger.error("Mean noise power spectrum not found: %s", NOISE_PS_MEAN_FILE)
    sys.exit(1)

noise_ps_mean = np.load(NOISE_PS_MEAN_FILE).astype(np.float32)
logger.info("Loaded mean noise power spectrum: %s", NOISE_PS_MEAN_FILE)
logger.info("Mean noise PS shape: %s", noise_ps_mean.shape)
logger.info("Mean noise PS values: %s", noise_ps_mean)


# ==============================================================
# LOAD TARGET POWER SPECTRUM (OBSERVED DATA - NOISE SUBTRACTED)
# ==============================================================

if not os.path.exists(TARGET_PS_FILE):
    logger.error("Target power spectrum (noise-subtracted) not found: %s", TARGET_PS_FILE)
    sys.exit(1)

target_ps = np.load(TARGET_PS_FILE).astype(np.float32)
logger.info("Loaded target power spectrum (noise-subtracted): %s", TARGET_PS_FILE)
logger.info("Target PS shape: %s", target_ps.shape)
logger.info("Target PS values: %s", target_ps)

# Verify shapes match
if target_ps.shape != noise_ps_mean.shape:
    logger.error("Shape mismatch! Target PS: %s, Mean noise PS: %s", 
                 target_ps.shape, noise_ps_mean.shape)
    sys.exit(1)


# ==============================================================
# K-BINS SETUP (SAME AS USED FOR TARGET PS)
# ==============================================================

def build_k_edges_kbins(ngrid, box, nbins, log_bins=True):
    """Build k-edges and k-bins using same convention as script.set_k_edges."""
    kmin = 2.0 * np.pi / box
    kmax = np.pi * ngrid / box

    if log_bins:
        if nbins < 2:
            dlnk = 0.1
            nbins = int((np.log(kmax) - np.log(kmin)) / dlnk)
        lnk_edges = np.linspace(np.log(kmin), np.log(kmax), num=nbins + 1, endpoint=True)
        lnk_bins = 0.5 * (lnk_edges[:-1] + lnk_edges[1:])
        k_edges = np.exp(lnk_edges)
        k_bins = np.exp(lnk_bins)
    else:
        if nbins < 2:
            nbins = int(ngrid / 2) - 1
        k_edges = np.linspace(kmin, kmax, num=nbins + 1, endpoint=True)
        k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

    return k_edges, k_bins


k_edges, k_bins = build_k_edges_kbins(ngrid, box, args.nbins, log_bins=True)
logger.info("k_edges: %s", k_edges)
logger.info("k_bins: %s", k_bins)


# ==============================================================
# POWER SPECTRUM HELPER (STANDALONE - NO MATTER_DATA REQUIRED)
# ==============================================================

# Initialize ionization_map helper with dummy density for PS computation
dummy_density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
ps_helper = script.ionization_map(
    matter_fields=None,
    densitycontr_arr=dummy_density,
    box=box,
    z=z_target,
    omega_m=omega_m,
    h=h,
)

logger.info("Power spectrum helper initialized.")


def compute_dimensionless_ps(temp_map_mk):
    """
    Compute dimensionless power spectrum from a 3D temperature map in mK.
    
    Parameters
    ----------
    temp_map_mk : np.ndarray
        3D temperature map in mK units, shape (ngrid, ngrid, ngrid)
    
    Returns
    -------
    delta2 : np.ndarray
        Dimensionless power spectrum Delta^2(k) = k^3 * P(k) / (2*pi^2), shape (nbins,)
    """
    # Use units='' since input is already in mK
    powspec_binned, kount = ps_helper.get_binned_powspec(
        temp_map_mk, k_edges, convolve=False, units='', bin_weighted=False
    )
    
    valid = kount > 0
    k_valid = k_bins[valid]
    pk_valid = powspec_binned[valid]
    
    # Delta^2(k) = k^3 * P(k) / (2*pi^2)
    delta2 = (k_valid ** 3) * pk_valid / (2.0 * np.pi ** 2)
    
    return delta2.astype(np.float32)


def compute_noise_subtracted_ps(temp_map_mk):
    """
    Compute noise-subtracted dimensionless power spectrum.
    
    1. Compute raw dimensionless PS from temperature map
    2. Subtract mean noise PS to remove noise bias
    
    Parameters
    ----------
    temp_map_mk : np.ndarray
        3D temperature map in mK units, shape (ngrid, ngrid, ngrid)
    
    Returns
    -------
    delta2_nnsub : np.ndarray
        Noise-subtracted dimensionless power spectrum, shape (nbins,)
    """
    # Compute raw dimensionless PS
    delta2_raw = compute_dimensionless_ps(temp_map_mk)
    
    # Subtract mean noise PS
    delta2_nnsub = delta2_raw - noise_ps_mean
    
    return delta2_nnsub


# ==============================================================
# SEED HANDLING (EXCLUDE TRAINING/VALIDATION/TARGET SEEDS)
# ==============================================================

excluded_seeds = set()

# Load and exclude training seeds
training_seeds_file = f"{SEED_DIR}/random_seeds_training.npy"
if os.path.exists(training_seeds_file):
    excluded_seeds.update(np.load(training_seeds_file).tolist())
    logger.info("Excluded %d training seeds", len(np.load(training_seeds_file)))

# Load and exclude validation seeds
validation_seeds_file = f"{SEED_DIR}/random_seeds_validation.npy"
if os.path.exists(validation_seeds_file):
    excluded_seeds.update(np.load(validation_seeds_file).tolist())
    logger.info("Excluded %d validation seeds", len(np.load(validation_seeds_file)))

# Exclude target seed
excluded_seeds.add(TARGET_SEED)
logger.info("Total excluded seeds: %d", len(excluded_seeds))

used_seeds = set()


def get_unique_seed():
    """Get a unique random seed not in excluded or already used set."""
    while True:
        s = np.random.randint(0, 2**31 - 1)
        if s not in excluded_seeds and s not in used_seeds:
            used_seeds.add(s)
            return s


# ==============================================================
# NOISE FILE HANDLING (RANDOM SELECTION WITHOUT REPLACEMENT)
# ==============================================================

noise_files = sorted([
    os.path.join(NOISE_DIR, f)
    for f in os.listdir(NOISE_DIR)
    if f.endswith(".npz")
])

logger.info("Found %d noise files in %s", len(noise_files), NOISE_DIR)

# Shuffle for random selection
np.random.shuffle(noise_files)
noise_index = 0


def get_noise_cube():
    """Get a noise cube from the shuffled list."""
    global noise_index
    
    if noise_index >= len(noise_files):
        # Reshuffle and reset if we've used all noise files
        logger.warning("All noise files used, reshuffling...")
        np.random.shuffle(noise_files)
        noise_index = 0
    
    noise_file = noise_files[noise_index]
    noise_index += 1
    
    noise_data = np.load(noise_file)
    if "noisecube_21cm" in noise_data:
        return noise_data["noisecube_21cm"]
    elif "noise_cube" in noise_data:
        return noise_data["noise_cube"]
    else:
        raise KeyError(f"Could not find noise cube in {noise_file}")


# ==============================================================
# TEMPERATURE MAP SIMULATION FUNCTION
# ==============================================================

def simulate_temp_map(QHII_val, log10Mmin_val, seed):
    """
    Simulate a 21cm brightness temperature map for given parameters.
    
    Parameters
    ----------
    QHII_val : float
        Target mass-weighted ionized fraction
    log10Mmin_val : float
        log10 of minimum halo mass
    seed : int
        Random seed for MUSIC
    
    Returns
    -------
    Tb : np.ndarray
        Brightness temperature map in mK, shape (ngrid, ngrid, ngrid)
    """
    outpath = os.path.join(args.outdir, f"temp_seed_{seed}")
    snap_path = f"{outpath}/snap_000"
    os.makedirs(outpath, exist_ok=True)

    try:
        # Run MUSIC to generate density field
        two_lpt.run_music(
            MUSIC_EXEC, box, [z_target],
            seed, outpath, "snap", dx,
            omega_m, omega_l, omega_b,
            h, sigma_8, ns
        )

        # Load simulation data
        sim_data = script.default_simulation_data(
            snap_path, outpath,
            sigma_8=sigma_8,
            ns=ns,
            omega_b=omega_b,
            scaledist=scaledist
        )

        matter_data = script.matter_fields(
            sim_data, ngrid, outpath,
            overwrite_files=False
        )

        ion_map = script.ionization_map(matter_data)

        # Compute fcoll and zeta from QHII
        fcoll_arr = matter_data.get_fcoll_for_Mmin(log10Mmin_val)
        mass_weighted_fcoll = np.mean(fcoll_arr * (1 + matter_data.densitycontr_arr))
        
        if mass_weighted_fcoll <= 0:
            raise RuntimeError("Mass-weighted fcoll is non-positive")
        
        zeta = QHII_val / mass_weighted_fcoll

        # Compute ionization field
        qi_arr = ion_map.get_qi(zeta * fcoll_arr)
        xHI_arr = 1 - qi_arr

        # Mean brightness temperature
        Tb_bar = 27.0 * (omega_b * h**2 / 0.023) * np.sqrt(
            0.15 / (omega_m * h**2) * (1 + z_target) / 10.0
        )

        # Brightness temperature map
        Tb = Tb_bar * xHI_arr * (1 + matter_data.densitycontr_arr)

    finally:
        # Clean up temporary files
        if os.path.exists(outpath):
            shutil.rmtree(outpath, ignore_errors=True)

    return Tb.astype(np.float32)


# ==============================================================
# FULL SIMULATION PIPELINE: TEMP MAP -> NOISY -> MS -> PS -> NNSUB
# ==============================================================

def simulate_noise_subtracted_power_spectrum(QHII_val, log10Mmin_val):
    """
    Full simulation pipeline with noise subtraction:
    1. Generate temperature map
    2. Add noise
    3. Mean subtract
    4. Compute dimensionless power spectrum
    5. **Subtract mean noise PS**
    
    Returns
    -------
    delta2_nnsub : np.ndarray
        Noise-subtracted dimensionless power spectrum, shape (nbins,)
    """
    # Get unique seed
    seed = get_unique_seed()
    
    # 1. Generate temperature map (signal)
    Tb_signal = simulate_temp_map(QHII_val, log10Mmin_val, seed)
    
    # 2. Add noise
    noise_cube = get_noise_cube()
    Tb_noisy = Tb_signal + noise_cube
    
    # 3. Mean subtract
    Tb_ms = Tb_noisy - np.mean(Tb_noisy)
    
    # 4. Compute dimensionless power spectrum and subtract mean noise PS
    delta2_nnsub = compute_noise_subtracted_ps(Tb_ms)
    
    return delta2_nnsub


# ==============================================================
# SBI SETUP
# ==============================================================

# Prior over 2 parameters: [Q^M_HII, log10Mmin]
prior = BoxUniform(
    low=torch.tensor([0.1, 7.5]),
    high=torch.tensor([0.95, 10.5])
)

logger.info("Prior bounds:")
logger.info("  Q^M_HII: [0.1, 0.95]")
logger.info("  log10Mmin: [7.5, 10.5]")

# Density estimator (MAF)
density_estimator_build_fun = posterior_nn(model="maf")

# Initialize inference
inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)

# Initial proposal is the prior
proposal = prior

# Observed power spectrum as torch tensor (NOISE-SUBTRACTED)
x_obs = torch.tensor(target_ps, dtype=torch.float32)
logger.info("Observed PS (x_obs, noise-subtracted) shape: %s", x_obs.shape)


# ==============================================================
# SNPE ROUNDS
# ==============================================================

posterior = None

for r in range(args.rounds):
    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING ROUND %d / %d", r + 1, args.rounds)
    logger.info("=" * 60)
    
    # Sample parameters from current proposal
    theta = proposal.sample((args.sims_per_round,))  # shape (N, 2)
    logger.info("Sampled %d parameter sets from proposal", args.sims_per_round)
    
    # Simulate power spectra
    summaries = []
    
    for i in tqdm(range(args.sims_per_round), desc=f"Simulations (round {r+1})"):
        QHII_val, log10Mmin_val = theta[i].cpu().numpy()
        sim_seed = get_unique_seed()
        
        # Get noise file info before loading
        if noise_index >= len(noise_files):
            logger.warning("All noise files used, reshuffling...")
            np.random.shuffle(noise_files)
            noise_index = 0
        
        noise_file = noise_files[noise_index]
        noise_seed_str = os.path.basename(noise_file).split("seed")[-1].split(".")[0]
        noise_seed = int(noise_seed_str)
        
        logger.info(
            "Running MUSIC for sim_seed=%d | noise_seed=%d | QHII=%.4f | log10Mmin=%.4f",
            sim_seed, noise_seed, QHII_val, log10Mmin_val
        )
        
        try:
            # 1. Generate temperature map (signal)
            Tb_signal = simulate_temp_map(QHII_val, log10Mmin_val, sim_seed)
            
            # 2. Add noise
            noise_data = np.load(noise_file)
            if "noisecube_21cm" in noise_data:
                noise_cube = noise_data["noisecube_21cm"]
            elif "noise_cube" in noise_data:
                noise_cube = noise_data["noise_cube"]
            else:
                raise KeyError(f"Could not find noise cube in {noise_file}")
            noise_index += 1
            
            Tb_noisy = Tb_signal + noise_cube
            
            # 3. Mean subtract
            Tb_ms = Tb_noisy - np.mean(Tb_noisy)
            
            # 4. Compute dimensionless power spectrum
            delta2_raw = compute_dimensionless_ps(Tb_ms)
            
            # 5. **SUBTRACT MEAN NOISE PS** (key difference!)
            delta2_nnsub = delta2_raw - noise_ps_mean
            
            summaries.append(delta2_nnsub)
            
        except Exception as e:
            logger.warning("Simulation %d failed: %s. Using zeros.", i, str(e))
            noise_index += 1  # Still increment to avoid reusing failed noise file
            summaries.append(np.zeros(args.nbins, dtype=np.float32))
    
    # Stack summaries into tensor
    x = torch.tensor(np.stack(summaries, axis=0), dtype=torch.float32)  # shape (N, nbins)
    logger.info("Summaries tensor shape: %s", x.shape)
    
    # Append simulations to inference
    if r == 0:
        inference = inference.append_simulations(theta, x)
    else:
        inference = inference.append_simulations(theta, x, proposal=proposal)
    
    # Train density estimator
    logger.info("Training density estimator...")
    density_estimator = inference.train(
        force_first_round_loss=(r == 0),
        show_train_summary=True,
    )
    
    # Build posterior
    posterior = inference.build_posterior(density_estimator)
    
    # Update proposal for next round
    proposal = posterior.set_default_x(x_obs)
    
    logger.info("Round %d completed.", r + 1)


# ==============================================================
# FINAL POSTERIOR SAMPLING
# ==============================================================

logger.info("")
logger.info("=" * 60)
logger.info("SAMPLING FROM FINAL POSTERIOR")
logger.info("=" * 60)

posterior = posterior.set_default_x(x_obs)
samples = posterior.sample((96000,))  # shape (96000, 2)
samples_np = samples.cpu().numpy()

logger.info("Final posterior samples shape: %s", samples_np.shape)

# Save posterior samples
samples_file = os.path.join(args.outdir, "snpe_ps_nnsub_posterior_samples.npy")
np.save(samples_file, samples_np)
logger.info("Saved posterior samples to: %s", samples_file)


# ==============================================================
# CREDIBLE INTERVALS
# ==============================================================

def credible_interval(x):
    return np.percentile(x, [16, 50, 84])

ci_results = np.array([credible_interval(samples_np[:, i]) for i in range(2)])

theta_true = np.array([0.54, 9.0])  # Fiducial values

logger.info("")
logger.info("=" * 60)
logger.info("SBI RESULTS (Noise-Subtracted Power Spectrum Summary)")
logger.info("=" * 60)

labels = [r"Q^M_HII", r"log10Mmin"]
for i, label in enumerate(labels):
    lo, mid, hi = ci_results[i]
    logger.info("%s: %.4f (+%.4f, -%.4f)", label, mid, hi - mid, mid - lo)

logger.info("True values: Q^M_HII = %.2f, log10Mmin = %.1f", theta_true[0], theta_true[1])

# Save results summary
results_file = os.path.join(args.outdir, "sbi_ps_nnsub_results.npz")
np.savez(
    results_file,
    samples=samples_np,
    true_params=theta_true,
    credible_intervals=ci_results,
    target_ps=target_ps,
    noise_ps_mean=noise_ps_mean,
    k_bins=k_bins,
)
logger.info("Saved results to: %s", results_file)


# ==============================================================
# CORNER PLOT
# ==============================================================

logger.info("Generating corner plot...")

fig = corner.corner(
    samples_np,
    labels=[r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"],
    bins=50,
    smooth=1.0,
    color="C0",
    show_titles=True,
    title_fmt=".3f",
    label_kwargs={"fontsize": 14},
)

# Overlay true values
corner.overplot_lines(fig, theta_true, color="red", linewidth=2)
corner.overplot_points(
    fig,
    theta_true[None, :],
    marker="x",
    color="red",
    markersize=10,
)

# Add legend
handles = [
    plt.Line2D([], [], color="C0", label="Noise-Sub PS Posterior"),
    plt.Line2D([], [], color="red", linestyle="--", label="Truth"),
]
fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)

corner_file = os.path.join(args.outdir, "snpe_ps_nnsub_corner.png")
plt.savefig(corner_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved corner plot to: %s", corner_file)


# ==============================================================
# COMPLETION
# ==============================================================

logger.info("")
logger.info("=" * 80)
logger.info("SBI WITH NOISE-SUBTRACTED POWER SPECTRUM COMPLETED SUCCESSFULLY")
logger.info("=" * 80)
logger.info("Output directory: %s", args.outdir)
logger.info("Posterior samples: %s", samples_file)
logger.info("Corner plot: %s", corner_file)
logger.info("Results summary: %s", results_file)
logger.info("")
logger.info("Key difference from standard SBI:")
logger.info("  - Mean noise PS subtracted from each simulation's PS")
logger.info("  - Target PS is also noise-subtracted (dimless_nnsub_ps.npy)")
logger.info("  - This removes noise bias from the summary statistic")
