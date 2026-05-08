#!/usr/bin/env python
# coding: utf-8

"""MCMC with Noise-Subtracted Dimensionless Power Spectrum.

This script performs Markov Chain Monte Carlo (MCMC) inference using emcee,
with the noise-subtracted dimensionless 21cm power spectrum as the summary statistic.

Workflow:
1. Load a pre-generated posterior snapshot (fixed MUSIC output)
2. Load pre-computed observed PS (dimless_nnsub_ps.npy) and mean noise PS
3. For each MCMC parameter set [Q^M_HII, log10Mmin]:
   - Compute temperature field from the fixed density field
   - Add the fixed noise cube
   - Mean subtract the signal+noise
   - Compute dimensionless power spectrum
   - **Subtract mean noise PS** (to remove noise bias)
4. Compare against observed power spectrum via Gaussian likelihood
5. Run emcee MCMC sampler

Key difference from mcmc_with_power_spectrum.py:
- Subtracts mean noise PS from each simulated PS
- Uses pre-computed noise-subtracted observed PS (dimless_nnsub_ps.npy)
- Uses noise-subtracted covariance matrix (cov_matrix_nnsub.npy)
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import logging
import sys
import os
import numpy as np
from multiprocessing import Pool
import emcee
import corner
import matplotlib.pyplot as plt

import script


# ==============================================================
# ARGUMENTS
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(description="MCMC with Noise-Subtracted Power Spectrum")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers")
    p.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps")
    p.add_argument("--nsteps", type=int, default=3000, help="Number of production steps")
    p.add_argument("--nbins", type=int, default=10, help="Number of k-bins for power spectrum")
    p.add_argument("--ncores", type=int, default=1, help="Number of CPU cores for parallel MCMC (use 0 for all available)")
    return p.parse_args()


args = parse_args()
os.makedirs(args.outdir, exist_ok=True)


# ==============================================================
# FIXED PATHS
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise"

# Posterior snapshot seed file
POSTERIOR_SEED_FILE = f"{PROJECT_DIR}/posterior_seed.npy"

# Fixed noise file for posterior analysis
NOISE_FILE = f"{PROJECT_DIR}/posterior_data/AAstar_noise21cm_cube_seed1689456004_posterior.npz"

# Observed power spectrum (NOISE-SUBTRACTED)
OBS_PS_FILE = f"{PROJECT_DIR}/target_data/power_spectra/dimless_nnsub_ps.npy"

# Mean noise power spectrum (for subtraction)
NOISE_PS_MEAN_FILE = f"{PROJECT_DIR}/noise_maps/power_spectra/PS_noise_maps_mean.npy"

# Covariance matrix (NOISE-SUBTRACTED)
COV_MATRIX_FILE = f"{PROJECT_DIR}/power_spectra_nn_sub/cov_matrix_nnsub.npy"


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

# Fiducial parameters (true values)
theta_fid = np.array([0.54, 9.0])  # [Q^M_HII, log10Mmin]


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


logger = setup_logging(os.path.join(args.outdir, "mcmc_ps_nnsub_run.log"))

logger.info("=" * 80)
logger.info("MCMC WITH NOISE-SUBTRACTED POWER SPECTRUM - PIPELINE STARTED")
logger.info("=" * 80)


# ==============================================================
# LOAD MEAN NOISE POWER SPECTRUM (FOR SUBTRACTION)
# ==============================================================

if not os.path.exists(NOISE_PS_MEAN_FILE):
    logger.error("Mean noise power spectrum not found: %s", NOISE_PS_MEAN_FILE)
    sys.exit(1)

noise_ps_mean = np.load(NOISE_PS_MEAN_FILE).astype(np.float64)
logger.info("Loaded mean noise power spectrum: %s", NOISE_PS_MEAN_FILE)
logger.info("Mean noise PS shape: %s", noise_ps_mean.shape)
logger.info("Mean noise PS values: %s", noise_ps_mean)


# ==============================================================
# LOAD OBSERVED POWER SPECTRUM (PRE-COMPUTED, NOISE-SUBTRACTED)
# ==============================================================

if not os.path.exists(OBS_PS_FILE):
    logger.error("Observed PS file (noise-subtracted) not found: %s", OBS_PS_FILE)
    sys.exit(1)

ps_obs = np.load(OBS_PS_FILE).astype(np.float64)
logger.info("Loaded observed power spectrum (noise-subtracted): %s", OBS_PS_FILE)
logger.info("Observed PS shape: %s", ps_obs.shape)
logger.info("Observed PS values: %s", ps_obs)

# Verify shapes match
if ps_obs.shape != noise_ps_mean.shape:
    logger.error("Shape mismatch! Observed PS: %s, Mean noise PS: %s",
                 ps_obs.shape, noise_ps_mean.shape)
    sys.exit(1)


# ==============================================================
# LOAD POSTERIOR SNAPSHOT
# ==============================================================

if not os.path.exists(POSTERIOR_SEED_FILE):
    logger.error("Posterior seed file not found: %s", POSTERIOR_SEED_FILE)
    sys.exit(1)

posterior_seed = int(np.load(POSTERIOR_SEED_FILE)[0])
logger.info("Loaded posterior seed: %d", posterior_seed)

snap_path = f"{PROJECT_DIR}/output_posterior_seed_{posterior_seed}/snap_000"
outpath = f"{PROJECT_DIR}/output_posterior_seed_{posterior_seed}"

if not os.path.exists(snap_path):
    logger.error("Snapshot not found at: %s", snap_path)
    sys.exit(1)

logger.info("Using posterior snapshot from: %s", snap_path)


# ==============================================================
# LOAD FIXED NOISE CUBE
# ==============================================================

if not os.path.exists(NOISE_FILE):
    logger.error("Noise file not found: %s", NOISE_FILE)
    sys.exit(1)

noise_data = np.load(NOISE_FILE)
if "noisecube_21cm" in noise_data:
    noise_cube = noise_data["noisecube_21cm"].astype(np.float32)
elif "noise_cube" in noise_data:
    noise_cube = noise_data["noise_cube"].astype(np.float32)
else:
    logger.error("Could not find noise cube in %s", NOISE_FILE)
    sys.exit(1)

logger.info("Loaded fixed noise cube from: %s", NOISE_FILE)
logger.info("Noise cube shape: %s", noise_cube.shape)


# ==============================================================
# INITIALIZE SIMULATION DATA (ONCE)
# ==============================================================

logger.info("Initializing simulation data...")

sim_data = script.default_simulation_data(
    snap_path, outpath,
    sigma_8=sigma_8, ns=ns,
    omega_b=omega_b, scaledist=scaledist
)

matter_data = script.matter_fields(
    sim_data, ngrid, outpath,
    overwrite_files=False
)

ionization_map = script.ionization_map(matter_data)

logger.info("Simulation data initialized successfully.")


# ==============================================================
# K-BINS SETUP
# ==============================================================

matter_data.initialize_powspec()
k_edges, k_bins = matter_data.set_k_edges(nbins=args.nbins, log_bins=True)
logger.info("k_edges: %s", k_edges)
logger.info("k_bins: %s", k_bins)


# ==============================================================
# POWER SPECTRUM COMPUTATION FUNCTION
# ==============================================================

def compute_dimensionless_ps_from_Tb(Tb_ms):
    """
    Compute dimensionless power spectrum from mean-subtracted temperature map.
    
    Parameters
    ----------
    Tb_ms : np.ndarray
        Mean-subtracted 3D temperature map in mK, shape (ngrid, ngrid, ngrid)
    
    Returns
    -------
    delta2 : np.ndarray
        Dimensionless power spectrum Delta^2(k), shape (nbins,)
    """
    # Use units='' since input is already in mK
    powspec_binned, kount = ionization_map.get_binned_powspec(
        Tb_ms, k_edges, convolve=False, units='', bin_weighted=False
    )
    
    valid = kount > 0
    k_valid = k_bins[valid]
    pk_valid = powspec_binned[valid]
    
    # Delta^2(k) = k^3 * P(k) / (2*pi^2)
    delta2 = (k_valid ** 3) * pk_valid / (2.0 * np.pi ** 2)
    
    return delta2.astype(np.float64)


def simulate_noise_subtracted_ps(QHII_val, log10Mmin_val):
    """
    Simulate noise-subtracted dimensionless power spectrum for given parameters.
    
    Pipeline:
    1. Compute temperature field from fixed density field
    2. Add fixed noise cube
    3. Mean subtract
    4. Compute dimensionless power spectrum
    5. **Subtract mean noise PS** (key step for noise bias removal)
    
    Parameters
    ----------
    QHII_val : float
        Mass-weighted ionized fraction Q^M_HII
    log10Mmin_val : float
        Log10 of minimum halo mass
    
    Returns
    -------
    delta2_nnsub : np.ndarray
        Noise-subtracted dimensionless power spectrum, shape (nbins,)
    """
    # Collapsed fraction field for given Mmin
    fcoll_arr = matter_data.get_fcoll_for_Mmin(log10Mmin_val)
    
    # Mass-weighted collapsed fraction
    mass_weighted_fcoll = np.mean(fcoll_arr * (1.0 + matter_data.densitycontr_arr))
    
    if mass_weighted_fcoll <= 0:
        raise RuntimeError("Mass-weighted fcoll is non-positive. Check Mmin or density field.")
    
    # Compute zeta from Q^M_HII
    zeta = QHII_val / mass_weighted_fcoll
    
    # Ionization field
    qi_arr = ionization_map.get_qi(zeta * fcoll_arr)
    xHI_arr = 1 - qi_arr
    
    # Mean brightness temperature
    Tb_bar = 27.0 * (omega_b * h**2 / 0.023) * np.sqrt(
        0.15 / (omega_m * h**2) * (1 + z_target) / 10.0
    )
    
    # Brightness temperature map (signal)
    Tb_signal = Tb_bar * xHI_arr * (1 + matter_data.densitycontr_arr)
    
    # Add fixed noise
    Tb_noisy = Tb_signal + noise_cube
    
    # Mean subtract
    Tb_ms = Tb_noisy - np.mean(Tb_noisy)
    
    # Compute dimensionless power spectrum
    delta2_raw = compute_dimensionless_ps_from_Tb(Tb_ms)
    
    # **SUBTRACT MEAN NOISE PS** (key difference!)
    delta2_nnsub = delta2_raw - noise_ps_mean
    
    return delta2_nnsub


# ==============================================================
# COVARIANCE MATRIX (NOISE-SUBTRACTED)
# ==============================================================

if os.path.exists(COV_MATRIX_FILE):
    logger.info("Loading noise-subtracted covariance matrix from: %s", COV_MATRIX_FILE)
    Sigma = np.load(COV_MATRIX_FILE)
else:
    logger.warning("Covariance matrix not found at: %s", COV_MATRIX_FILE)
    logger.warning("Using diagonal covariance approximation (10%% fractional variance)")
    frac_var = 0.1  # 10% fractional variance
    Sigma = np.diag((frac_var * np.abs(ps_obs)) ** 2)

# Regularize for numerical stability
eps = 1e-6 * np.trace(Sigma) / Sigma.shape[0]
Sigma += eps * np.eye(Sigma.shape[0])

Sigma_inv = np.linalg.inv(Sigma)
logdet_Sigma = np.linalg.slogdet(Sigma)[1]

logger.info("Covariance matrix shape: %s", Sigma.shape)
logger.info("Covariance matrix regularized.")

# Save covariance matrix copy to output directory
np.save(os.path.join(args.outdir, "covariance_matrix_used.npy"), Sigma)


# ==============================================================
# PRIOR, LIKELIHOOD, POSTERIOR
# ==============================================================

# Prior bounds
QHII_min, QHII_max = 0.1, 0.95
log10Mmin_min, log10Mmin_max = 7.5, 10.5

logger.info("Prior bounds:")
logger.info("  Q^M_HII: [%.2f, %.2f]", QHII_min, QHII_max)
logger.info("  log10Mmin: [%.1f, %.1f]", log10Mmin_min, log10Mmin_max)


def log_prior(params):
    """Uniform prior over parameter space."""
    QHII, log10Mmin = params
    if QHII_min <= QHII <= QHII_max and log10Mmin_min <= log10Mmin <= log10Mmin_max:
        # Uniform prior
        return -np.log(QHII_max - QHII_min) - np.log(log10Mmin_max - log10Mmin_min)
    return -np.inf


def log_likelihood(params):
    """Gaussian log-likelihood with noise-subtracted PS."""
    QHII, log10Mmin = params
    
    try:
        ps_theta = simulate_noise_subtracted_ps(QHII, log10Mmin)
    except RuntimeError as e:
        return -np.inf
    
    d = ps_obs - ps_theta
    chi2 = d @ Sigma_inv @ d
    return -0.5 * (chi2 + logdet_Sigma + len(ps_obs) * np.log(2.0 * np.pi))


def log_posterior(params):
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ==============================================================
# RUN EMCEE MCMC
# ==============================================================

logger.info("")
logger.info("=" * 60)
logger.info("STARTING EMCEE MCMC")
logger.info("=" * 60)

ndim = 2
nwalkers = args.nwalkers
nburn = args.nburn
nsteps = args.nsteps

logger.info("MCMC configuration:")
logger.info("  ndim = %d", ndim)
logger.info("  nwalkers = %d", nwalkers)
logger.info("  nburn = %d", nburn)
logger.info("  nsteps = %d", nsteps)

# Initialize walkers around fiducial
start_center = theta_fid.copy()
start_spread = 0.01

# Ensure initial positions are within prior bounds
p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    while True:
        proposal = start_center + start_spread * np.random.randn(ndim)
        if (QHII_min < proposal[0] < QHII_max and
            log10Mmin_min < proposal[1] < log10Mmin_max):
            p0[i] = proposal
            break

logger.info("Initial walker positions initialized around fiducial.")

# ==============================================================
# SET UP PARALLEL POOL (IF REQUESTED)
# ==============================================================

ncores = args.ncores
if ncores == 0:
    ncores = os.cpu_count()

logger.info("Using %d CPU cores for parallelization", ncores)

if ncores > 1:
    pool = Pool(ncores)
    logger.info("Pool initialized with %d workers.", ncores)
else:
    pool = None
    logger.info("Running in serial mode (no multiprocessing).")

# Create sampler with optional pool
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)

# Burn-in
logger.info("Running burn-in (%d steps)...", nburn)
state = sampler.run_mcmc(p0, nburn, progress=True)
sampler.reset()

# Production
logger.info("Running production (%d steps)...", nsteps)
sampler.run_mcmc(state, nsteps, progress=True)

# Get chain
mcmc_chain = sampler.get_chain(discard=0, flat=True)  # shape (nwalkers * nsteps, 2)

logger.info("MCMC completed.")
logger.info("Final chain shape: %s", mcmc_chain.shape)

# Close pool if used
if pool is not None:
    pool.close()
    pool.join()
    logger.info("Pool closed.")


# ==============================================================
# SAVE RESULTS
# ==============================================================

# Save chain
chain_file = os.path.join(args.outdir, "mcmc_ps_nnsub_chain.npy")
np.save(chain_file, mcmc_chain)
logger.info("Saved MCMC chain to: %s", chain_file)

# Save sampler statistics
acceptance_fraction = np.mean(sampler.acceptance_fraction)
logger.info("Mean acceptance fraction: %.3f", acceptance_fraction)

try:
    autocorr_time = sampler.get_autocorr_time(quiet=True)
    logger.info("Autocorrelation time: %s", autocorr_time)
except Exception as e:
    autocorr_time = np.array([np.nan, np.nan])
    logger.warning("Could not compute autocorrelation time: %s", str(e))


# ==============================================================
# CREDIBLE INTERVALS
# ==============================================================

def credible_interval(x):
    return np.percentile(x, [16, 50, 84])


ci_results = np.array([credible_interval(mcmc_chain[:, i]) for i in range(ndim)])

logger.info("")
logger.info("=" * 60)
logger.info("MCMC RESULTS (Noise-Subtracted Power Spectrum Summary)")
logger.info("=" * 60)

labels = [r"Q^M_HII", r"log10Mmin"]
for i, label in enumerate(labels):
    lo, mid, hi = ci_results[i]
    logger.info("%s: %.4f (+%.4f, -%.4f)", label, mid, hi - mid, mid - lo)

logger.info("True (fiducial) values: Q^M_HII = %.2f, log10Mmin = %.1f", theta_fid[0], theta_fid[1])

# Save results summary
results_file = os.path.join(args.outdir, "mcmc_ps_nnsub_results.npz")
np.savez(
    results_file,
    chain=mcmc_chain,
    true_params=theta_fid,
    credible_intervals=ci_results,
    observed_ps=ps_obs,
    noise_ps_mean=noise_ps_mean,
    k_bins=k_bins,
    acceptance_fraction=acceptance_fraction,
    autocorr_time=autocorr_time,
)
logger.info("Saved results to: %s", results_file)


# ==============================================================
# CORNER PLOT
# ==============================================================

logger.info("Generating corner plot...")

fig = corner.corner(
    mcmc_chain,
    labels=[r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"],
    bins=50,
    smooth=1.0,
    color="C1",
    show_titles=True,
    title_fmt=".3f",
    label_kwargs={"fontsize": 14},
)

# Overlay true values
corner.overplot_lines(fig, theta_fid, color="red", linewidth=2)
corner.overplot_points(
    fig,
    theta_fid[None, :],
    marker="x",
    color="red",
    markersize=10,
)

# Add legend
handles = [
    plt.Line2D([], [], color="C1", label="MCMC Posterior (Noise-Sub PS)"),
    plt.Line2D([], [], color="red", linestyle="--", label="Truth"),
]
fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)

corner_file = os.path.join(args.outdir, "mcmc_ps_nnsub_corner.png")
plt.savefig(corner_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved corner plot to: %s", corner_file)


# ==============================================================
# TRACE PLOTS
# ==============================================================

logger.info("Generating trace plots...")

# Get full chain for trace plots (not flattened)
full_chain = sampler.get_chain()  # shape (nsteps, nwalkers, ndim)

fig, axes = plt.subplots(ndim, 1, figsize=(10, 6), sharex=True)
labels_plot = [r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"]

for i in range(ndim):
    ax = axes[i]
    ax.plot(full_chain[:, :, i], alpha=0.3, linewidth=0.5)
    ax.axhline(theta_fid[i], color='red', linestyle='--', linewidth=1.5, label='Truth')
    ax.set_ylabel(labels_plot[i], fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel("Step", fontsize=12)
plt.tight_layout()

trace_file = os.path.join(args.outdir, "mcmc_ps_nnsub_trace.png")
plt.savefig(trace_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved trace plot to: %s", trace_file)


# ==============================================================
# MARGINAL HISTOGRAMS
# ==============================================================

logger.info("Generating marginal histograms...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(mcmc_chain[:, 0], bins=50, density=True, color='skyblue', edgecolor='k', alpha=0.7)
ax.axvline(theta_fid[0], color='red', linestyle='--', linewidth=2, label='Truth')
ax.axvline(ci_results[0, 1], color='C1', linestyle='-', linewidth=2, label='Median')
ax.axvline(ci_results[0, 0], color='C1', linestyle=':', linewidth=1.5, label='16/84%')
ax.axvline(ci_results[0, 2], color='C1', linestyle=':', linewidth=1.5)
ax.set_xlabel(r"$Q^M_{\mathrm{HII}}$", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.legend(fontsize=10)
ax.set_title("Marginal Posterior: $Q^M_{\\mathrm{HII}}$", fontsize=12)

ax = axes[1]
ax.hist(mcmc_chain[:, 1], bins=50, density=True, color='lightgreen', edgecolor='k', alpha=0.7)
ax.axvline(theta_fid[1], color='red', linestyle='--', linewidth=2, label='Truth')
ax.axvline(ci_results[1, 1], color='C1', linestyle='-', linewidth=2, label='Median')
ax.axvline(ci_results[1, 0], color='C1', linestyle=':', linewidth=1.5, label='16/84%')
ax.axvline(ci_results[1, 2], color='C1', linestyle=':', linewidth=1.5)
ax.set_xlabel(r"$\log_{10} M_{\min}$", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.legend(fontsize=10)
ax.set_title("Marginal Posterior: $\\log_{10} M_{\\min}$", fontsize=12)

plt.tight_layout()

marginal_file = os.path.join(args.outdir, "mcmc_ps_nnsub_marginals.png")
plt.savefig(marginal_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved marginal histograms to: %s", marginal_file)


# ==============================================================
# COMPLETION
# ==============================================================

logger.info("")
logger.info("=" * 80)
logger.info("MCMC WITH NOISE-SUBTRACTED POWER SPECTRUM COMPLETED SUCCESSFULLY")
logger.info("=" * 80)
logger.info("Output directory: %s", args.outdir)
logger.info("MCMC chain: %s", chain_file)
logger.info("Corner plot: %s", corner_file)
logger.info("Trace plot: %s", trace_file)
logger.info("Marginal histograms: %s", marginal_file)
logger.info("Results summary: %s", results_file)
logger.info("")
logger.info("Key difference from standard MCMC:")
logger.info("  - Mean noise PS subtracted from each simulated PS")
logger.info("  - Observed PS is noise-subtracted (dimless_nnsub_ps.npy)")
logger.info("  - Uses noise-subtracted covariance matrix (cov_matrix_nnsub.npy)")
logger.info("  - This removes noise bias from the summary statistic")
