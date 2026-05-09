#!/usr/bin/env python
# coding: utf-8

"""MCMC with wedge-filtered noisy cylindrical 2D power spectrum (10x10; no mean-noise subtraction).

This script runs emcee MCMC using the *wedge-filtered noisy* cylindrical 2D power spectrum
as the summary statistic.

Pipeline per parameter set [Q^M_HII, log10Mmin]
---------------------------------------------
1) Use the *fixed* posterior matter-field realization (snap_000)
2) Build the signal Tb cube from (Q^M_HII, log10Mmin)
3) Add the *same fixed* noise cube for every likelihood evaluation
4) Mean-subtract the noisy Tb cube
5) Remove the wedge by FFT -> wedge mask -> inverse FFT
6) Compute 10x10 linearly binned cylindrical 2DPS using the `script` backend (units="")
7) Compare to the precomputed observed wedge-filtered noisy 2DPS with a Gaussian likelihood

Notes
-----
- No mean noisy-2DPS subtraction is performed.
- The wedge mask keeps modes with k_parallel >= C * k_perp.
- `units=""` is used for 2DPS computation (Tb is already in mK).
- Script is intended to run on the cluster from:
  /scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import logging
import os
import sys
from multiprocessing import Pool

import numpy as np

import corner
import emcee
import matplotlib.pyplot as plt

import script


# ==============================================================
# ARGUMENTS
# ==============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCMC with wedge-filtered noisy cylindrical 2DPS (10x10)"
    )
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers")
    p.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps")
    p.add_argument("--nsteps", type=int, default=3000, help="Number of production steps")
    p.add_argument("--nbins_par", type=int, default=10, help="Number of k_parallel bins")
    p.add_argument("--nbins_perp", type=int, default=10, help="Number of k_perp bins")
    p.add_argument(
        "--wedge_C",
        type=float,
        default=3.140347,
        help="Wedge slope C for keeping modes with k_parallel >= C * k_perp.",
    )
    p.add_argument(
        "--ncores",
        type=int,
        default=1,
        help="Number of CPU cores for parallel MCMC (use 0 for all available)",
    )
    return p.parse_args()


args = parse_args()
os.makedirs(args.outdir, exist_ok=True)


# ==============================================================
# FIXED PATHS (SCRATCH)
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/wedge_filtered_Tb"

# Posterior snapshot seed file
POSTERIOR_SEED_FILE = f"{PROJECT_DIR}/posterior_seed.npy"

# Fixed noise file for posterior analysis
NOISE_FILE = f"{PROJECT_DIR}/AAstar_noise21cm_cube_seed1689456004_posterior.npz"

# Observed wedge-filtered noisy 2DPS (mock observation)
OBS_2DPS_FILE = (
    f"{PROJECT_DIR}/target_data/"
    "target_tempmap_w_noise_ms_wedge_filtered_seed1259935638_2dps_script_linear_10x10.npy"
)

# Covariance matrix for flattened 2DPS
COV_MATRIX_FILE = f"{PROJECT_DIR}/cov_files/covariance_training_plus_validation_2dps_wedge_filtered_main.npy"


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


def setup_logging(logpath: str) -> logging.Logger:
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


logger = setup_logging(os.path.join(args.outdir, "mcmc_2dps_wedge_filtered_noisy_run.log"))

logger.info("=" * 80)
logger.info("MCMC WITH WEDGE-FILTERED NOISY CYLINDRICAL 2DPS - START")
logger.info("=" * 80)


# ==============================================================
# LOAD OBSERVED 2DPS + BINNING
# ==============================================================

if not os.path.exists(OBS_2DPS_FILE):
    logger.error("Observed 2DPS file not found: %s", OBS_2DPS_FILE)
    sys.exit(1)

ps2d_obs = np.asarray(np.load(OBS_2DPS_FILE), dtype=np.float64)
logger.info("Loaded observed wedge-filtered noisy 2DPS: %s", OBS_2DPS_FILE)
logger.info("Observed 2DPS shape: %s", ps2d_obs.shape)

expected_shape = (args.nbins_par, args.nbins_perp)
if ps2d_obs.shape != expected_shape:
    logger.error(
        "Observed 2DPS shape mismatch: got %s, expected %s", ps2d_obs.shape, expected_shape
    )
    sys.exit(1)

ps_obs = ps2d_obs.reshape(-1).astype(np.float64)
logger.info("Flattened observed 2DPS length: %d", ps_obs.size)

obs_dir = os.path.dirname(OBS_2DPS_FILE)
obs_base = os.path.basename(OBS_2DPS_FILE)

suffix = "_2dps_script_linear_10x10.npy"
if not obs_base.endswith(suffix):
    logger.error("Unexpected observed 2DPS filename: %s", obs_base)
    sys.exit(1)

obs_prefix = obs_base[: -len(suffix)]

k_par_edges_file = os.path.join(obs_dir, f"{obs_prefix}_k_par_edges_linear.npy")
k_perp_edges_file = os.path.join(obs_dir, f"{obs_prefix}_k_perp_edges_linear.npy")

if os.path.exists(k_par_edges_file) and os.path.exists(k_perp_edges_file):
    k_par_edges = np.asarray(np.load(k_par_edges_file), dtype=np.float64)
    k_perp_edges = np.asarray(np.load(k_perp_edges_file), dtype=np.float64)
    logger.info("Loaded k_par_edges: %s", k_par_edges_file)
    logger.info("Loaded k_perp_edges: %s", k_perp_edges_file)
else:
    logger.warning("k-edge files not found; computing linear edges from FFT grid.")

    k1d = np.abs(np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi).astype(np.float64)
    nonzero = k1d[k1d > 0.0]
    if nonzero.size == 0:
        logger.error("Invalid FFT k-grid for ngrid=%d", ngrid)
        sys.exit(1)

    kmin = float(np.min(nonzero))
    kmax = float(np.max(nonzero))
    k_par_edges = np.linspace(kmin, kmax, args.nbins_par + 1)
    k_perp_edges = np.linspace(kmin, kmax, args.nbins_perp + 1)

if k_par_edges.size != args.nbins_par + 1 or k_perp_edges.size != args.nbins_perp + 1:
    logger.error(
        "k-edge size mismatch: k_par_edges=%d (expected %d), k_perp_edges=%d (expected %d)",
        k_par_edges.size,
        args.nbins_par + 1,
        k_perp_edges.size,
        args.nbins_perp + 1,
    )
    sys.exit(1)

logger.info("k_par_edges: %s", np.array2string(k_par_edges, precision=6))
logger.info("k_perp_edges: %s", np.array2string(k_perp_edges, precision=6))


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
    logger.error("Fixed noise file not found: %s", NOISE_FILE)
    sys.exit(1)

with np.load(NOISE_FILE) as noise_data:
    if "noisecube_21cm" in noise_data:
        noise_cube = noise_data["noisecube_21cm"].astype(np.float32, copy=False)
    elif "noise_cube" in noise_data:
        noise_cube = noise_data["noise_cube"].astype(np.float32, copy=False)
    elif "noise" in noise_data:
        noise_cube = noise_data["noise"].astype(np.float32, copy=False)
    elif len(noise_data.files) == 1:
        noise_cube = noise_data[noise_data.files[0]].astype(np.float32, copy=False)
    else:
        logger.error(
            "Noise cube key not found in %s. Available keys: %s",
            NOISE_FILE,
            noise_data.files,
        )
        sys.exit(1)

logger.info("Loaded fixed noise cube from: %s", NOISE_FILE)
logger.info("Noise cube shape: %s", noise_cube.shape)


# ==============================================================
# INITIALIZE SIMULATION DATA (ONCE)
# ==============================================================

logger.info("Initializing simulation data...")

sim_data = script.default_simulation_data(
    snap_path,
    outpath,
    sigma_8=sigma_8,
    ns=ns,
    omega_b=omega_b,
    scaledist=scaledist,
)

matter_data = script.matter_fields(
    sim_data,
    ngrid,
    outpath,
    overwrite_files=False,
)

ionization_map = script.ionization_map(matter_data)

# Initialize powspec helpers once (FFTW plan/k-grids)
matter_data.initialize_powspec()

# Validate noise cube shape
if noise_cube.shape != matter_data.densitycontr_arr.shape:
    logger.error(
        "Noise cube shape %s does not match density field shape %s",
        noise_cube.shape,
        matter_data.densitycontr_arr.shape,
    )
    sys.exit(1)

logger.info("Simulation data initialized successfully.")


# ==============================================================
# WEDGE FILTER HELPERS
# ==============================================================


def get_3d_k_grids(ngrid_val: int, box_val: float) -> tuple[np.ndarray, np.ndarray]:
    k1d = np.fft.fftfreq(ngrid_val, d=box_val / ngrid_val) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_par = np.abs(kz)
    k_perp = np.sqrt(kx**2 + ky**2)
    return k_par, k_perp


def apply_wedge_filter(map_3d: np.ndarray, wedge_mask: np.ndarray) -> np.ndarray:
    ft = np.fft.fftn(map_3d)
    ft[~wedge_mask] = 0.0
    return np.fft.ifftn(ft).real.astype(np.float32, copy=False)


k_par_grid, k_perp_grid = get_3d_k_grids(ngrid, box)
wedge_mask = k_par_grid >= float(args.wedge_C) * k_perp_grid
logger.info("Wedge slope C: %.6f", float(args.wedge_C))
logger.info(
    "Wedge mask keeps %d / %d Fourier cells (%.4f).",
    int(np.count_nonzero(wedge_mask)),
    int(wedge_mask.size),
    float(np.mean(wedge_mask)),
)


# ==============================================================
# 2DPS COMPUTATION
# ==============================================================


def compute_flat_2dps_from_Tb(Tb_mk: np.ndarray) -> np.ndarray:
    """Compute flattened 10x10 cylindrical 2DPS from a Tb cube (in mK)."""

    ps2d, kount = ionization_map.get_binned_powspec_cylindrical(
        Tb_mk.astype(np.float32, copy=False),
        k_par_edges,
        k_perp_edges,
        convolve=False,
        units="",
    )

    ps2d = np.asarray(ps2d, dtype=np.float64)
    kount = np.asarray(kount, dtype=np.float64)

    if ps2d.shape != expected_shape:
        raise RuntimeError(f"Expected 2DPS shape {expected_shape}, got {ps2d.shape}.")

    empty = kount == 0
    if np.any(empty):
        ps2d = ps2d.copy()
        ps2d[empty] = 0.0

    if not np.all(np.isfinite(ps2d)):
        raise RuntimeError("Non-finite entries found in 2DPS result.")

    return ps2d.reshape(-1).astype(np.float64)


def simulate_wedge_filtered_noisy_2dps(QHII_val: float, log10Mmin_val: float) -> np.ndarray:
    """Simulate wedge-filtered noisy flattened cylindrical 2DPS (fixed noise)."""

    fcoll_arr = matter_data.get_fcoll_for_Mmin(log10Mmin_val)

    mass_weighted_fcoll = np.mean(fcoll_arr * (1.0 + matter_data.densitycontr_arr))
    if mass_weighted_fcoll <= 0:
        raise RuntimeError("Mass-weighted fcoll is non-positive. Check Mmin or density field.")

    zeta = QHII_val / mass_weighted_fcoll

    qi_arr = ionization_map.get_qi(zeta * fcoll_arr)
    xHI_arr = 1.0 - qi_arr

    Tb_bar = 27.0 * (omega_b * h**2 / 0.023) * np.sqrt(
        0.15 / (omega_m * h**2) * (1.0 + z_target) / 10.0
    )

    Tb_signal = Tb_bar * xHI_arr * (1.0 + matter_data.densitycontr_arr)

    Tb_noisy = Tb_signal + noise_cube
    Tb_ms = Tb_noisy - np.mean(Tb_noisy, dtype=np.float64)

    Tb_wedge_filtered = apply_wedge_filter(Tb_ms.astype(np.float32, copy=False), wedge_mask)

    return compute_flat_2dps_from_Tb(Tb_wedge_filtered)


# ==============================================================
# COVARIANCE MATRIX
# ==============================================================

if os.path.exists(COV_MATRIX_FILE):
    logger.info("Loading covariance matrix from: %s", COV_MATRIX_FILE)
    Sigma = np.asarray(np.load(COV_MATRIX_FILE), dtype=np.float64)
else:
    logger.warning("Covariance matrix not found at: %s", COV_MATRIX_FILE)
    logger.warning("Using diagonal covariance approximation (10%% fractional variance)")
    frac_var = 0.1
    Sigma = np.diag((frac_var * np.maximum(np.abs(ps_obs), 1e-12)) ** 2)

if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
    logger.error("Covariance matrix must be square. Got shape %s", Sigma.shape)
    sys.exit(1)

if Sigma.shape[0] != ps_obs.size:
    logger.error(
        "Covariance dimension mismatch: cov is %s but summary length is %d",
        Sigma.shape,
        ps_obs.size,
    )
    sys.exit(1)

eps = 1e-6 * np.trace(Sigma) / Sigma.shape[0]
Sigma = Sigma + eps * np.eye(Sigma.shape[0])

Sigma_inv = np.linalg.inv(Sigma)
logdet_Sigma = np.linalg.slogdet(Sigma)[1]

logger.info("Covariance matrix shape: %s", Sigma.shape)
logger.info("Covariance matrix regularized.")

np.save(os.path.join(args.outdir, "covariance_matrix_used.npy"), Sigma)


# ==============================================================
# PRIOR, LIKELIHOOD, POSTERIOR
# ==============================================================

QHII_min, QHII_max = 0.1, 0.95
log10Mmin_min, log10Mmin_max = 7.5, 10.5

logger.info("Prior bounds:")
logger.info("  Q^M_HII: [%.2f, %.2f]", QHII_min, QHII_max)
logger.info("  log10Mmin: [%.1f, %.1f]", log10Mmin_min, log10Mmin_max)


def log_prior(params: np.ndarray) -> float:
    QHII, log10Mmin = params
    if QHII_min <= QHII <= QHII_max and log10Mmin_min <= log10Mmin <= log10Mmin_max:
        return -np.log(QHII_max - QHII_min) - np.log(log10Mmin_max - log10Mmin_min)
    return -np.inf


def log_likelihood(params: np.ndarray) -> float:
    QHII, log10Mmin = params

    try:
        ps_theta = simulate_wedge_filtered_noisy_2dps(QHII, log10Mmin)
    except Exception:
        return -np.inf

    d = ps_obs - ps_theta
    chi2 = d @ Sigma_inv @ d
    return -0.5 * (chi2 + logdet_Sigma + len(ps_obs) * np.log(2.0 * np.pi))


def log_posterior(params: np.ndarray) -> float:
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

start_center = theta_fid.copy()
start_spread = 0.01

p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    while True:
        proposal = start_center + start_spread * np.random.randn(ndim)
        if (QHII_min < proposal[0] < QHII_max) and (
            log10Mmin_min < proposal[1] < log10Mmin_max
        ):
            p0[i] = proposal
            break

logger.info("Initial walker positions initialized around fiducial.")

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

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)

logger.info("Running burn-in (%d steps)...", nburn)
state = sampler.run_mcmc(p0, nburn, progress=True)
sampler.reset()

logger.info("Running production (%d steps)...", nsteps)
sampler.run_mcmc(state, nsteps, progress=True)

mcmc_chain = sampler.get_chain(discard=0, flat=True)

logger.info("MCMC completed.")
logger.info("Final chain shape: %s", mcmc_chain.shape)

if pool is not None:
    pool.close()
    pool.join()
    logger.info("Pool closed.")


# ==============================================================
# SAVE RESULTS
# ==============================================================

chain_file = os.path.join(args.outdir, "mcmc_2dps_wedge_filtered_noisy_chain.npy")
np.save(chain_file, mcmc_chain)
logger.info("Saved MCMC chain to: %s", chain_file)

acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
logger.info("Mean acceptance fraction: %.3f", acceptance_fraction)

try:
    autocorr_time = sampler.get_autocorr_time(quiet=True)
    logger.info("Autocorrelation time: %s", autocorr_time)
except Exception as e:
    autocorr_time = np.array([np.nan, np.nan])
    logger.warning("Could not compute autocorrelation time: %s", str(e))

np.save(os.path.join(args.outdir, "observed_2dps_10x10.npy"), ps2d_obs)


# ==============================================================
# CREDIBLE INTERVALS
# ==============================================================


def credible_interval(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, [16, 50, 84])


ci_results = np.array([credible_interval(mcmc_chain[:, i]) for i in range(ndim)])

logger.info("")
logger.info("=" * 60)
logger.info("MCMC RESULTS (Wedge-filtered Noisy Cylindrical 2DPS Summary)")
logger.info("=" * 60)

labels = [r"Q^M_HII", r"log10Mmin"]
for i, label in enumerate(labels):
    lo, med, hi = ci_results[i]
    logger.info("%s: 16/50/84 = [%.4f, %.4f, %.4f]", label, lo, med, hi)

logger.info(
    "True (fiducial) values: Q^M_HII = %.2f, log10Mmin = %.1f",
    theta_fid[0],
    theta_fid[1],
)

results_file = os.path.join(args.outdir, "mcmc_2dps_wedge_filtered_noisy_results.npz")
np.savez(
    results_file,
    chain=mcmc_chain,
    true_params=theta_fid,
    credible_intervals=ci_results,
    observed_2dps=ps2d_obs,
    observed_2dps_flat=ps_obs,
    k_par_edges=k_par_edges,
    k_perp_edges=k_perp_edges,
    covariance_used=Sigma,
    acceptance_fraction=acceptance_fraction,
    autocorr_time=autocorr_time,
    noise_file=NOISE_FILE,
    snapshot_path=snap_path,
    wedge_C=float(args.wedge_C),
    wedge_mask_kept_fraction=float(np.mean(wedge_mask)),
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

corner.overplot_lines(fig, theta_fid, color="red", linewidth=2)
corner.overplot_points(
    fig,
    theta_fid[None, :],
    marker="x",
    color="red",
    markersize=10,
)

handles = [
    plt.Line2D([], [], color="C1", label="MCMC Posterior (Wedge-filtered Noisy 2DPS)"),
    plt.Line2D([], [], color="red", linestyle="--", label="Truth"),
]
fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)

corner_file = os.path.join(args.outdir, "mcmc_2dps_wedge_filtered_noisy_corner.png")
plt.savefig(corner_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved corner plot to: %s", corner_file)


# ==============================================================
# TRACE PLOTS
# ==============================================================

logger.info("Generating trace plots...")

full_chain = sampler.get_chain()

fig, axes = plt.subplots(ndim, 1, figsize=(10, 6), sharex=True)
labels_plot = [r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"]

for i in range(ndim):
    ax = axes[i]
    ax.plot(full_chain[:, :, i], alpha=0.3)
    ax.set_ylabel(labels_plot[i], fontsize=12)

axes[-1].set_xlabel("Step", fontsize=12)
plt.tight_layout()

trace_file = os.path.join(args.outdir, "mcmc_2dps_wedge_filtered_noisy_trace.png")
plt.savefig(trace_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved trace plot to: %s", trace_file)


# ==============================================================
# MARGINAL HISTOGRAMS
# ==============================================================

logger.info("Generating marginal histograms...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(mcmc_chain[:, 0], bins=50, density=True, color="skyblue", edgecolor="k", alpha=0.7)
ax.axvline(theta_fid[0], color="red", linestyle="--", linewidth=2, label="Truth")
ax.axvline(ci_results[0, 1], color="C1", linestyle="-", linewidth=2, label="Median")
ax.axvline(ci_results[0, 0], color="C1", linestyle=":", linewidth=1.5, label="16/84%")
ax.axvline(ci_results[0, 2], color="C1", linestyle=":", linewidth=1.5)
ax.set_xlabel(r"$Q^M_{\mathrm{HII}}$", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.legend(fontsize=10)
ax.set_title("Marginal Posterior: $Q^M_{\\mathrm{HII}}$", fontsize=12)

ax = axes[1]
ax.hist(mcmc_chain[:, 1], bins=50, density=True, color="lightgreen", edgecolor="k", alpha=0.7)
ax.axvline(theta_fid[1], color="red", linestyle="--", linewidth=2, label="Truth")
ax.axvline(ci_results[1, 1], color="C1", linestyle="-", linewidth=2, label="Median")
ax.axvline(ci_results[1, 0], color="C1", linestyle=":", linewidth=1.5, label="16/84%")
ax.axvline(ci_results[1, 2], color="C1", linestyle=":", linewidth=1.5)
ax.set_xlabel(r"$\\log_{10} M_{\min}$", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.legend(fontsize=10)
ax.set_title("Marginal Posterior: $\\log_{10} M_{\\min}$", fontsize=12)

plt.tight_layout()

marginal_file = os.path.join(args.outdir, "mcmc_2dps_wedge_filtered_noisy_marginals.png")
plt.savefig(marginal_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved marginal histograms to: %s", marginal_file)


# ==============================================================
# COMPLETION
# ==============================================================

logger.info("")
logger.info("=" * 80)
logger.info("MCMC WITH WEDGE-FILTERED NOISY CYLINDRICAL 2DPS COMPLETED")
logger.info("=" * 80)
logger.info("Output directory: %s", args.outdir)
logger.info("MCMC chain: %s", chain_file)
logger.info("Corner plot: %s", corner_file)
logger.info("Trace plot: %s", trace_file)
logger.info("Marginal histograms: %s", marginal_file)
logger.info("Results summary: %s", results_file)
logger.info("")
logger.info("Key settings:")
logger.info("  - Fixed snapshot: %s", snap_path)
logger.info("  - Fixed noise file: %s", NOISE_FILE)
logger.info("  - Wedge slope C: %.6f", float(args.wedge_C))
logger.info("  - No mean noisy-2DPS subtraction")
logger.info("  - Summary statistic: 10x10 cylindrical 2DPS (flattened to 100)")
