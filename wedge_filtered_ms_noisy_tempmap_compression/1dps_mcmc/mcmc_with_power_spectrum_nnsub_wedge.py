#!/usr/bin/env python
# coding: utf-8

"""MCMC with wedge-filtered, noise-subtracted dimensionless power spectrum.

For each parameter choice [Q^M_HII, log10Mmin], this script performs:
1) Build 21cm brightness-temperature signal from a fixed posterior snapshot.
2) Add fixed posterior noise cube.
3) Mean-subtract the noisy map.
4) Apply wedge filtering in Fourier space.
5) Compute dimensionless power spectrum Delta^2(k).
6) Subtract precomputed mean wedge-filtered noise power spectrum.

The resulting summary enters a Gaussian likelihood used by emcee.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import logging
import os
import sys
import numpy as np
from multiprocessing import Pool

import emcee
import corner
import matplotlib.pyplot as plt
from scipy.integrate import quad

import script


# ==============================================================
# ARGUMENTS
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MCMC with wedge-filtered, noise-subtracted power spectrum"
    )
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers")
    p.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps")
    p.add_argument("--nsteps", type=int, default=3000, help="Number of production steps")
    p.add_argument("--nbins", type=int, default=10, help="Number of k-bins")
    p.add_argument(
        "--ncores",
        type=int,
        default=1,
        help="Number of CPU cores for parallel MCMC (0 => all available)",
    )
    p.add_argument(
        "--wedge_C",
        type=float,
        default=None,
        help="Explicit wedge slope C. If omitted, compute from cosmology.",
    )
    p.add_argument(
        "--global_seed",
        type=int,
        default=None,
        help="Optional global RNG seed for reproducibility.",
    )
    return p.parse_args()


args = parse_args()
os.makedirs(args.outdir, exist_ok=True)


# ==============================================================
# FIXED PATHS
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge"
BASE_NOISE_PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise"

# Posterior seed file in wedge project
POSTERIOR_SEED_FILE = f"{PROJECT_DIR}/posterior_seed.npy"

# Fixed posterior noise cube (provided by user)
NOISE_FILE = (
    f"{BASE_NOISE_PROJECT_DIR}/posterior_data/"
    "AAstar_noise21cm_cube_seed1689456004_posterior.npz"
)

# Observed wedge-filtered NN-subtracted target PS (provided by user)
OBS_PS_FILE = (
    f"{PROJECT_DIR}/target_data/power_spectra/"
    "delta2_nn_sub_target_wedge_filtered_ms_noisy_temp_map.npy"
)

# Mean wedge-filtered noise PS for subtraction (provided by user)
NOISE_PS_MEAN_FILE = f"{PROJECT_DIR}/noise_maps/PS_noise_maps_wedge_filtered_mean.npy"

# Covariance matrix for wedge PS likelihood (requested)
COV_MATRIX_FILE = f"{PROJECT_DIR}/cov_matrix/covariance_matrix_ps_wedge.npy"


# ==============================================================
# COSMOLOGY & SIMULATION PARAMETERS
# ==============================================================

box = 128.0
dx = 1.0
z_target = 7.0
omega_m = 0.308
omega_l = 1.0 - omega_m
omega_b = 0.0482
h = 0.678
sigma_8 = 0.829
ns = 0.961
ngrid = 64
scaledist = 1e-3

# Fiducial parameters (truth marker)
theta_fid = np.array([0.54, 9.0], dtype=np.float64)  # [Q^M_HII, log10Mmin]

# Prior bounds
QHII_min, QHII_max = 0.1, 0.95
log10Mmin_min, log10Mmin_max = 7.5, 10.5


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


logger = setup_logging(os.path.join(args.outdir, "mcmc_ps_nnsub_wedge_run.log"))

logger.info("=" * 80)
logger.info("MCMC WITH WEDGE-FILTERED NOISE-SUBTRACTED POWER SPECTRUM - STARTED")
logger.info("=" * 80)

if args.global_seed is not None:
    np.random.seed(args.global_seed)
    logger.info("Set numpy global seed: %d", args.global_seed)


# ==============================================================
# WEDGE FILTER HELPERS
# ==============================================================

def compute_wedge_slope(z, om, ol):
    """Compute wedge slope C = x(z) H(z) / (c (1+z))."""

    def efunc(zp):
        return np.sqrt(om * (1.0 + zp) ** 3 + ol)

    integral_z, _ = quad(lambda zp: 1.0 / efunc(zp), 0.0, z)
    return integral_z * efunc(z) / (1.0 + z)


def get_3d_k_grids(ng, box_size):
    """Generate k_parallel and k_perp grids in np.fft.fftn ordering."""
    k1d = np.fft.fftfreq(ng, d=box_size / ng) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_par = np.abs(kz)
    k_perp = np.sqrt(kx ** 2 + ky ** 2)
    return k_par, k_perp


def apply_wedge_filter(map_3d, wedge_mask):
    """Apply wedge filtering in Fourier space and return real-space map."""
    ft = np.fft.fftn(map_3d)
    ft[~wedge_mask] = 0.0
    return np.fft.ifftn(ft).real.astype(np.float32, copy=False)


wedge_C = float(args.wedge_C) if args.wedge_C is not None else compute_wedge_slope(z_target, omega_m, omega_l)
k_par_3d, k_perp_3d = get_3d_k_grids(ngrid, box)
wedge_mask = k_par_3d > wedge_C * k_perp_3d
wedge_kept_frac = float(np.count_nonzero(wedge_mask) / wedge_mask.size)

logger.info("Wedge slope C: %.6f", wedge_C)
logger.info("Wedge mask keeps %.2f%% Fourier cells", 100.0 * wedge_kept_frac)


# ==============================================================
# LOAD INPUT SUMMARIES
# ==============================================================

if not os.path.exists(NOISE_PS_MEAN_FILE):
    logger.error("Mean wedge-noise PS file not found: %s", NOISE_PS_MEAN_FILE)
    sys.exit(1)
noise_ps_mean = np.load(NOISE_PS_MEAN_FILE).astype(np.float64)
logger.info("Loaded mean wedge-noise PS: %s", NOISE_PS_MEAN_FILE)
logger.info("Mean wedge-noise PS shape: %s", noise_ps_mean.shape)

if not os.path.exists(OBS_PS_FILE):
    logger.error("Observed wedge NN-subtracted PS file not found: %s", OBS_PS_FILE)
    sys.exit(1)
ps_obs = np.load(OBS_PS_FILE).astype(np.float64)
logger.info("Loaded observed wedge NN-subtracted PS: %s", OBS_PS_FILE)
logger.info("Observed PS shape: %s", ps_obs.shape)

if ps_obs.shape != noise_ps_mean.shape:
    logger.error(
        "Shape mismatch: observed PS %s vs mean wedge-noise PS %s",
        ps_obs.shape,
        noise_ps_mean.shape,
    )
    sys.exit(1)


# ==============================================================
# LOAD FIXED POSTERIOR SNAPSHOT + NOISE CUBE
# ==============================================================

def resolve_snapshot_paths(project_dir, posterior_seed):
    """Resolve snapshot path robustly across common output folder names."""
    candidates = [
        (
            f"{project_dir}/output_posterior_seed_{posterior_seed}/snap_000",
            f"{project_dir}/output_posterior_seed_{posterior_seed}",
        ),
        (
            f"{project_dir}/output_posterior/snap_000",
            f"{project_dir}/output_posterior",
        ),
    ]
    for snap_path, outpath in candidates:
        if os.path.exists(snap_path):
            return snap_path, outpath
    return None, None


if not os.path.exists(POSTERIOR_SEED_FILE):
    logger.error("Posterior seed file not found: %s", POSTERIOR_SEED_FILE)
    sys.exit(1)

posterior_seed = int(np.load(POSTERIOR_SEED_FILE)[0])
logger.info("Loaded posterior seed: %d", posterior_seed)

snap_path, outpath = resolve_snapshot_paths(PROJECT_DIR, posterior_seed)
if snap_path is None:
    logger.error(
        "Could not find posterior snapshot. Tried output_posterior_seed_<seed>/snap_000 "
        "and output_posterior/snap_000 under %s",
        PROJECT_DIR,
    )
    sys.exit(1)

logger.info("Using posterior snapshot from: %s", snap_path)

if not os.path.exists(NOISE_FILE):
    logger.error("Fixed posterior noise file not found: %s", NOISE_FILE)
    sys.exit(1)

noise_data = np.load(NOISE_FILE)
if "noisecube_21cm" in noise_data:
    noise_cube = noise_data["noisecube_21cm"].astype(np.float32)
elif "noise_cube" in noise_data:
    noise_cube = noise_data["noise_cube"].astype(np.float32)
else:
    logger.error("Could not find noise cube key in: %s", NOISE_FILE)
    sys.exit(1)

logger.info("Loaded fixed noise cube: %s", NOISE_FILE)
logger.info("Noise cube shape: %s", noise_cube.shape)


# ==============================================================
# INITIALIZE SIMULATION OBJECTS ONCE
# ==============================================================

logger.info("Initializing simulation objects from fixed posterior snapshot...")

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

logger.info("Simulation objects initialized successfully.")


# ==============================================================
# K-BINS SETUP
# ==============================================================

matter_data.initialize_powspec()
k_edges, k_bins = matter_data.set_k_edges(nbins=args.nbins, log_bins=True)
logger.info("k_edges: %s", k_edges)
logger.info("k_bins: %s", k_bins)

if len(k_bins) != len(ps_obs):
    logger.error(
        "k-bin count mismatch: script computed %d bins, observed summary has %d entries. "
        "Pass matching --nbins.",
        len(k_bins),
        len(ps_obs),
    )
    sys.exit(1)


# ==============================================================
# FORWARD MODEL: THETA -> WEDGE NN-SUB SUMMARY
# ==============================================================

def compute_dimensionless_ps_from_Tb(tb_map_ms_wedge):
    """Compute Delta^2(k) from wedge-filtered mean-subtracted map."""
    powspec_binned, kount = ionization_map.get_binned_powspec(
        tb_map_ms_wedge,
        k_edges,
        convolve=False,
        units="",
        bin_weighted=False,
    )

    valid = kount > 0
    k_valid = k_bins[valid]
    pk_valid = powspec_binned[valid]

    delta2 = (k_valid ** 3) * pk_valid / (2.0 * np.pi ** 2)
    return delta2.astype(np.float64)


def simulate_wedge_nnsub_ps(qhii_val, log10mmin_val):
    """Simulate wedge-filtered NN-subtracted PS for one parameter point."""
    fcoll_arr = matter_data.get_fcoll_for_Mmin(log10mmin_val)
    mass_weighted_fcoll = np.mean(fcoll_arr * (1.0 + matter_data.densitycontr_arr))

    if mass_weighted_fcoll <= 0:
        raise RuntimeError("Mass-weighted fcoll is non-positive.")

    zeta = qhii_val / mass_weighted_fcoll
    qi_arr = ionization_map.get_qi(zeta * fcoll_arr)
    xHI_arr = 1.0 - qi_arr

    tb_bar = 27.0 * (omega_b * h ** 2 / 0.023) * np.sqrt(
        0.15 / (omega_m * h ** 2) * (1.0 + z_target) / 10.0
    )

    tb_signal = tb_bar * xHI_arr * (1.0 + matter_data.densitycontr_arr)
    tb_noisy = tb_signal + noise_cube

    # Exactly matches requested order before PS extraction.
    tb_ms = tb_noisy - np.mean(tb_noisy)
    tb_wedge = apply_wedge_filter(tb_ms, wedge_mask)

    delta2_raw_wedge = compute_dimensionless_ps_from_Tb(tb_wedge)
    delta2_nnsub_wedge = delta2_raw_wedge - noise_ps_mean

    return delta2_nnsub_wedge


# ==============================================================
# COVARIANCE (FROM FILE, REGULARIZED)
# ==============================================================

if not os.path.exists(COV_MATRIX_FILE):
    logger.error("Covariance matrix file not found: %s", COV_MATRIX_FILE)
    sys.exit(1)

Sigma = np.load(COV_MATRIX_FILE).astype(np.float64)

if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
    logger.error("Covariance matrix must be square. Got shape: %s", str(Sigma.shape))
    sys.exit(1)

if Sigma.shape[0] != len(ps_obs):
    logger.error(
        "Covariance/summary dimension mismatch: cov=%s, summary length=%d",
        str(Sigma.shape),
        len(ps_obs),
    )
    sys.exit(1)

eps = 1e-6 * np.trace(Sigma) / Sigma.shape[0]
Sigma += eps * np.eye(Sigma.shape[0])

Sigma_inv = np.linalg.inv(Sigma)
logdet_Sigma = np.linalg.slogdet(Sigma)[1]

np.save(os.path.join(args.outdir, "covariance_matrix_used.npy"), Sigma)
logger.info("Loaded covariance matrix: %s", COV_MATRIX_FILE)
logger.info("Covariance shape: %s", str(Sigma.shape))
logger.info("Applied covariance regularization eps=%.6e", eps)


# ==============================================================
# PRIOR / LIKELIHOOD / POSTERIOR
# ==============================================================

def log_prior(params):
    qhii, log10mmin = params
    if QHII_min <= qhii <= QHII_max and log10Mmin_min <= log10mmin <= log10Mmin_max:
        return -np.log(QHII_max - QHII_min) - np.log(log10Mmin_max - log10Mmin_min)
    return -np.inf


def log_likelihood(params):
    qhii, log10mmin = params
    try:
        ps_theta = simulate_wedge_nnsub_ps(qhii, log10mmin)
    except RuntimeError:
        return -np.inf

    d = ps_obs - ps_theta
    chi2 = d @ Sigma_inv @ d
    return -0.5 * (chi2 + logdet_Sigma + len(ps_obs) * np.log(2.0 * np.pi))


def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ==============================================================
# RUN EMCEE
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

p0 = np.zeros((nwalkers, ndim), dtype=np.float64)
for i in range(nwalkers):
    while True:
        proposal = start_center + start_spread * np.random.randn(ndim)
        if (
            QHII_min < proposal[0] < QHII_max
            and log10Mmin_min < proposal[1] < log10Mmin_max
        ):
            p0[i] = proposal
            break

ncores = args.ncores
if ncores == 0:
    ncores = os.cpu_count() or 1
logger.info("Using %d CPU cores for MCMC", ncores)

if ncores > 1:
    pool = Pool(ncores)
    logger.info("Initialized multiprocessing pool with %d workers.", ncores)
else:
    pool = None
    logger.info("Running in serial mode.")

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)

logger.info("Running burn-in (%d steps)...", nburn)
state = sampler.run_mcmc(p0, nburn, progress=True)
sampler.reset()

logger.info("Running production (%d steps)...", nsteps)
sampler.run_mcmc(state, nsteps, progress=True)

mcmc_chain = sampler.get_chain(discard=0, flat=True)
logger.info("MCMC completed. Chain shape: %s", mcmc_chain.shape)

if pool is not None:
    pool.close()
    pool.join()


# ==============================================================
# SAVE RESULTS + PLOTS
# ==============================================================

chain_file = os.path.join(args.outdir, "mcmc_ps_nnsub_wedge_chain.npy")
np.save(chain_file, mcmc_chain)
logger.info("Saved chain: %s", chain_file)

acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
logger.info("Mean acceptance fraction: %.3f", acceptance_fraction)

try:
    autocorr_time = sampler.get_autocorr_time(quiet=True)
except Exception as exc:
    autocorr_time = np.array([np.nan, np.nan])
    logger.warning("Could not compute autocorrelation time: %s", str(exc))


def credible_interval(x):
    return np.percentile(x, [16.0, 50.0, 84.0])


ci_results = np.array([credible_interval(mcmc_chain[:, i]) for i in range(ndim)])

results_file = os.path.join(args.outdir, "mcmc_ps_nnsub_wedge_results.npz")
np.savez(
    results_file,
    chain=mcmc_chain,
    true_params=theta_fid,
    credible_intervals=ci_results,
    observed_ps=ps_obs,
    noise_ps_mean=noise_ps_mean,
    k_bins=k_bins,
    wedge_C=wedge_C,
    acceptance_fraction=acceptance_fraction,
    autocorr_time=autocorr_time,
)
logger.info("Saved results: %s", results_file)

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
    plt.Line2D([], [], color="C1", label="MCMC Posterior (Wedge NN-sub PS)"),
    plt.Line2D([], [], color="red", linestyle="--", label="Truth"),
]
fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)

corner_file = os.path.join(args.outdir, "mcmc_ps_nnsub_wedge_corner.png")
plt.savefig(corner_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved corner plot: %s", corner_file)

logger.info("Generating trace plots...")
full_chain = sampler.get_chain()

fig, axes = plt.subplots(ndim, 1, figsize=(10, 6), sharex=True)
labels_plot = [r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(full_chain[:, :, i], alpha=0.3, linewidth=0.5)
    ax.axhline(theta_fid[i], color="red", linestyle="--", linewidth=1.5, label="Truth")
    ax.set_ylabel(labels_plot[i], fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Step", fontsize=12)
plt.tight_layout()

trace_file = os.path.join(args.outdir, "mcmc_ps_nnsub_wedge_trace.png")
plt.savefig(trace_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved trace plot: %s", trace_file)

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
ax.set_xlabel(r"$\log_{10} M_{\min}$", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.legend(fontsize=10)
ax.set_title("Marginal Posterior: $\\log_{10} M_{\\min}$", fontsize=12)

plt.tight_layout()

marginal_file = os.path.join(args.outdir, "mcmc_ps_nnsub_wedge_marginals.png")
plt.savefig(marginal_file, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved marginal histograms: %s", marginal_file)

logger.info("=" * 80)
logger.info("MCMC WITH WEDGE-FILTERED NOISE-SUBTRACTED POWER SPECTRUM COMPLETED")
logger.info("=" * 80)
logger.info("Output directory: %s", args.outdir)
logger.info("Chain: %s", chain_file)
logger.info("Corner: %s", corner_file)
logger.info("Trace: %s", trace_file)
logger.info("Marginals: %s", marginal_file)
logger.info("Results: %s", results_file)
