#!/usr/bin/env python
# coding: utf-8

"""Parallel SNPE pipeline for wedge-filtered power-spectrum summaries.

Per simulation, preprocessing follows:
signal + noise -> mean subtraction -> wedge filtering -> Delta^2(k) -> Delta^2(k)-<NN>(k).

Simulation is parallelized with multiprocessing workers; SNPE training remains
sequential per round.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import shutil
from scipy.integrate import quad
import sys
import traceback

import numpy as np
from tqdm import tqdm

import script
from script import two_lpt


# ==============================================================
# FIXED PATHS
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge"
BASE_NOISE_PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise"

TARGET_SEED = 1259935638
TARGET_PS_RAW_FILE = (
    f"{PROJECT_DIR}/target_data/power_spectra/"
    "delta2_target_wedge_filtered_ms_noisy_temp_map.npy"
)
TARGET_PS_NNSUB_FILE = (
    f"{PROJECT_DIR}/target_data/power_spectra/"
    "delta2_nn_sub_target_wedge_filtered_ms_noisy_temp_map.npy"
)
NOISE_PS_MEAN_FILE = (
    f"{PROJECT_DIR}/noise_maps/PS_noise_maps_wedge_filtered_mean.npy"
)

NOISE_DIR = f"{BASE_NOISE_PROJECT_DIR}/noise_maps/noise_maps_sbi"

SEED_DIR = BASE_NOISE_PROJECT_DIR
MUSIC_EXEC = "/scratch/subhankar/software/music/build/MUSIC"


# ==============================================================
# COSMOLOGY
# ==============================================================

BOX = 128.0
DX = 1.0
Z_TARGET = 7.0
OMEGA_M = 0.308
OMEGA_L = 1 - OMEGA_M
OMEGA_B = 0.0482
HUBBLE = 0.678
SIGMA_8 = 0.829
NS = 0.961
NGRID = 64
SCALEDIST = 1e-3

PRIOR_Q_MIN = 0.1
PRIOR_Q_MAX = 0.95
PRIOR_M_MIN = 7.5
PRIOR_M_MAX = 10.5

THETA_TRUE = np.array([0.54, 9.0], dtype=np.float32)


# ==============================================================
# WEDGE FILTER
# ==============================================================


def compute_wedge_slope(z, omega_m, omega_l):
    """Compute foreground wedge slope C = x(z) H(z) / (c (1+z))."""

    def efunc(zp):
        return np.sqrt(omega_m * (1.0 + zp) ** 3 + omega_l)

    integral_z, _ = quad(lambda zp: 1.0 / efunc(zp), 0.0, z)
    return integral_z * efunc(z) / (1.0 + z)


def get_3d_k_grids(ngrid, box):
    """Generate k_parallel and k_perp grids matching np.fft.fftn ordering."""
    k1d = np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_par = np.abs(kz)
    k_perp = np.sqrt(kx ** 2 + ky ** 2)
    return k_par, k_perp


def apply_wedge_filter(map_3d, wedge_mask):
    """Apply wedge filtering in Fourier space and return filtered real-space map."""
    ft = np.fft.fftn(map_3d)
    ft[~wedge_mask] = 0.0
    return np.fft.ifftn(ft).real.astype(np.float32, copy=False)


# ==============================================================
# ARGUMENTS / LOGGING
# ==============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel SBI with default MAF for power-spectrum summaries and "
            "round-wise health checks."
        )
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sims_per_round", type=int, default=2000)
    parser.add_argument("--nbins", type=int, default=10)
    parser.add_argument(
        "--summary_mode",
        type=str,
        default="nnsub",
        choices=("nnsub",),
        help="nnsub: Delta^2(k)-<NN>(k) after wedge filtering.",
    )
    parser.add_argument(
        "--target_ps_file",
        type=str,
        default=None,
        help="Optional explicit target PS .npy path.",
    )
    parser.add_argument(
        "--noise_ps_mean_file",
        type=str,
        default=NOISE_PS_MEAN_FILE,
        help="Mean noise PS file used when --summary_mode nnsub.",
    )
    parser.add_argument(
        "--wedge_C",
        type=float,
        default=None,
        help="Explicit wedge slope C. If omitted, compute from z/omega_m/omega_l.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of worker processes for simulation stage. 0 = all CPUs.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Multiprocessing imap chunk size.",
    )
    parser.add_argument(
        "--mp_start_method",
        type=str,
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible seed/noise assignment.",
    )
    parser.add_argument(
        "--global_seed",
        type=int,
        default=None,
        help="Optional global seed for python/numpy/torch RNGs.",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary MUSIC run folders for debugging.",
    )
    parser.add_argument(
        "--health_edge_q_margin",
        type=float,
        default=0.03,
        help="Q edge margin from prior bounds for edge-fraction check.",
    )
    parser.add_argument(
        "--health_edge_m_margin",
        type=float,
        default=0.15,
        help="log10Mmin edge margin from prior bounds for edge-fraction check.",
    )
    parser.add_argument(
        "--health_theta_edge_frac_max",
        type=float,
        default=0.05,
        help="Max acceptable edge fraction of proposal samples (round >=2).",
    )
    parser.add_argument(
        "--health_std_growth_factor",
        type=float,
        default=1.25,
        help="Max allowed round-to-round growth factor for proposal std.",
    )
    parser.add_argument(
        "--health_q_std_max",
        type=float,
        default=0.08,
        help="Max acceptable absolute std(QMHII) for proposal samples (round >=2).",
    )
    parser.add_argument(
        "--health_m_std_max",
        type=float,
        default=0.50,
        help="Max acceptable absolute std(log10Mmin) for proposal samples (round >=2).",
    )
    parser.add_argument(
        "--health_dist_growth_factor",
        type=float,
        default=1.15,
        help="Max allowed growth factor for median summary distance to x_obs.",
    )
    parser.add_argument(
        "--health_posterior_samples",
        type=int,
        default=4000,
        help="Number of posterior samples per round for health diagnostics.",
    )
    parser.add_argument(
        "--health_post_width_growth_factor",
        type=float,
        default=1.25,
        help="Max allowed growth factor for posterior width from one round to next.",
    )
    parser.add_argument(
        "--health_post_q_width_max",
        type=float,
        default=0.20,
        help="Max acceptable posterior 68%% width for QMHII (round >=2).",
    )
    parser.add_argument(
        "--health_post_m_width_max",
        type=float,
        default=1.00,
        help="Max acceptable posterior 68%% width for log10Mmin (round >=2).",
    )
    parser.add_argument(
        "--health_post_edge_frac_max",
        type=float,
        default=0.02,
        help="Max acceptable posterior edge fraction at x_obs (round >=2).",
    )
    parser.add_argument(
        "--health_neg_bin_frac_max",
        type=float,
        default=0.95,
        help=(
            "Max acceptable median fraction of negative PS bins "
            "(used only for nnsub mode, round >=2)."
        ),
    )
    parser.add_argument(
        "--abort_on_health_fail",
        action="store_true",
        help="Abort immediately when any health check fails.",
    )
    return parser.parse_args()


def setup_logging(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(logpath, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


# ==============================================================
# WORKER HELPERS
# ==============================================================

_WORKER_CFG = {}
_WORKER_PS_HELPER = None
_WORKER_NOISE_PS_MEAN = None


def build_k_edges_kbins(ngrid, box, nbins, log_bins=True):
    """Build k-edges and k-bins using script.set_k_edges-compatible convention."""
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

    return k_edges.astype(np.float64), k_bins.astype(np.float64)


def init_worker(worker_cfg):
    global _WORKER_CFG
    global _WORKER_PS_HELPER
    global _WORKER_NOISE_PS_MEAN

    _WORKER_CFG = worker_cfg

    dummy_density = np.zeros(
        (worker_cfg["ngrid"], worker_cfg["ngrid"], worker_cfg["ngrid"]),
        dtype=np.float32,
    )
    _WORKER_PS_HELPER = script.ionization_map(
        matter_fields=None,
        densitycontr_arr=dummy_density,
        box=worker_cfg["box"],
        z=worker_cfg["z_target"],
        omega_m=worker_cfg["omega_m"],
        h=worker_cfg["hubble"],
    )

    _WORKER_NOISE_PS_MEAN = None
    if worker_cfg["summary_mode"] == "nnsub":
        _WORKER_NOISE_PS_MEAN = np.load(worker_cfg["noise_ps_mean_file"]).astype(
            np.float32
        )


def load_noise_cube(noise_file):
    with np.load(noise_file) as noise_data:
        if "noisecube_21cm" in noise_data:
            return noise_data["noisecube_21cm"].astype(np.float32, copy=False)
        if "noise_cube" in noise_data:
            return noise_data["noise_cube"].astype(np.float32, copy=False)
    raise RuntimeError(f"Noise cube key not found in file: {noise_file}")


def extract_noise_seed(noise_file):
    match = re.search(r"seed(\d+)", os.path.basename(noise_file))
    if match:
        return int(match.group(1))
    return None


def compute_dimensionless_ps(temp_map_mk):
    cfg = _WORKER_CFG
    powspec_binned, kount = _WORKER_PS_HELPER.get_binned_powspec(
        temp_map_mk,
        cfg["k_edges"],
        convolve=False,
        units="",
        bin_weighted=False,
    )

    delta2 = np.zeros(cfg["nbins"], dtype=np.float64)
    valid = kount > 0
    delta2[valid] = (
        (cfg["k_bins"][valid] ** 3) * powspec_binned[valid] / (2.0 * np.pi**2)
    )
    return delta2.astype(np.float32)


def simulate_temp_map_worker(qhii_val, log10mmin_val, seed):
    cfg = _WORKER_CFG
    outpath = os.path.join(cfg["temp_root"], f"temp_seed_{seed}")
    snap_path = f"{outpath}/snap_000"

    os.makedirs(outpath, exist_ok=True)

    try:
        two_lpt.run_music(
            cfg["music_exec"],
            cfg["box"],
            [cfg["z_target"]],
            seed,
            outpath,
            "snap",
            cfg["dx"],
            cfg["omega_m"],
            cfg["omega_l"],
            cfg["omega_b"],
            cfg["hubble"],
            cfg["sigma_8"],
            cfg["ns"],
        )

        sim_data = script.default_simulation_data(
            snap_path,
            outpath,
            sigma_8=cfg["sigma_8"],
            ns=cfg["ns"],
            omega_b=cfg["omega_b"],
            scaledist=cfg["scaledist"],
        )

        matter_data = script.matter_fields(
            sim_data,
            cfg["ngrid"],
            outpath,
            overwrite_files=False,
        )

        ion_map = script.ionization_map(matter_data)

        fcoll = matter_data.get_fcoll_for_Mmin(log10mmin_val)
        mass_weighted = np.mean(fcoll * (1.0 + matter_data.densitycontr_arr))
        if mass_weighted <= 0:
            raise RuntimeError(
                f"Non-positive mass-weighted fcoll for log10Mmin={log10mmin_val}."
            )

        zeta = qhii_val / mass_weighted
        qi = ion_map.get_qi(zeta * fcoll)
        xhi = 1.0 - qi

        tb_bar = 27.0 * (cfg["omega_b"] * cfg["hubble"] ** 2 / 0.023) * np.sqrt(
            0.15 / (cfg["omega_m"] * cfg["hubble"] ** 2) * (1.0 + cfg["z_target"]) / 10.0
        )

        tb_signal = tb_bar * xhi * (1.0 + matter_data.densitycontr_arr)
        return tb_signal.astype(np.float32, copy=False)

    finally:
        shutil.rmtree(outpath, ignore_errors=True)


def worker_simulation(task):
    idx, qhii_val, log10mmin_val, sim_seed, noise_file = task
    try:
        tb_signal = simulate_temp_map_worker(qhii_val, log10mmin_val, sim_seed)
        noise_cube = load_noise_cube(noise_file)
        tb_sim = tb_signal + noise_cube
        tb_sim -= np.mean(tb_sim, dtype=np.float64)
        tb_sim = apply_wedge_filter(tb_sim, _WORKER_CFG["wedge_mask"])

        summary = compute_dimensionless_ps(tb_sim)
        if _WORKER_CFG["summary_mode"] == "nnsub":
            summary = summary - _WORKER_NOISE_PS_MEAN

        if not np.all(np.isfinite(summary)):
            raise RuntimeError("Non-finite entries found in simulated summary.")

        return {
            "idx": idx,
            "summary": summary.astype(np.float32, copy=False),
            "neg_bin_frac": float(np.mean(summary < 0.0)),
            "sim_seed": sim_seed,
            "noise_file": noise_file,
            "error": None,
        }
    except Exception:
        return {
            "idx": idx,
            "summary": None,
            "neg_bin_frac": None,
            "sim_seed": sim_seed,
            "noise_file": noise_file,
            "error": traceback.format_exc(),
        }


# ==============================================================
# SEED / HEALTH HELPERS
# ==============================================================


def load_excluded_seeds(seed_dir):
    excluded = set()
    training_seed_file = f"{seed_dir}/random_seeds_training.npy"
    validation_seed_file = f"{seed_dir}/random_seeds_validation.npy"

    if not os.path.exists(training_seed_file):
        raise FileNotFoundError(f"Missing training seeds file: {training_seed_file}")
    if not os.path.exists(validation_seed_file):
        raise FileNotFoundError(f"Missing validation seeds file: {validation_seed_file}")

    excluded.update(np.load(training_seed_file).tolist())
    excluded.update(np.load(validation_seed_file).tolist())
    excluded.add(TARGET_SEED)
    return excluded


def draw_unique_seeds(rng, excluded, used, count):
    seeds = []
    while len(seeds) < count:
        candidates = rng.integers(0, 2**31 - 1, size=max(2 * count, 128))
        for candidate in candidates:
            seed = int(candidate)
            if seed in excluded or seed in used:
                continue
            used.add(seed)
            seeds.append(seed)
            if len(seeds) == count:
                break
    return seeds


def edge_fraction(q_vals, m_vals, q_margin, m_margin):
    q_low = PRIOR_Q_MIN + q_margin
    q_high = PRIOR_Q_MAX - q_margin
    m_low = PRIOR_M_MIN + m_margin
    m_high = PRIOR_M_MAX - m_margin
    on_edge = (
        (q_vals < q_low)
        | (q_vals > q_high)
        | (m_vals < m_low)
        | (m_vals > m_high)
    )
    return float(np.mean(on_edge))


def width_68(values):
    p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(p84 - p16), float(p16), float(p50), float(p84)


# ==============================================================
# MAIN
# ==============================================================


def main():
    import corner
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    from sbi.utils.get_nn_models import posterior_nn

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logger = setup_logging(os.path.join(args.outdir, "run.log"))

    logger.info("=" * 80)
    logger.info("PARALLEL WEDGE-FILTERED POWER-SPECTRUM SBI PIPELINE STARTED")
    logger.info("=" * 80)

    if args.rounds < 1:
        logger.error("--rounds must be >= 1")
        sys.exit(1)
    if args.sims_per_round < 1:
        logger.error("--sims_per_round must be >= 1")
        sys.exit(1)
    if args.nbins < 2:
        logger.error("--nbins must be >= 2")
        sys.exit(1)
    if args.chunksize < 1:
        logger.error("--chunksize must be >= 1")
        sys.exit(1)
    if args.health_posterior_samples < 200:
        logger.error("--health_posterior_samples must be >= 200")
        sys.exit(1)
    if args.health_std_growth_factor <= 1.0:
        logger.error("--health_std_growth_factor must be > 1")
        sys.exit(1)
    if args.health_dist_growth_factor <= 1.0:
        logger.error("--health_dist_growth_factor must be > 1")
        sys.exit(1)
    if args.health_post_width_growth_factor <= 1.0:
        logger.error("--health_post_width_growth_factor must be > 1")
        sys.exit(1)

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        logger.info("Global seed set to %d (python/numpy/torch).", args.global_seed)

    logger.info("Summary mode: %s", args.summary_mode)
    logger.info("Number of k-bins: %d", args.nbins)
    logger.info(
        "Health checks enabled: edge_frac<=%.3f, std_growth<=%.2f, q_std<=%.3f, "
        "m_std<=%.3f, dist_growth<=%.2f, post_width_growth<=%.2f, post_q_width<=%.3f, "
        "post_m_width<=%.3f, post_edge_frac<=%.3f, neg_bin_frac<=%.3f (nnsub only)",
        args.health_theta_edge_frac_max,
        args.health_std_growth_factor,
        args.health_q_std_max,
        args.health_m_std_max,
        args.health_dist_growth_factor,
        args.health_post_width_growth_factor,
        args.health_post_q_width_max,
        args.health_post_m_width_max,
        args.health_post_edge_frac_max,
        args.health_neg_bin_frac_max,
    )

    k_edges, k_bins = build_k_edges_kbins(NGRID, BOX, args.nbins, log_bins=True)
    logger.info("k_edges: %s", np.array2string(k_edges, precision=6))
    logger.info("k_bins: %s", np.array2string(k_bins, precision=6))

    target_path = args.target_ps_file
    if target_path is None:
        target_path = TARGET_PS_NNSUB_FILE if args.summary_mode == "nnsub" else TARGET_PS_RAW_FILE

    if not os.path.exists(target_path):
        logger.error("Target power spectrum not found: %s", target_path)
        sys.exit(1)
    target_ps = np.load(target_path).astype(np.float32)
    if target_ps.ndim != 1 or target_ps.shape[0] != args.nbins:
        logger.error(
            "Target PS shape mismatch: expected (%d,), got %s",
            args.nbins,
            str(target_ps.shape),
        )
        sys.exit(1)
    logger.info("Loaded target PS: %s", target_path)
    logger.info("Target PS shape: %s", str(target_ps.shape))

    noise_ps_mean = None
    if args.summary_mode == "nnsub":
        if not os.path.exists(args.noise_ps_mean_file):
            logger.error(
                "Mean noise PS file not found for nnsub mode: %s",
                args.noise_ps_mean_file,
            )
            sys.exit(1)
        noise_ps_mean = np.load(args.noise_ps_mean_file).astype(np.float32)
        if noise_ps_mean.ndim != 1 or noise_ps_mean.shape[0] != args.nbins:
            logger.error(
                "Mean noise PS shape mismatch: expected (%d,), got %s",
                args.nbins,
                str(noise_ps_mean.shape),
            )
            sys.exit(1)
        logger.info("Loaded mean noise PS: %s", args.noise_ps_mean_file)

    # ----------------------------------------------------------
    # SBI SETUP
    # ----------------------------------------------------------
    prior = BoxUniform(
        low=torch.tensor([PRIOR_Q_MIN, PRIOR_M_MIN], dtype=torch.float32),
        high=torch.tensor([PRIOR_Q_MAX, PRIOR_M_MAX], dtype=torch.float32),
    )
    inference = SNPE(prior=prior, density_estimator=posterior_nn(model="maf"))
    logger.info(
        "Using default sbi==0.22.0 MAF settings: hidden_features=50, num_blocks=2, "
        "num_transforms=5."
    )

    proposal = prior
    x_obs = torch.tensor(target_ps, dtype=torch.float32)

    # ----------------------------------------------------------
    # SEED / NOISE HANDLING
    # ----------------------------------------------------------
    try:
        excluded = load_excluded_seeds(SEED_DIR)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    used = set()

    if not os.path.isdir(NOISE_DIR):
        logger.error("Noise directory not found: %s", NOISE_DIR)
        sys.exit(1)

    noise_files = sorted(
        [
            os.path.join(NOISE_DIR, fname)
            for fname in os.listdir(NOISE_DIR)
            if fname.endswith(".npz")
        ]
    )
    if len(noise_files) == 0:
        logger.error("No noise files found in: %s", NOISE_DIR)
        sys.exit(1)

    rng_seed = args.random_seed if args.random_seed is not None else args.global_seed
    rng = np.random.default_rng(rng_seed)
    logger.info("Seed/noise assignment RNG seed: %s", str(rng_seed))
    rng.shuffle(noise_files)

    total_required = args.rounds * args.sims_per_round
    if len(noise_files) < total_required:
        logger.error(
            "Not enough noise files for no-reuse policy: required=%d, found=%d",
            total_required,
            len(noise_files),
        )
        sys.exit(1)

    noise_index = 0

    # ----------------------------------------------------------
    # PARALLEL SETUP
    # ----------------------------------------------------------
    requested_workers = args.n_workers
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    n_workers = max(1, min(requested_workers, args.sims_per_round))

    logger.info("Requested workers: %d", args.n_workers)
    logger.info("Using workers: %d", n_workers)
    logger.info("Multiprocessing start method: %s", args.mp_start_method)
    logger.info("Multiprocessing chunksize: %d", args.chunksize)

    temp_root = os.path.join(args.outdir, "temp_music_runs")
    os.makedirs(temp_root, exist_ok=True)

    wedge_slope = (
        float(args.wedge_C)
        if args.wedge_C is not None
        else compute_wedge_slope(Z_TARGET, OMEGA_M, OMEGA_L)
    )
    k_par_3d, k_perp_3d = get_3d_k_grids(NGRID, BOX)
    wedge_mask = k_par_3d > wedge_slope * k_perp_3d
    wedge_kept_frac = float(np.count_nonzero(wedge_mask) / wedge_mask.size)
    logger.info("Wedge slope C = %.6f", wedge_slope)
    logger.info("Wedge mask keeps %.2f%% of Fourier cells", 100.0 * wedge_kept_frac)

    worker_cfg = {
        "temp_root": temp_root,
        "music_exec": MUSIC_EXEC,
        "box": BOX,
        "dx": DX,
        "z_target": Z_TARGET,
        "omega_m": OMEGA_M,
        "omega_l": OMEGA_L,
        "omega_b": OMEGA_B,
        "hubble": HUBBLE,
        "sigma_8": SIGMA_8,
        "ns": NS,
        "ngrid": NGRID,
        "scaledist": SCALEDIST,
        "summary_mode": args.summary_mode,
        "nbins": args.nbins,
        "k_edges": k_edges,
        "k_bins": k_bins,
        "noise_ps_mean_file": args.noise_ps_mean_file,
        "wedge_mask": wedge_mask,
    }

    pool = None
    if n_workers > 1:
        ctx = mp.get_context(args.mp_start_method)
        pool = ctx.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(worker_cfg,),
        )
        logger.info("Worker pool initialized.")
    else:
        init_worker(worker_cfg)
        logger.info("Running in serial mode for simulations.")

    # ----------------------------------------------------------
    # SNPE ROUNDS + HEALTH CHECKS
    # ----------------------------------------------------------
    health_records = []
    unhealthy_rounds = []
    prev_metrics = None
    abort_error = None
    posterior = None

    try:
        for round_idx in range(args.rounds):
            round_num = round_idx + 1
            logger.info("Starting round %d/%d", round_num, args.rounds)

            theta = proposal.sample((args.sims_per_round,))
            theta_np = theta.cpu().numpy()
            theta_q = theta_np[:, 0]
            theta_m = theta_np[:, 1]

            theta_q_std = float(np.std(theta_q))
            theta_m_std = float(np.std(theta_m))
            theta_q_mean = float(np.mean(theta_q))
            theta_m_mean = float(np.mean(theta_m))
            theta_edge_frac = edge_fraction(
                theta_q,
                theta_m,
                q_margin=args.health_edge_q_margin,
                m_margin=args.health_edge_m_margin,
            )

            sim_seeds = draw_unique_seeds(
                rng,
                excluded,
                used,
                args.sims_per_round,
            )
            round_noise_files = noise_files[noise_index : noise_index + args.sims_per_round]
            noise_index += args.sims_per_round

            tasks = []
            for i in range(args.sims_per_round):
                qhii_val = float(theta_np[i, 0])
                log10mmin_val = float(theta_np[i, 1])
                sim_seed = int(sim_seeds[i])
                noise_file = round_noise_files[i]
                noise_seed = extract_noise_seed(noise_file)
                noise_seed_str = str(noise_seed) if noise_seed is not None else "unknown"
                logger.info(
                    "Round %d sim %d/%d: QMHII=%.6f | log10Mmin=%.6f | sim_seed=%d | noise_seed=%s",
                    round_num,
                    i + 1,
                    args.sims_per_round,
                    qhii_val,
                    log10mmin_val,
                    sim_seed,
                    noise_seed_str,
                )
                tasks.append((i, qhii_val, log10mmin_val, sim_seed, noise_file))

            summaries = np.empty((args.sims_per_round, args.nbins), dtype=np.float32)
            neg_bin_fracs = np.empty(args.sims_per_round, dtype=np.float32)

            if pool is not None:
                result_iter = pool.imap_unordered(
                    worker_simulation,
                    tasks,
                    chunksize=args.chunksize,
                )
            else:
                result_iter = map(worker_simulation, tasks)

            for result in tqdm(
                result_iter,
                total=args.sims_per_round,
                desc=f"Round {round_num}/{args.rounds}",
                leave=False,
            ):
                if result["error"] is not None:
                    logger.error(
                        "Worker failed: idx=%d sim_seed=%d noise_file=%s",
                        result["idx"],
                        result["sim_seed"],
                        result["noise_file"],
                    )
                    logger.error("Worker traceback:\n%s", result["error"])
                    raise RuntimeError("Simulation worker failed.")

                idx = result["idx"]
                summaries[idx] = result["summary"]
                neg_bin_fracs[idx] = result["neg_bin_frac"]

            if not np.all(np.isfinite(summaries)):
                raise RuntimeError("Non-finite summaries detected after simulation.")

            scale = np.maximum(np.abs(target_ps), 1.0)
            summary_dist = np.linalg.norm((summaries - target_ps[None, :]) / scale[None, :], axis=1)
            dist_median = float(np.median(summary_dist))
            dist_p16, dist_p84 = np.percentile(summary_dist, [16.0, 84.0])
            dist_p16 = float(dist_p16)
            dist_p84 = float(dist_p84)

            neg_bin_frac_median = float(np.median(neg_bin_fracs))
            neg_bin_frac_p84 = float(np.percentile(neg_bin_fracs, 84.0))

            x = torch.tensor(summaries, dtype=torch.float32)

            inference = inference.append_simulations(
                theta,
                x,
                proposal=proposal,
            )

            density_estimator = inference.train(
                force_first_round_loss=(round_idx == 0)
            )

            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_obs)

            post_samples = proposal.sample((args.health_posterior_samples,)).cpu().numpy()
            post_q = post_samples[:, 0]
            post_m = post_samples[:, 1]
            post_q_width, post_q16, post_q50, post_q84 = width_68(post_q)
            post_m_width, post_m16, post_m50, post_m84 = width_68(post_m)
            post_edge_frac = edge_fraction(
                post_q,
                post_m,
                q_margin=args.health_edge_q_margin,
                m_margin=args.health_edge_m_margin,
            )

            health_flags = []
            if round_idx >= 1:
                if theta_edge_frac > args.health_theta_edge_frac_max:
                    health_flags.append(
                        f"proposal_edge_frac={theta_edge_frac:.4f} > {args.health_theta_edge_frac_max:.4f}"
                    )
                if theta_q_std > args.health_q_std_max:
                    health_flags.append(
                        f"proposal_q_std={theta_q_std:.4f} > {args.health_q_std_max:.4f}"
                    )
                if theta_m_std > args.health_m_std_max:
                    health_flags.append(
                        f"proposal_m_std={theta_m_std:.4f} > {args.health_m_std_max:.4f}"
                    )
                if post_q_width > args.health_post_q_width_max:
                    health_flags.append(
                        f"posterior_q_width={post_q_width:.4f} > {args.health_post_q_width_max:.4f}"
                    )
                if post_m_width > args.health_post_m_width_max:
                    health_flags.append(
                        f"posterior_m_width={post_m_width:.4f} > {args.health_post_m_width_max:.4f}"
                    )
                if post_edge_frac > args.health_post_edge_frac_max:
                    health_flags.append(
                        f"posterior_edge_frac={post_edge_frac:.4f} > {args.health_post_edge_frac_max:.4f}"
                    )
                if (
                    args.summary_mode == "nnsub"
                    and neg_bin_frac_median > args.health_neg_bin_frac_max
                ):
                    health_flags.append(
                        f"median_negative_bin_frac={neg_bin_frac_median:.4f} > {args.health_neg_bin_frac_max:.4f}"
                    )

                if prev_metrics is not None:
                    if theta_q_std > prev_metrics["theta_q_std"] * args.health_std_growth_factor:
                        health_flags.append(
                            f"proposal_q_std growth {theta_q_std:.4f}/{prev_metrics['theta_q_std']:.4f} > {args.health_std_growth_factor:.2f}"
                        )
                    if theta_m_std > prev_metrics["theta_m_std"] * args.health_std_growth_factor:
                        health_flags.append(
                            f"proposal_m_std growth {theta_m_std:.4f}/{prev_metrics['theta_m_std']:.4f} > {args.health_std_growth_factor:.2f}"
                        )
                    if dist_median > prev_metrics["dist_median"] * args.health_dist_growth_factor:
                        health_flags.append(
                            f"summary_distance growth {dist_median:.4f}/{prev_metrics['dist_median']:.4f} > {args.health_dist_growth_factor:.2f}"
                        )
                    if post_q_width > prev_metrics["post_q_width"] * args.health_post_width_growth_factor:
                        health_flags.append(
                            f"posterior_q_width growth {post_q_width:.4f}/{prev_metrics['post_q_width']:.4f} > {args.health_post_width_growth_factor:.2f}"
                        )
                    if post_m_width > prev_metrics["post_m_width"] * args.health_post_width_growth_factor:
                        health_flags.append(
                            f"posterior_m_width growth {post_m_width:.4f}/{prev_metrics['post_m_width']:.4f} > {args.health_post_width_growth_factor:.2f}"
                        )

            health_status = "healthy" if len(health_flags) == 0 else "unhealthy"
            logger.info(
                "Round %d health: status=%s | proposal_std=(%.4f, %.4f) | proposal_edge_frac=%.4f | "
                "dist_med=%.4f [p16=%.4f, p84=%.4f] | posterior_width68=(%.4f, %.4f) | "
                "posterior_edge_frac=%.4f | neg_bin_frac_med=%.4f [p84=%.4f]",
                round_num,
                health_status,
                theta_q_std,
                theta_m_std,
                theta_edge_frac,
                dist_median,
                dist_p16,
                dist_p84,
                post_q_width,
                post_m_width,
                post_edge_frac,
                neg_bin_frac_median,
                neg_bin_frac_p84,
            )

            if health_flags:
                for flag in health_flags:
                    logger.warning("Round %d health flag: %s", round_num, flag)
                unhealthy_rounds.append(round_num)
                if args.abort_on_health_fail:
                    abort_error = RuntimeError(
                        f"Health checks failed in round {round_num}."
                    )

            health_records.append(
                {
                    "round": round_num,
                    "status": health_status,
                    "proposal_q_mean": theta_q_mean,
                    "proposal_m_mean": theta_m_mean,
                    "proposal_q_std": theta_q_std,
                    "proposal_m_std": theta_m_std,
                    "proposal_q_min": float(np.min(theta_q)),
                    "proposal_q_max": float(np.max(theta_q)),
                    "proposal_m_min": float(np.min(theta_m)),
                    "proposal_m_max": float(np.max(theta_m)),
                    "proposal_edge_frac": theta_edge_frac,
                    "summary_distance_median": dist_median,
                    "summary_distance_p16": dist_p16,
                    "summary_distance_p84": dist_p84,
                    "median_negative_bin_frac": neg_bin_frac_median,
                    "p84_negative_bin_frac": neg_bin_frac_p84,
                    "posterior_q_p16": post_q16,
                    "posterior_q_p50": post_q50,
                    "posterior_q_p84": post_q84,
                    "posterior_m_p16": post_m16,
                    "posterior_m_p50": post_m50,
                    "posterior_m_p84": post_m84,
                    "posterior_q_width68": post_q_width,
                    "posterior_m_width68": post_m_width,
                    "posterior_edge_frac": post_edge_frac,
                    "flags": health_flags,
                }
            )

            prev_metrics = {
                "theta_q_std": theta_q_std,
                "theta_m_std": theta_m_std,
                "dist_median": dist_median,
                "post_q_width": post_q_width,
                "post_m_width": post_m_width,
            }

            logger.info("Finished round %d/%d", round_num, args.rounds)

            if abort_error is not None:
                logger.error("Aborting after round %d due to health-check failure.", round_num)
                break

    finally:
        if pool is not None:
            pool.close()
            pool.join()
            logger.info("Worker pool closed.")

        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
            logger.info("Temporary MUSIC directory removed: %s", temp_root)
        else:
            logger.info("Temporary MUSIC directory kept: %s", temp_root)

    health_path = os.path.join(args.outdir, "snpe_round_health.json")
    with open(health_path, "w", encoding="utf-8") as f:
        json.dump(health_records, f, indent=2)
    logger.info("Saved round health report: %s", health_path)

    if unhealthy_rounds:
        logger.warning(
            "Health checks flagged rounds: %s",
            ",".join(str(r) for r in unhealthy_rounds),
        )
    else:
        logger.info("All round health checks passed.")

    if abort_error is not None:
        raise abort_error

    # ----------------------------------------------------------
    # FINAL POSTERIOR
    # ----------------------------------------------------------
    posterior = posterior.set_default_x(x_obs)
    samples = posterior.sample((96000,))
    samples_np = samples.cpu().numpy()

    samples_name = (
        "snpe_ps_nnsub_wedge_posterior_samples.npy"
        if args.summary_mode == "nnsub"
        else "snpe_ps_wedge_posterior_samples.npy"
    )
    samples_path = os.path.join(args.outdir, samples_name)
    np.save(samples_path, samples_np)
    logger.info("Saved posterior samples: %s", samples_path)

    q_width, q16, q50, q84 = width_68(samples_np[:, 0])
    m_width, m16, m50, m84 = width_68(samples_np[:, 1])
    logger.info("Final posterior QMHII: %.4f (+%.4f, -%.4f)", q50, q84 - q50, q50 - q16)
    logger.info("Final posterior log10Mmin: %.4f (+%.4f, -%.4f)", m50, m84 - m50, m50 - m16)
    logger.info("Truth: QMHII=%.2f, log10Mmin=%.1f", THETA_TRUE[0], THETA_TRUE[1])

    results_name = (
        "sbi_ps_nnsub_wedge_results.npz"
        if args.summary_mode == "nnsub"
        else "sbi_ps_wedge_results.npz"
    )
    results_path = os.path.join(args.outdir, results_name)
    np.savez(
        results_path,
        samples=samples_np,
        true_params=THETA_TRUE,
        target_ps=target_ps,
        k_bins=k_bins.astype(np.float32),
        summary_mode=args.summary_mode,
        noise_ps_mean=noise_ps_mean,
    )
    logger.info("Saved results summary: %s", results_path)

    fig = corner.corner(
        samples_np,
        labels=[r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"],
        bins=50,
        smooth=1.0,
        color="C0",
        show_titles=True,
        title_fmt=".3f",
    )
    corner.overplot_lines(fig, THETA_TRUE, color="red", linewidth=2)
    corner.overplot_points(fig, THETA_TRUE[None, :], marker="x", color="red")

    corner_name = (
        "snpe_ps_nnsub_wedge_corner.png"
        if args.summary_mode == "nnsub"
        else "snpe_ps_wedge_corner.png"
    )
    corner_path = os.path.join(args.outdir, corner_name)
    plt.savefig(corner_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved corner plot: %s", corner_path)

    logger.info("PARALLEL WEDGE-FILTERED POWER-SPECTRUM SBI COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
