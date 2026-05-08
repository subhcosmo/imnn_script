#!/usr/bin/env python3
# coding: utf-8

"""Parallel SNPE pipeline for noisy 2DPS summaries with health checks.

This forward model:
1) Simulates brightness-temperature cubes from (QMHII, log10Mmin) using MUSIC + script.
2) Adds one independent 21cm noise cube to each simulated Tb cube.
3) Mean-subtracts the noisy cube.
4) Computes a 10x10 linear-binned cylindrical 2DPS using script backend.
5) Trains SNPE (default MAF) round-by-round and reports health diagnostics.

Temporary density-field folders are deleted after each simulation by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import traceback

import numpy as np
from tqdm import tqdm

import script
from script import two_lpt


# ==============================================================
# FIXED PATHS
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/2dps_paramter_estimation/noisy_Tb"

TARGET_2DPS_FILE = (
    f"{PROJECT_DIR}/target_data/"
    "target_tempmap_w_noise_ms_seed1259935638_2dps_script_linear_10x10.npy"
)

TRAINING_SEEDS_FILE = f"{PROJECT_DIR}/random_seeds_training.npy"
VALIDATION_SEEDS_FILE = f"{PROJECT_DIR}/random_seeds_validation.npy"
TARGET_SEED_FILE = f"{PROJECT_DIR}/target_data_seed.npy"
NOISE_DIR = (
    "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise/"
    "noise_maps/noise_maps_sbi"
)

MUSIC_EXEC = "/scratch/subhankar/software/music/build/MUSIC"


# ==============================================================
# COSMOLOGY
# ==============================================================

BOX = 128.0
DX = 1.0
Z_TARGET = 7.0
OMEGA_M = 0.308
OMEGA_L = 1.0 - OMEGA_M
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
# ARGUMENTS / LOGGING
# ==============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel SBI with default MAF for noisy cylindrical 2DPS summaries "
            "and round-wise health checks."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=f"{PROJECT_DIR}/sbi_output_seed_12345",
    )
    parser.add_argument("--target_2dps_file", type=str, default=TARGET_2DPS_FILE)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sims_per_round", type=int, default=2000)
    parser.add_argument("--nbins_par", type=int, default=10)
    parser.add_argument("--nbins_perp", type=int, default=10)

    parser.add_argument("--training_seeds_file", type=str, default=TRAINING_SEEDS_FILE)
    parser.add_argument("--validation_seeds_file", type=str, default=VALIDATION_SEEDS_FILE)
    parser.add_argument("--target_seed_file", type=str, default=TARGET_SEED_FILE)
    parser.add_argument(
        "--noise_dir",
        type=str,
        default=NOISE_DIR,
        help="Directory containing per-simulation 21cm noise cube .npz files.",
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
        help="Optional RNG seed for reproducible simulation-seed and noise-file assignment.",
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
        "--no_save_round_data",
        dest="save_round_data",
        action="store_false",
        help="Do not save per-round theta, summaries, and simulation seeds.",
    )
    parser.set_defaults(save_round_data=True)

    parser.add_argument("--health_edge_q_margin", type=float, default=0.03)
    parser.add_argument("--health_edge_m_margin", type=float, default=0.15)
    parser.add_argument("--health_theta_edge_frac_max", type=float, default=0.05)
    parser.add_argument("--health_std_growth_factor", type=float, default=1.25)
    parser.add_argument("--health_q_std_max", type=float, default=0.08)
    parser.add_argument("--health_m_std_max", type=float, default=0.50)
    parser.add_argument("--health_dist_growth_factor", type=float, default=1.15)
    parser.add_argument("--health_posterior_samples", type=int, default=4000)
    parser.add_argument("--health_post_width_growth_factor", type=float, default=1.25)
    parser.add_argument("--health_post_q_width_max", type=float, default=0.20)
    parser.add_argument("--health_post_m_width_max", type=float, default=1.00)
    parser.add_argument("--health_post_edge_frac_max", type=float, default=0.02)
    parser.add_argument(
        "--health_neg_bin_frac_max",
        type=float,
        default=0.05,
        help="Max acceptable median fraction of negative noisy 2DPS bins (round >=2).",
    )
    parser.add_argument(
        "--abort_on_health_fail",
        action="store_true",
        help="Abort immediately when any health check fails.",
    )
    return parser.parse_args()


def setup_logging(logpath: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(logpath, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ==============================================================
# WORKER HELPERS
# ==============================================================

_WORKER_CFG = {}
_WORKER_PS_HELPER = None


def setup_linear_k_bins(
    ngrid: int,
    box: float,
    nbins_par: int,
    nbins_perp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Match the Fisher linear-binning convention from FFT frequencies.
    k1d = np.abs(np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi).astype(np.float64)
    nonzero = k1d[k1d > 0.0]
    if nonzero.size == 0:
        raise ValueError(f"Invalid FFT k-grid for ngrid={ngrid}.")

    kmin = float(np.min(nonzero))
    kmax = float(np.max(nonzero))
    k_par_edges = np.linspace(kmin, kmax, nbins_par + 1)
    k_perp_edges = np.linspace(kmin, kmax, nbins_perp + 1)
    k_par_bins = 0.5 * (k_par_edges[:-1] + k_par_edges[1:])
    k_perp_bins = 0.5 * (k_perp_edges[:-1] + k_perp_edges[1:])
    return k_par_edges, k_perp_edges, k_par_bins, k_perp_bins


def init_worker(worker_cfg: dict) -> None:
    global _WORKER_CFG
    global _WORKER_PS_HELPER

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
        method="PC",
    )


def compute_2dps_summary(temp_map_ms_mk: np.ndarray) -> tuple[np.ndarray, float]:
    cfg = _WORKER_CFG

    ps2d, kount = _WORKER_PS_HELPER.get_binned_powspec_cylindrical(
        temp_map_ms_mk.astype(np.float32, copy=False),
        cfg["k_par_edges"],
        cfg["k_perp_edges"],
        convolve=False,
        units="",
    )

    ps2d = np.asarray(ps2d, dtype=np.float64)
    kount = np.asarray(kount, dtype=np.float64)

    expected_shape = cfg["summary_shape"]
    if ps2d.shape != expected_shape:
        raise RuntimeError(f"Expected 2DPS shape {expected_shape}, got {ps2d.shape}.")
    if kount.shape != expected_shape:
        raise RuntimeError(f"Expected kount shape {expected_shape}, got {kount.shape}.")

    empty = kount == 0
    if np.any(empty):
        ps2d[empty] = 0.0

    if not np.all(np.isfinite(ps2d)):
        raise RuntimeError("Non-finite entries found in simulated 2DPS summary.")

    neg_bin_frac = float(np.mean(ps2d < 0.0))
    return ps2d.reshape(-1).astype(np.float32), neg_bin_frac


def load_noise_cube(noise_file: str) -> np.ndarray:
    with np.load(noise_file) as noise_data:
        if "noisecube_21cm" in noise_data:
            return noise_data["noisecube_21cm"].astype(np.float32, copy=False)
        if "noise_cube" in noise_data:
            return noise_data["noise_cube"].astype(np.float32, copy=False)
        if "noise" in noise_data:
            return noise_data["noise"].astype(np.float32, copy=False)
    raise RuntimeError(f"Noise cube key not found in file: {noise_file}")


def simulate_temp_map_worker(qhii_val: float, log10mmin_val: float, seed: int) -> np.ndarray:
    cfg = _WORKER_CFG
    outpath = os.path.join(cfg["temp_root"], f"temp_seed_{seed}")
    snap_path = os.path.join(outpath, "snap_000")

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
        if not cfg.get("keep_temp", False):
            shutil.rmtree(outpath, ignore_errors=True)


def worker_simulation(task):
    idx, qhii_val, log10mmin_val, sim_seed, noise_file = task
    try:
        tb_signal = simulate_temp_map_worker(qhii_val, log10mmin_val, sim_seed)
        noise_cube = load_noise_cube(noise_file)
        if noise_cube.shape != tb_signal.shape:
            raise RuntimeError(
                f"Noise shape {noise_cube.shape} does not match Tb shape {tb_signal.shape}."
            )

        tb_noisy = tb_signal + noise_cube
        tb_noisy_ms = tb_noisy - np.mean(tb_noisy, dtype=np.float64)
        summary_flat, neg_bin_frac = compute_2dps_summary(tb_noisy_ms)

        return {
            "idx": idx,
            "summary": summary_flat,
            "neg_bin_frac": neg_bin_frac,
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


def load_excluded_seeds(
    training_seeds_file: str,
    validation_seeds_file: str,
    target_seed_file: str,
) -> set[int]:
    excluded: set[int] = set()

    if not os.path.exists(training_seeds_file):
        raise FileNotFoundError(f"Missing training seeds file: {training_seeds_file}")
    if not os.path.exists(validation_seeds_file):
        raise FileNotFoundError(f"Missing validation seeds file: {validation_seeds_file}")
    if not os.path.exists(target_seed_file):
        raise FileNotFoundError(f"Missing target seed file: {target_seed_file}")

    excluded.update(np.atleast_1d(np.load(training_seeds_file)).astype(np.int64).tolist())
    excluded.update(np.atleast_1d(np.load(validation_seeds_file)).astype(np.int64).tolist())
    excluded.update(np.atleast_1d(np.load(target_seed_file)).astype(np.int64).tolist())
    return excluded


def draw_unique_seeds(
    rng: np.random.Generator,
    excluded: set[int],
    used: set[int],
    count: int,
) -> list[int]:
    seeds: list[int] = []
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


def extract_noise_seed(noise_file: str) -> int | None:
    match = re.search(r"seed(\d+)", os.path.basename(noise_file))
    if match:
        return int(match.group(1))
    return None


def edge_fraction(
    q_vals: np.ndarray,
    m_vals: np.ndarray,
    q_margin: float,
    m_margin: float,
) -> float:
    q_low = PRIOR_Q_MIN + q_margin
    q_high = PRIOR_Q_MAX - q_margin
    m_low = PRIOR_M_MIN + m_margin
    m_high = PRIOR_M_MAX - m_margin
    on_edge = (q_vals < q_low) | (q_vals > q_high) | (m_vals < m_low) | (m_vals > m_high)
    return float(np.mean(on_edge))


def width_68(values: np.ndarray) -> tuple[float, float, float, float]:
    p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(p84 - p16), float(p16), float(p50), float(p84)


# ==============================================================
# MAIN
# ==============================================================


def main() -> None:
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
    logger.info("PARALLEL NOISY 2DPS SBI PIPELINE STARTED")
    logger.info("=" * 80)

    if args.rounds < 1:
        logger.error("--rounds must be >= 1")
        sys.exit(1)
    if args.sims_per_round < 1:
        logger.error("--sims_per_round must be >= 1")
        sys.exit(1)
    if args.nbins_par < 2 or args.nbins_perp < 2:
        logger.error("--nbins_par and --nbins_perp must be >= 2")
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

    k_par_edges, k_perp_edges, k_par_bins, k_perp_bins = setup_linear_k_bins(
        NGRID,
        BOX,
        args.nbins_par,
        args.nbins_perp,
    )
    summary_shape = (args.nbins_par, args.nbins_perp)
    summary_size = args.nbins_par * args.nbins_perp

    logger.info("2DPS bins: %d x %d", args.nbins_par, args.nbins_perp)
    logger.info("k_par_edges: %s", np.array2string(k_par_edges, precision=6))
    logger.info("k_perp_edges: %s", np.array2string(k_perp_edges, precision=6))

    if not os.path.exists(args.target_2dps_file):
        logger.error("Target 2DPS file not found: %s", args.target_2dps_file)
        sys.exit(1)

    target_2dps = np.load(args.target_2dps_file).astype(np.float32)
    if target_2dps.shape != summary_shape:
        logger.error(
            "Target 2DPS shape mismatch: expected %s, got %s",
            str(summary_shape),
            str(target_2dps.shape),
        )
        sys.exit(1)
    if not np.all(np.isfinite(target_2dps)):
        logger.error("Target 2DPS contains non-finite values: %s", args.target_2dps_file)
        sys.exit(1)

    target_summary = target_2dps.reshape(-1)
    x_obs = torch.tensor(target_summary, dtype=torch.float32)
    logger.info("Loaded target 2DPS: %s", args.target_2dps_file)
    logger.info(
        "Target 2DPS stats: min=%.6e max=%.6e median=%.6e neg_bin_frac=%.4f",
        float(np.min(target_2dps)),
        float(np.max(target_2dps)),
        float(np.median(target_2dps)),
        float(np.mean(target_2dps < 0.0)),
    )
    logger.info(
        "Summary mode: 2DPS of mean-subtracted simulated Tb plus noise; "
        "no noise-mean 2DPS subtraction."
    )

    target_base = os.path.splitext(args.target_2dps_file)[0]
    suffix = "_2dps_script_linear_10x10"
    target_prefix = target_base[: -len(suffix)] if target_base.endswith(suffix) else target_base
    target_k_files = {
        "k_par_edges": f"{target_prefix}_k_par_edges_linear.npy",
        "k_perp_edges": f"{target_prefix}_k_perp_edges_linear.npy",
    }
    for label, path in target_k_files.items():
        if os.path.exists(path):
            saved_edges = np.load(path)
            expected_edges = k_par_edges if label == "k_par_edges" else k_perp_edges
            if saved_edges.shape != expected_edges.shape or not np.allclose(
                saved_edges,
                expected_edges,
                rtol=1e-6,
                atol=1e-8,
            ):
                logger.error(
                    "Target %s mismatch. Expected %s from current settings, got %s from %s",
                    label,
                    np.array2string(expected_edges, precision=8),
                    np.array2string(saved_edges, precision=8),
                    path,
                )
                sys.exit(1)
            logger.info("Verified target %s: %s", label, path)
        else:
            logger.warning("Target %s file not found for consistency check: %s", label, path)

    # ----------------------------------------------------------
    # SBI SETUP
    # ----------------------------------------------------------
    prior = BoxUniform(
        low=torch.tensor([PRIOR_Q_MIN, PRIOR_M_MIN], dtype=torch.float32),
        high=torch.tensor([PRIOR_Q_MAX, PRIOR_M_MAX], dtype=torch.float32),
    )
    inference = SNPE(prior=prior, density_estimator=posterior_nn(model="maf"))
    logger.info(
        "Using default sbi MAF settings: hidden_features=50, num_blocks=2, num_transforms=5."
    )

    proposal = prior

    # ----------------------------------------------------------
    # SEED / NOISE HANDLING
    # ----------------------------------------------------------
    try:
        excluded = load_excluded_seeds(
            args.training_seeds_file,
            args.validation_seeds_file,
            args.target_seed_file,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    used: set[int] = set()
    rng_seed = args.random_seed if args.random_seed is not None else args.global_seed
    rng = np.random.default_rng(rng_seed)
    logger.info("Simulation-seed/noise assignment RNG seed: %s", str(rng_seed))
    logger.info("Excluded seeds count: %d", len(excluded))

    if not os.path.isdir(args.noise_dir):
        logger.error("Noise directory not found: %s", args.noise_dir)
        sys.exit(1)

    noise_files = sorted(
        [
            os.path.join(args.noise_dir, fname)
            for fname in os.listdir(args.noise_dir)
            if fname.endswith(".npz")
        ]
    )
    if len(noise_files) == 0:
        logger.error("No noise files found in: %s", args.noise_dir)
        sys.exit(1)

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

    logger.info("Noise directory: %s", args.noise_dir)
    logger.info("Noise files available: %d", len(noise_files))
    logger.info("Noise files required without reuse: %d", total_required)

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
    logger.info("Save per-round simulation data: %s", str(args.save_round_data))

    temp_root = os.path.join(args.outdir, "temp_music_runs")
    os.makedirs(temp_root, exist_ok=True)

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
        "k_par_edges": k_par_edges,
        "k_perp_edges": k_perp_edges,
        "summary_shape": summary_shape,
        "keep_temp": args.keep_temp,
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

            sim_seeds = draw_unique_seeds(rng, excluded, used, args.sims_per_round)
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

            summaries = np.empty((args.sims_per_round, summary_size), dtype=np.float32)
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

            if args.save_round_data:
                round_data_path = os.path.join(
                    args.outdir,
                    f"round_{round_num:02d}_simulation_data.npz",
                )
                np.savez_compressed(
                    round_data_path,
                    theta=theta_np.astype(np.float32),
                    summaries=summaries,
                    sim_seeds=np.asarray(sim_seeds, dtype=np.int64),
                    noise_files=np.asarray(round_noise_files),
                    neg_bin_fracs=neg_bin_fracs,
                    summary_shape=np.array(summary_shape, dtype=np.int64),
                    k_par_edges=k_par_edges.astype(np.float32),
                    k_perp_edges=k_perp_edges.astype(np.float32),
                )
                logger.info("Saved round simulation data: %s", round_data_path)

            scale = np.maximum(np.abs(target_summary), 1.0)
            summary_dist = np.linalg.norm(
                (summaries - target_summary[None, :]) / scale[None, :],
                axis=1,
            )
            dist_median = float(np.median(summary_dist))
            dist_p16, dist_p84 = np.percentile(summary_dist, [16.0, 84.0])
            dist_p16 = float(dist_p16)
            dist_p84 = float(dist_p84)

            neg_bin_frac_median = float(np.median(neg_bin_fracs))
            neg_bin_frac_p84 = float(np.percentile(neg_bin_fracs, 84.0))

            x = torch.tensor(summaries, dtype=torch.float32)

            inference = inference.append_simulations(theta, x, proposal=proposal)
            density_estimator = inference.train(force_first_round_loss=(round_idx == 0))

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
                if neg_bin_frac_median > args.health_neg_bin_frac_max:
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
                    abort_error = RuntimeError(f"Health checks failed in round {round_num}.")

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
            exc_type, _, _ = sys.exc_info()
            if exc_type is None:
                pool.close()
            else:
                pool.terminate()
            pool.join()
            logger.info("Worker pool closed.")

        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
            logger.info("Temporary MUSIC directory removed: %s", temp_root)
        else:
            logger.info("Temporary MUSIC directory kept: %s", temp_root)

    health_path = os.path.join(args.outdir, "snpe_round_health.json")
    with open(health_path, "w", encoding="utf-8") as file_obj:
        json.dump(health_records, file_obj, indent=2)
    logger.info("Saved round health report: %s", health_path)

    if unhealthy_rounds:
        logger.warning(
            "Health checks flagged rounds: %s",
            ",".join(str(rnd) for rnd in unhealthy_rounds),
        )
    else:
        logger.info("All round health checks passed.")

    if abort_error is not None:
        raise abort_error

    if posterior is None:
        raise RuntimeError("Posterior object was not built. No rounds were completed.")

    # ----------------------------------------------------------
    # FINAL POSTERIOR
    # ----------------------------------------------------------
    posterior = posterior.set_default_x(x_obs)
    samples = posterior.sample((96000,))
    samples_np = samples.cpu().numpy()

    samples_path = os.path.join(args.outdir, "snpe_2dps_noisy_posterior_samples.npy")
    np.save(samples_path, samples_np)
    logger.info("Saved posterior samples: %s", samples_path)

    q_width, q16, q50, q84 = width_68(samples_np[:, 0])
    m_width, m16, m50, m84 = width_68(samples_np[:, 1])
    logger.info("Final posterior QMHII: %.4f (+%.4f, -%.4f)", q50, q84 - q50, q50 - q16)
    logger.info("Final posterior log10Mmin: %.4f (+%.4f, -%.4f)", m50, m84 - m50, m50 - m16)
    logger.info("Truth: QMHII=%.2f, log10Mmin=%.1f", THETA_TRUE[0], THETA_TRUE[1])

    results_path = os.path.join(args.outdir, "sbi_2dps_noisy_results.npz")
    np.savez(
        results_path,
        samples=samples_np,
        true_params=THETA_TRUE,
        target_2dps=target_2dps,
        target_summary=target_summary,
        summary_shape=np.array(summary_shape, dtype=np.int64),
        k_par_edges=k_par_edges.astype(np.float32),
        k_perp_edges=k_perp_edges.astype(np.float32),
        k_par_bins=k_par_bins.astype(np.float32),
        k_perp_bins=k_perp_bins.astype(np.float32),
        noise_dir=args.noise_dir,
        summary_mode="noisy_ms_2dps_no_noise_mean_subtraction",
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

    corner_path = os.path.join(args.outdir, "snpe_2dps_noisy_corner.png")
    plt.savefig(corner_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved corner plot: %s", corner_path)

    logger.info("PARALLEL NOISY 2DPS SBI COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
