#!/usr/bin/env python
# coding: utf-8

"""Parallel SNPE pipeline for wedge-filtered Temp+Noise+IMNN summaries.

This version keeps SNPE training sequential (as required) and parallelizes the
expensive MUSIC+map simulation stage within each round. Each simulation now
follows the same preprocessing as the wedge-filtered IMNN training data:

signal + noise -> mean subtraction -> FFT -> wedge mask -> inverse FFT.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import sys
import traceback

import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

import script
from script import two_lpt


# ==============================================================
# FIXED PATHS
# ==============================================================

PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge"
BASE_NOISE_PROJECT_DIR = "/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise"

TARGET_SEED = 1259935638
TARGET_MAP = (
    f"{PROJECT_DIR}/target_data/target_tempmap_w_noise_ms_wedge_filtered_seed1259935638.npy"
)

MODEL_DIR = (
    f"{PROJECT_DIR}/imnn_output_wedge_filtered_fast_try2_correct/model_wedge_filtered_tempmap_ngrid64_fast/model_cnn_wedge_filtered_tempmap_fast"
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


# ==============================================================
# WEDGE FILTER
# ==============================================================


def compute_wedge_slope(z, omega_m, omega_l):
    """Compute the foreground wedge slope C = x(z) H(z) / (c (1+z))."""

    def efunc(zp):
        return np.sqrt(omega_m * (1.0 + zp) ** 3 + omega_l)

    integral_z, _ = quad(lambda zp: 1.0 / efunc(zp), 0.0, z)
    return integral_z * efunc(z) / (1.0 + z)


def get_3d_k_grids(ngrid, box):
    """Generate 3D cylindrical k-grids matching np.fft.fftn output ordering."""
    k1d = np.fft.fftfreq(ngrid, d=box / ngrid) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_par = np.abs(kz)
    k_perp = np.sqrt(kx ** 2 + ky ** 2)
    return k_par, k_perp


def apply_wedge_filter(map_3d, wedge_mask):
    """Apply wedge filtering in Fourier space and return the real-space map."""
    ft = np.fft.fftn(map_3d)
    ft[~wedge_mask] = 0.0
    return np.fft.ifftn(ft).real.astype(np.float32, copy=False)


# ==============================================================
# ARGUMENTS / LOGGING
# ==============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel SBI with default MAF and round-wise health checks."
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sims_per_round", type=int, default=2000)
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
        help="Optional global seed for numpy/torch/tensorflow RNGs.",
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


def init_worker(worker_cfg):
    global _WORKER_CFG
    _WORKER_CFG = worker_cfg


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
        return {
            "idx": idx,
            "tb_ms_wedge": tb_sim.astype(np.float32, copy=False),
            "sim_seed": sim_seed,
            "noise_file": noise_file,
            "error": None,
        }
    except Exception:
        return {
            "idx": idx,
            "tb_ms_wedge": None,
            "sim_seed": sim_seed,
            "noise_file": noise_file,
            "error": traceback.format_exc(),
        }


# ==============================================================
# SEED HELPERS
# ==============================================================


def load_excluded_seeds(seed_dir):
    excluded = set()
    excluded.update(np.load(f"{seed_dir}/random_seeds_training.npy").tolist())
    excluded.update(np.load(f"{seed_dir}/random_seeds_validation.npy").tolist())
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


def get_npz_field(npz_obj, candidate_keys):
    for key in candidate_keys:
        if key in npz_obj:
            return npz_obj[key]
    available = ", ".join(npz_obj.files)
    keys_str = ", ".join(candidate_keys)
    raise KeyError(
        f"Could not find any of [{keys_str}] in estimator file. "
        f"Available keys: [{available}]"
    )


def edge_fraction(
    q_vals,
    m_vals,
    q_margin,
    m_margin,
):
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
    import tensorflow as tf
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    from sbi.utils.get_nn_models import posterior_nn

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logger = setup_logging(os.path.join(args.outdir, "run.log"))

    logger.info("=" * 80)
    logger.info("PARALLEL SBI PIPELINE STARTED (WEDGE-FILTERED)")
    logger.info("=" * 80)

    if args.rounds < 1:
        logger.error("--rounds must be >= 1")
        sys.exit(1)
    if args.sims_per_round < 1:
        logger.error("--sims_per_round must be >= 1")
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
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        tf.random.set_seed(args.global_seed)
        logger.info("Global seed set to %d (numpy/torch/tensorflow).", args.global_seed)

    logger.info(
        "Health checks enabled: edge_frac<=%.3f, std_growth<=%.2f, q_std<=%.3f, m_std<=%.3f, "
        "dist_growth<=%.2f, post_width_growth<=%.2f, post_q_width<=%.3f, post_m_width<=%.3f, "
        "post_edge_frac<=%.3f",
        args.health_theta_edge_frac_max,
        args.health_std_growth_factor,
        args.health_q_std_max,
        args.health_m_std_max,
        args.health_dist_growth_factor,
        args.health_post_width_growth_factor,
        args.health_post_q_width_max,
        args.health_post_m_width_max,
        args.health_post_edge_frac_max,
    )

    # ----------------------------------------------------------
    # LOAD TARGET
    # ----------------------------------------------------------
    if not os.path.exists(TARGET_MAP):
        logger.error("Target map not found: %s", TARGET_MAP)
        sys.exit(1)

    tb_obs = np.load(TARGET_MAP)
    if tb_obs.ndim == 3:
        tb_obs = tb_obs[..., None]

    logger.info("Target shape: %s", tb_obs.shape)
    logger.info("Target mean: %.6e", np.mean(tb_obs))
    logger.info("Target source: wedge-filtered mean-subtracted noisy target map")

    # ----------------------------------------------------------
    # IMNN ARCHITECTURE (MATCH TRAINING)
    # ----------------------------------------------------------
    reg_strength = 1e-3
    inputs = tf.keras.Input(shape=(64, 64, 64, 1))

    x = tf.keras.layers.Conv3D(
        16,
        5,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(inputs)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.AveragePooling3D(2)(x)

    x = tf.keras.layers.Conv3D(
        16,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.AveragePooling3D(2)(x)

    x = tf.keras.layers.Conv3D(
        32,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.AveragePooling3D(2)(x)

    x = tf.keras.layers.Conv3D(
        32,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.AveragePooling3D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        16,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    weights_path = os.path.join(MODEL_DIR, "weights.h5")
    estimator_path = os.path.join(MODEL_DIR, "estimator.npz")

    if not os.path.exists(weights_path):
        logger.error("IMNN weights not found: %s", weights_path)
        sys.exit(1)
    if not os.path.exists(estimator_path):
        logger.error("Estimator file not found: %s", estimator_path)
        sys.exit(1)

    model.load_weights(weights_path)
    logger.info("IMNN model loaded.")

    # ----------------------------------------------------------
    # LOAD ESTIMATOR
    # ----------------------------------------------------------
    with np.load(estimator_path) as est_data:
        finv_arr = get_npz_field(est_data, ["Finv"])
        theta_fid_arr = get_npz_field(est_data, ["theta_fid", "θ_fid"])
        dmu_dtheta_arr = get_npz_field(est_data, ["dmu_dtheta", "dμ_dθ"])
        cinv_arr = get_npz_field(est_data, ["Cinv"])
        mu_arr = get_npz_field(est_data, ["mu", "μ"])

    finv = tf.constant(finv_arr, dtype=tf.float32)
    theta_fid_est = tf.constant(theta_fid_arr, dtype=tf.float32)
    dmu_dtheta = tf.constant(dmu_dtheta_arr, dtype=tf.float32)
    cinv = tf.constant(cinv_arr, dtype=tf.float32)
    mu = tf.constant(mu_arr, dtype=tf.float32)

    @tf.function
    def imnn_estimator(x_in):
        summary = model(x_in, training=False)
        return theta_fid_est + tf.einsum(
            "ij,jk,kl,ml->mi",
            finv,
            dmu_dtheta,
            cinv,
            summary - mu,
        )

    def get_summary(x_in):
        x_arr = np.asarray(x_in, dtype=np.float32)
        if x_arr.ndim == 4:
            x_arr = x_arr[None, ...]
        return model(x_arr, training=False).numpy()[0]

    # ----------------------------------------------------------
    # GAUSSIAN APPROXIMATION
    # ----------------------------------------------------------
    theta_ml = imnn_estimator(tb_obs[None, ...]).numpy()[0]
    fisher = np.linalg.inv(finv.numpy())
    cov = np.linalg.inv(fisher)
    sigma = np.sqrt(np.diag(cov))

    logger.info("Gaussian ML estimate: %s", theta_ml)
    logger.info("1sigma uncertainties: %s", sigma)

    np.save(os.path.join(args.outdir, "gaussian_ml.npy"), theta_ml)
    np.save(os.path.join(args.outdir, "gaussian_cov.npy"), cov)

    # ----------------------------------------------------------
    # SBI SETUP
    # ----------------------------------------------------------
    prior = BoxUniform(
        low=torch.tensor([0.1, 7.5], dtype=torch.float32),
        high=torch.tensor([0.95, 10.5], dtype=torch.float32),
    )

    inference = SNPE(prior=prior, density_estimator=posterior_nn(model="maf"))
    logger.info(
        "Using default sbi==0.22.0 MAF settings: hidden_features=50, num_blocks=2, num_transforms=5."
    )

    proposal = prior
    x_obs_summary = torch.tensor(get_summary(tb_obs), dtype=torch.float32)
    x_obs_summary_np = x_obs_summary.cpu().numpy()
    summary_dim = int(x_obs_summary.shape[0])

    # ----------------------------------------------------------
    # SEED / NOISE HANDLING
    # ----------------------------------------------------------
    excluded = load_excluded_seeds(SEED_DIR)
    used = set()

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

    wedge_slope = compute_wedge_slope(Z_TARGET, OMEGA_M, OMEGA_L)
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
            round_noise_files = noise_files[
                noise_index : noise_index + args.sims_per_round
            ]
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

            summaries = np.empty((args.sims_per_round, summary_dim), dtype=np.float32)

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
                tb_ms_wedge = result["tb_ms_wedge"]
                summaries[idx] = get_summary(tb_ms_wedge[..., None])

            summary_dist = np.linalg.norm(
                summaries - x_obs_summary_np[None, :],
                axis=1,
            )
            dist_median = float(np.median(summary_dist))
            dist_p16, dist_p84 = np.percentile(summary_dist, [16.0, 84.0])
            dist_p16 = float(dist_p16)
            dist_p84 = float(dist_p84)

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
            proposal = posterior.set_default_x(x_obs_summary)

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
                "dist_med=%.4f [p16=%.4f, p84=%.4f] | posterior_width68=(%.4f, %.4f) | posterior_edge_frac=%.4f",
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
    # POSTERIOR SAMPLES
    # ----------------------------------------------------------
    posterior = posterior.set_default_x(x_obs_summary)
    samples = posterior.sample((96000,))
    samples_np = samples.cpu().numpy()

    np.save(
        os.path.join(args.outdir, "snpe_posterior_samples.npy"),
        samples_np,
    )

    # ----------------------------------------------------------
    # CORNER PLOT
    # ----------------------------------------------------------
    theta_fid = np.array([0.54, 9.0], dtype=np.float32)

    fig = corner.corner(
        samples_np,
        labels=[r"$Q^M_{\mathrm{HII}}$", r"$\log_{10} M_{\min}$"],
        bins=50,
        smooth=1.0,
        color="C0",
        show_titles=True,
        title_fmt=".3f",
    )

    corner.overplot_lines(fig, theta_fid, color="red", linewidth=2)
    corner.overplot_points(fig, theta_fid[None, :], marker="x", color="red")

    plt.savefig(
        os.path.join(args.outdir, "snpe_corner.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("PARALLEL SBI COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
