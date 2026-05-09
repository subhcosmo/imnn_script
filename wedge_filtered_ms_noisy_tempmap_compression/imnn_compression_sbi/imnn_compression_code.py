#!/usr/bin/env python
# coding: utf-8

"""
Fast IMNN training script for wedge-filtered mean-subtracted temperature maps.

This version keeps the SAME network architecture as the baseline script and only
adds runtime accelerations:
- optional mixed precision (GPU)
- optional XLA JIT
- larger configurable IMNN at_once batch
- GPU memory-growth setup
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import imnn_tf
import matplotlib.pyplot as plt

from matplotlib import dviread as mpl_dviread
from matplotlib.backends.backend_pdf import FigureCanvasPdf


# ---------------------------------------------------------------------------
# Matplotlib helpers (LaTeX-safe)
# ---------------------------------------------------------------------------

def configure_matplotlib_tex(use_tex: bool) -> None:
    if not use_tex:
        return

    def _disable_luatex_helper(self):
        raise FileNotFoundError("Disable luatex helper and use kpsewhich fallback.")

    mpl_dviread._LuatexKpsewhich._new_proc = _disable_luatex_helper
    mpl_dviread._find_tex_file.cache_clear()


def configure_style(use_tex: bool) -> None:
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "font.size": 12.0,
        "axes.labelsize": 14.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 11.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast IMNN training on wedge-filtered temperature maps")

    p.add_argument(
        "--train_dir",
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/maps_wedge_filtered",
        help="Directory with training wedge-filtered .npy files",
    )
    p.add_argument(
        "--val_dir",
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/maps_wedge_filtered",
        help="Directory with validation wedge-filtered .npy files",
    )
    p.add_argument(
        "--outdir",
        default="/scratch/subhankar/thesis/codes/imnn_w_ms_temp_map_w_noise_w_wedge/imnn_output_wedge_filtered_fast",
        help="Directory to save outputs",
    )

    p.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    p.add_argument("--min_iterations", type=int, default=350, help="Minimum training iterations")

    p.add_argument("--theta_qhii", type=float, default=0.54, help="Fiducial QHII")
    p.add_argument("--theta_log10mmin", type=float, default=9.0, help="Fiducial log10Mmin")
    p.add_argument("--delta_qhii", type=float, default=0.01, help="Finite difference step for QHII")
    p.add_argument("--delta_log10mmin", type=float, default=0.02, help="Finite difference step for log10Mmin")

    p.add_argument("--learning_rate", type=float, default=2e-4, help="Adam learning rate")
    p.add_argument("--reg_strength", type=float, default=1e-3, help="L2 regularization")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")

    p.add_argument("--at_once", type=int, default=200, help="IMNN at_once batch (memory-speed tradeoff)")

    # Runtime-only accelerations (architecture unchanged)
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision on GPU")
    p.add_argument("--xla", action="store_true", help="Enable XLA JIT")
    p.add_argument("--intra_threads", type=int, default=0, help="TF intra-op threads (0=auto)")
    p.add_argument("--inter_threads", type=int, default=0, help="TF inter-op threads (0=auto)")

    p.add_argument("--no_usetex", action="store_true", help="Disable external LaTeX in plots")

    return p.parse_args()


# ---------------------------------------------------------------------------
# TF runtime setup
# ---------------------------------------------------------------------------

def setup_tf_runtime(args: argparse.Namespace):
    if args.intra_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
    if args.inter_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.optimizer.set_jit(bool(args.xla))

    mixed_precision_active = False

    if args.mixed_precision and gpus:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        mixed_precision_active = True
    elif args.mixed_precision and not gpus:
        print("Requested --mixed_precision but no GPU detected. Falling back to float32.")

    print("TensorFlow runtime setup:")
    print(f"  GPUs detected      : {len(gpus)}")
    print(f"  XLA enabled        : {bool(args.xla)}")
    print(f"  mixed precision    : {mixed_precision_active}")
    print(f"  intra threads      : {args.intra_threads if args.intra_threads > 0 else 'auto'}")
    print(f"  inter threads      : {args.inter_threads if args.inter_threads > 0 else 'auto'}")

    return gpus, mixed_precision_active


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_map_dict(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=True).item()


def build_arrays(main_d, qhii_p_d, qhii_m_d, mmin_p_d, mmin_m_d, dtype=np.float32):
    seeds = sorted(main_d.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s))
    map_shape = main_d[seeds[0]].shape

    fiducial = np.array([main_d[s] for s in seeds], dtype=dtype)[..., np.newaxis]

    derivative = np.empty((len(seeds), 2, 2) + map_shape + (1,), dtype=dtype)
    for i, s in enumerate(seeds):
        derivative[i, 0, 0] = qhii_m_d[s][..., np.newaxis]
        derivative[i, 0, 1] = qhii_p_d[s][..., np.newaxis]
        derivative[i, 1, 0] = mmin_m_d[s][..., np.newaxis]
        derivative[i, 1, 1] = mmin_p_d[s][..., np.newaxis]

    return seeds, fiducial, derivative, map_shape


def fiducial_loader(seed, data):
    yield data[seed], seed


def derivative_loader(seed, deriv, param, data):
    yield data[seed, param, deriv], (np.int32(seed), np.int32(deriv), np.int32(param))


# ---------------------------------------------------------------------------
# Model (EXACT SAME ARCHITECTURE as baseline)
# ---------------------------------------------------------------------------

def build_model(input_shape, reg_strength=1e-3, dropout=0.3):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv3D(
        16,
        kernel_size=5,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.AveragePooling3D(pool_size=2)(x)

    x = tf.keras.layers.Conv3D(
        16,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.AveragePooling3D(pool_size=2)(x)

    x = tf.keras.layers.Conv3D(
        32,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.AveragePooling3D(pool_size=2)(x)

    x = tf.keras.layers.Conv3D(
        32,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.AveragePooling3D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(
        16,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Keep final output in float32 for numerical stability with mixed precision.
    outputs = tf.keras.layers.Dense(
        2,
        kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
        dtype="float32",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="imnn_tempmap_wedge_filtered_fast")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_pdf(fig: plt.Figure, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.splitext(path)[1].lower() == ".pdf":
        fig.set_canvas(FigureCanvasPdf(fig))
    plt.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_fisher(history_data: dict, out_path: str) -> None:
    epochs = np.arange(1, len(history_data["det_F"]) + 1)
    train_F = history_data["det_F"]
    val_F = history_data["val_det_F"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(epochs, train_F, color="#1f77b4", lw=1.8, label="Training")
    ax.plot(epochs, val_F, color="#d62728", lw=1.8, ls="--", label="Validation")
    ax.set_xlabel(r"Epoch", labelpad=4)
    ax.set_ylabel(r"$|\mathbf{F}_{\alpha\beta}|$", labelpad=4)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, pad=4)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="best", handlelength=1.5, handletextpad=0.4, columnspacing=0.6)

    save_pdf(fig, out_path)


def plot_three_panel(history_data: dict, out_path: str) -> None:
    epochs = np.arange(1, len(history_data["det_F"]) + 1)

    train_F = history_data["det_F"]
    val_F = history_data["val_det_F"]

    train_C = history_data.get("det_C")
    val_C = history_data.get("val_det_C")
    train_Cinv = history_data.get("det_Cinv")
    val_Cinv = history_data.get("val_det_Cinv")

    r_arr = history_data.get("r")
    lam_arr = history_data.get("lambda", history_data.get("Lambda", history_data.get("lam")))

    fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True, gridspec_kw={"hspace": 0.05})

    ax1 = axes[0]
    ax1.plot(epochs, train_F, color="#1f77b4", lw=1.5, label=r"$|\mathbf{F}_{\alpha\beta}|$ (train)")
    ax1.plot(epochs, val_F, color="#ff7f0e", lw=1.5, label=r"$|\mathbf{F}_{\alpha\beta}|$ (val)")
    ax1.set_ylabel(r"$|\mathbf{F}_{\alpha\beta}|$")
    ax1.legend(frameon=False, loc="upper left")
    ax1.tick_params(which="both", top=True, right=True)

    ax2 = axes[1]
    if all(x is not None for x in [train_C, val_C, train_Cinv, val_Cinv]):
        ax2.semilogy(epochs, train_C, color="#1f77b4", lw=1.5, label=r"$|\mathbf{C}|$ (train)")
        ax2.semilogy(epochs, val_C, color="#ff7f0e", lw=1.5, label=r"$|\mathbf{C}|$ (val)")
        ax2.semilogy(epochs, train_Cinv, color="#1f77b4", lw=1.5, ls="--", label=r"$|\mathbf{C}^{-1}|$ (train)")
        ax2.semilogy(epochs, val_Cinv, color="#ff7f0e", lw=1.5, ls="--", label=r"$|\mathbf{C}^{-1}|$ (val)")
        ax2.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.5)
        ax2.legend(frameon=False, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "det_C / det_Cinv not available in history", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_ylabel(r"$|\mathbf{C}|, |\mathbf{C}^{-1}|$")
    ax2.tick_params(which="both", top=True, right=True)

    ax3 = axes[2]
    if lam_arr is not None:
        ax3.semilogy(epochs, lam_arr, color="#1f77b4", lw=1.1, label=r"$\Lambda_\Sigma$")
        ax3.set_ylabel(r"$\Lambda_\Sigma$")
        ax3.legend(frameon=False, loc="upper left")
    else:
        ax3.set_ylabel("lambda (not available)")

    if r_arr is not None:
        ax3r = ax3.twinx()
        ax3r.plot(epochs, r_arr, color="#d62728", lw=1.1, ls="--", label=r"$r$")
        ax3r.set_ylabel(r"$r$")
        ax3r.tick_params(which="both", right=True)
        ax3r.legend(frameon=False, loc="upper right")

    ax3.set_xlabel("Epoch")
    ax3.tick_params(which="both", top=True, right=True)
    ax3.set_xlim(epochs[0], epochs[-1])

    save_pdf(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    use_tex = not args.no_usetex
    configure_matplotlib_tex(use_tex)
    configure_style(use_tex)

    gpus, mixed_precision_active = setup_tf_runtime(args)

    print("=" * 70)
    print("Fast IMNN Training on Wedge-Filtered Temperature Maps")
    print("=" * 70)
    print(f"Training dir  : {args.train_dir}")
    print(f"Validation dir: {args.val_dir}")
    print(f"Output dir    : {args.outdir}")

    # Load datasets
    train_main = load_map_dict(os.path.join(args.train_dir, "training_wedge_filtered_main.npy"))
    train_qhii_p = load_map_dict(os.path.join(args.train_dir, "training_wedge_filtered_QHII_plus.npy"))
    train_qhii_m = load_map_dict(os.path.join(args.train_dir, "training_wedge_filtered_QHII_minus.npy"))
    train_mmin_p = load_map_dict(os.path.join(args.train_dir, "training_wedge_filtered_Mmin_plus.npy"))
    train_mmin_m = load_map_dict(os.path.join(args.train_dir, "training_wedge_filtered_Mmin_minus.npy"))

    val_main = load_map_dict(os.path.join(args.val_dir, "validation_wedge_filtered_main.npy"))
    val_qhii_p = load_map_dict(os.path.join(args.val_dir, "validation_wedge_filtered_QHII_plus.npy"))
    val_qhii_m = load_map_dict(os.path.join(args.val_dir, "validation_wedge_filtered_QHII_minus.npy"))
    val_mmin_p = load_map_dict(os.path.join(args.val_dir, "validation_wedge_filtered_Mmin_plus.npy"))
    val_mmin_m = load_map_dict(os.path.join(args.val_dir, "validation_wedge_filtered_Mmin_minus.npy"))

    # Use float16 arrays only when mixed precision is actually active on GPU.
    arr_dtype = np.float16 if mixed_precision_active else np.float32

    train_seeds, fiducial, derivative, map_shape = build_arrays(
        train_main, train_qhii_p, train_qhii_m, train_mmin_p, train_mmin_m, dtype=arr_dtype
    )
    val_seeds, validation_fiducial, validation_derivative, _ = build_arrays(
        val_main, val_qhii_p, val_qhii_m, val_mmin_p, val_mmin_m, dtype=arr_dtype
    )

    print(f"Training seeds   : {len(train_seeds)}")
    print(f"Validation seeds : {len(val_seeds)}")
    print(f"Map shape        : {map_shape}")
    print(f"Numpy dtype      : {arr_dtype}")

    n_s = fiducial.shape[0]
    n_d = derivative.shape[0]
    input_shape = fiducial.shape[1:]

    model = build_model(input_shape=input_shape, reg_strength=args.reg_strength, dropout=args.dropout)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    model_dir = os.path.join(args.outdir, "model_wedge_filtered_tempmap_ngrid64_fast")
    os.makedirs(model_dir, exist_ok=True)

    history_npz = os.path.join(model_dir, "history.npz")
    summary_txt = os.path.join(args.outdir, "imnn_training_summary_wedge_filtered_fast.txt")
    fisher_pdf = os.path.join(args.outdir, "fisher_vs_epochs_wedge_filtered_fast.pdf")
    panel_pdf = os.path.join(args.outdir, "imnn_3panel_history_wedge_filtered_fast.pdf")
    builtin_pdf = os.path.join(args.outdir, "imnn_builtin_history_wedge_filtered_fast.pdf")

    at_once_batch = min(args.at_once, n_s) if args.at_once > 0 else min(200, n_s)
    print(f"IMNN at_once batch: {at_once_batch}")

    imnn = imnn_tf.IMNN(
        n_s=n_s,
        n_d=n_d,
        n_params=2,
        n_summaries=2,
        model=model,
        optimiser=opt,
        θ_fid=np.array([args.theta_qhii, args.theta_log10mmin]),
        δθ=np.array([args.delta_qhii, args.delta_log10mmin]),
        input_shape=input_shape,
        fiducial=lambda x: fiducial_loader(x, fiducial),
        derivative=lambda x, y, z: derivative_loader(x, y, z, derivative),
        validation_fiducial=lambda x: fiducial_loader(x, validation_fiducial),
        validation_derivative=lambda x, y, z: derivative_loader(x, y, z, validation_derivative),
        at_once=at_once_batch,
        directory=model_dir,
        filename="model_cnn_wedge_filtered_tempmap_fast",
        save=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Starting FAST IMNN training")
    print("=" * 70)
    imnn.fit(patience=args.patience, min_iterations=args.min_iterations)

    history_data = {k: np.asarray(v) for k, v in imnn.history.items()}
    np.savez(history_npz, **history_data)
    print(f"Saved history keys: {list(history_data.keys())}")

    fig_builtin = imnn.plot()
    if fig_builtin is None:
        fig_builtin = plt.gcf()
    if fig_builtin is not None:
        save_pdf(fig_builtin, builtin_pdf)

    plot_fisher(history_data, fisher_pdf)
    plot_three_panel(history_data, panel_pdf)

    train_F = history_data["det_F"]
    val_F = history_data["val_det_F"]
    best_val_epoch = int(np.argmax(val_F) + 1)

    summary_lines = [
        "=" * 70,
        "FAST IMNN Wedge-Filtered Temperature Map Training Summary",
        "=" * 70,
        f"Model dir                : {model_dir}",
        f"History npz              : {history_npz}",
        f"Built-in history plot    : {builtin_pdf}",
        f"Fisher plot              : {fisher_pdf}",
        f"3-panel plot             : {panel_pdf}",
        f"Epochs                   : {len(train_F)}",
        f"Final train |F|          : {train_F[-1]:.6e}",
        f"Final val |F|            : {val_F[-1]:.6e}",
        f"Best train |F|           : {np.max(train_F):.6e}",
        f"Best val |F|             : {np.max(val_F):.6e} (epoch {best_val_epoch})",
        f"Final Fisher matrix      :\n{imnn.F}",
        f"Final det(F) from matrix : {np.linalg.det(imnn.F):.6e}",
        f"History keys saved       : {list(history_data.keys())}",
        f"mixed_precision_requested: {args.mixed_precision}",
        f"mixed_precision_active   : {mixed_precision_active}",
        f"num_gpus_detected        : {len(gpus)}",
        f"xla                      : {args.xla}",
        f"at_once                  : {at_once_batch}",
        "=" * 70,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Saved summary: {summary_txt}")


if __name__ == "__main__":
    main()
