#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="IMNN Training (Tuning: batch=128, lr=5e-5, reg=1e-2)")
    p.add_argument("--train_dir", default="/scratch/subhankar/MUSIC/data_snaps_brightness_temp_maps_mean_sub_w_noise/thousand_data_both_change_QHII_log10Mmin_ngrid_64_doing_analysis/temp_plus_noise_maps_ms/temp_plus_noise_maps_ms_training")
    p.add_argument("--val_dir", default="/scratch/subhankar/MUSIC/data_snaps_brightness_temp_maps_mean_sub_w_noise/thousand_data_both_change_QHII_log10Mmin_ngrid_64_doing_analysis/temp_plus_noise_maps_ms/temp_plus_noise_maps_ms_validation")
    p.add_argument("--outdir", default="./imnn_output_opt_v2_batch_128_lr_5e-5_reg_1e-2")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_iterations", type=int, default=500)
    p.add_argument("--num_threads", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


args = parse_args()

num_threads = args.num_threads
if num_threads is None:
    num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))

os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import imnn_tf

tf.config.optimizer.set_jit(True)

print(f"TensorFlow: {tf.__version__}")
print("XLA JIT: Enabled")
print(f"Threads: {num_threads}")


tempnoise_main = np.load(os.path.join(args.train_dir, "training_temp_plus_noise_ms_main.npy"), allow_pickle=True).item()
tempnoise_QHII_plus = np.load(os.path.join(args.train_dir, "training_temp_plus_noise_ms_QHII_plus.npy"), allow_pickle=True).item()
tempnoise_QHII_minus = np.load(os.path.join(args.train_dir, "training_temp_plus_noise_ms_QHII_minus.npy"), allow_pickle=True).item()
tempnoise_Mmin_plus = np.load(os.path.join(args.train_dir, "training_temp_plus_noise_ms_Mmin_plus.npy"), allow_pickle=True).item()
tempnoise_Mmin_minus = np.load(os.path.join(args.train_dir, "training_temp_plus_noise_ms_Mmin_minus.npy"), allow_pickle=True).item()

val_main = np.load(os.path.join(args.val_dir, "validation_temp_plus_noise_ms_main.npy"), allow_pickle=True).item()
val_QHII_plus = np.load(os.path.join(args.val_dir, "validation_temp_plus_noise_ms_QHII_plus.npy"), allow_pickle=True).item()
val_QHII_minus = np.load(os.path.join(args.val_dir, "validation_temp_plus_noise_ms_QHII_minus.npy"), allow_pickle=True).item()
val_Mmin_plus = np.load(os.path.join(args.val_dir, "validation_temp_plus_noise_ms_Mmin_plus.npy"), allow_pickle=True).item()
val_Mmin_minus = np.load(os.path.join(args.val_dir, "validation_temp_plus_noise_ms_Mmin_minus.npy"), allow_pickle=True).item()

train_seeds = sorted(tempnoise_main.keys(), key=int)
val_seeds = sorted(val_main.keys(), key=int)
first_map = tempnoise_main[train_seeds[0]]
map_shape = first_map.shape

fiducial = np.array([tempnoise_main[s] for s in train_seeds], dtype=np.float32)[..., np.newaxis]
derivative = np.empty((len(train_seeds), 2, 2) + map_shape + (1,), dtype=np.float32)
for i, s in enumerate(train_seeds):
    derivative[i, 0, 0] = tempnoise_QHII_minus[s][..., np.newaxis]
    derivative[i, 0, 1] = tempnoise_QHII_plus[s][..., np.newaxis]
    derivative[i, 1, 0] = tempnoise_Mmin_minus[s][..., np.newaxis]
    derivative[i, 1, 1] = tempnoise_Mmin_plus[s][..., np.newaxis]

validation_fiducial = np.array([val_main[s] for s in val_seeds], dtype=np.float32)[..., np.newaxis]
validation_derivative = np.empty((len(val_seeds), 2, 2) + map_shape + (1,), dtype=np.float32)
for i, s in enumerate(val_seeds):
    validation_derivative[i, 0, 0] = val_QHII_minus[s][..., np.newaxis]
    validation_derivative[i, 0, 1] = val_QHII_plus[s][..., np.newaxis]
    validation_derivative[i, 1, 0] = val_Mmin_minus[s][..., np.newaxis]
    validation_derivative[i, 1, 1] = val_Mmin_plus[s][..., np.newaxis]


def fiducial_loader(seed, data):
    yield data[seed], seed


def derivative_loader(seed, deriv, param, data):
    yield data[seed, param, deriv], (np.int32(seed), np.int32(deriv), np.int32(param))


n_s = fiducial.shape[0]
n_d = derivative.shape[0]
input_shape = fiducial.shape[1:]

reg_strength = 1e-2

inputs = tf.keras.Input(shape=(64, 64, 64, 1))
x = tf.keras.layers.Conv3D(16, 5, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(reg_strength))(inputs)
x = tf.keras.layers.LeakyReLU(0.1)(x)
x = tf.keras.layers.AveragePooling3D(2)(x)
x = tf.keras.layers.Conv3D(16, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(reg_strength))(x)
x = tf.keras.layers.LeakyReLU(0.1)(x)
x = tf.keras.layers.AveragePooling3D(2)(x)
x = tf.keras.layers.Conv3D(32, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(reg_strength))(x)
x = tf.keras.layers.LeakyReLU(0.1)(x)
x = tf.keras.layers.AveragePooling3D(2)(x)
x = tf.keras.layers.Conv3D(32, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(reg_strength))(x)
x = tf.keras.layers.LeakyReLU(0.1)(x)
x = tf.keras.layers.AveragePooling3D(2)(x)
x = tf.keras.layers.GlobalAveragePooling3D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(16, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(reg_strength))(x)
x = tf.keras.layers.LeakyReLU(0.1)(x)
outputs = tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(reg_strength))(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="imnn_model")
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)

os.makedirs(args.outdir, exist_ok=True)
imnn = imnn_tf.IMNN(
    n_s=n_s,
    n_d=n_d,
    n_params=2,
    n_summaries=2,
    model=model,
    optimiser=opt,
    θ_fid=np.array([0.54, 9.0]),
    δθ=np.array([0.01, 0.02]),
    input_shape=input_shape,
    fiducial=lambda x: fiducial_loader(x, fiducial),
    derivative=lambda x, y, z: derivative_loader(x, y, z, derivative),
    validation_fiducial=lambda x: fiducial_loader(x, validation_fiducial),
    validation_derivative=lambda x, y, z: derivative_loader(x, y, z, validation_derivative),
    at_once=min(args.batch_size, n_s),
    directory=args.outdir,
    filename="model_opt_v2_batch_128_lr_5e-5_reg_1e-2",
    save=True,
    verbose=True,
)

print("Starting Training...")
imnn.fit(patience=args.patience, min_iterations=args.min_iterations)

history_path = os.path.join(args.outdir, "history.npz")
train_detF = np.asarray(imnn.history["det_F"])
val_detF = np.asarray(imnn.history["val_det_F"])
np.savez(history_path, det_F=train_detF, val_det_F=val_detF)

try:
    imnn.plot()
    plt.savefig(os.path.join(args.outdir, "training_history.png"), dpi=300, bbox_inches="tight")
    plt.close()
except Exception:
    pass

plt.style.use("default")
epochs = np.arange(1, len(train_detF) + 1)
plt.figure()
plt.plot(epochs, train_detF, label="Training")
plt.plot(epochs, val_detF, color="red", linestyle="--", label="Validation")
plt.legend()
plt.savefig(os.path.join(args.outdir, "fisher_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

np.save(os.path.join(args.outdir, "fisher_matrix.npy"), imnn.F)
print("Done!")
