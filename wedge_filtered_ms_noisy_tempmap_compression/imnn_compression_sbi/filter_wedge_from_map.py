#!/usr/bin/env python
# coding: utf-8

"""
Apply foreground wedge removal directly to 3D brightness temperature maps.

Workflow:
  1. Load real-space map T(x)
  2. FFT -> T(k)
  3. Zero out modes inside the wedge: k_par <= C * k_perp
  4. Inverse FFT -> T_clean(x)
  5. Save T_clean(x)

Usage:
    python filter_wedge_from_map.py \
        --training_input_dir ... \
        --outdir ... \
        --z 7.0
"""

import argparse
import os
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==============================================================
# COSMOLOGY / WEDGE
# ==============================================================

def compute_wedge_slope(z, omega_m, omega_l):
    """Compute C = x(z)*H(z) / (c*(1+z))."""
    def E(zp):
        return np.sqrt(omega_m * (1.0 + zp)**3 + omega_l)

    # I(z) = integral_0^z dz' / E(z')
    I_z, _ = quad(lambda zp: 1.0 / E(zp), 0.0, z)

    # C = I(z) * E(z) / (1+z)
    C = I_z * E(z) / (1.0 + z)
    return C

def get_3d_k_grids(ngrid, box):
    """
    Generate 3D k_par and k_perp grids corresponding to np.fft.fftn output.
    Standard numpy FFT layout: frequencies order is [0, 1, ...,  n/2-1, -n/2, ..., -1]
    """
    # 1D k-array for standard FFT layout
    k1d = np.fft.fftfreq(ngrid, d=box/ngrid) * 2.0 * np.pi
    
    # Meshgrid (indexing='ij' for matrix indexing)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    
    # Assuming Line of Sight is the 3rd axis (z-axis)
    # This matches common 21cm conventions and your cylindrical PS script structure
    k_par = np.abs(kz)
    k_perp = np.sqrt(kx**2 + ky**2)
    
    return k_par, k_perp


def setup_k_bins(ngrid, box, nbins_par, nbins_perp):
    """Set up linear k-bin edges for cylindrical binning."""
    kmin = 2.0 * np.pi / box
    kmax = np.pi * ngrid / box

    k_par_edges = np.linspace(kmin, kmax, nbins_par + 1)
    k_perp_edges = np.linspace(kmin, kmax, nbins_perp + 1)

    return k_par_edges, k_perp_edges


def bin_cylindrical_stat(field_3d, k_par, k_perp, k_par_edges, k_perp_edges):
    """Bin a 3D scalar field into cylindrical (k_par, k_perp) bins by mean value."""
    nbins_par = len(k_par_edges) - 1
    nbins_perp = len(k_perp_edges) - 1

    out = np.zeros((nbins_par, nbins_perp), dtype=np.float64)

    for j in range(nbins_perp):
        mask_perp = (k_perp >= k_perp_edges[j]) & (k_perp < k_perp_edges[j + 1])
        for i in range(nbins_par):
            mask_par = (k_par >= k_par_edges[i]) & (k_par < k_par_edges[i + 1])
            mask = mask_perp & mask_par

            cnt = np.count_nonzero(mask)
            if cnt > 0:
                out[i, j] = np.mean(field_3d[mask])
            else:
                out[i, j] = 0.0

    return out

# ==============================================================
# CLI
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Filter foreground wedge from 3D maps")
    
    # Input/Output
    p.add_argument("--training_input_dir", required=True, help="Dir with training .npy maps")
    p.add_argument("--validation_input_dir", required=True, help="Dir with validation .npy maps")
    p.add_argument("--outdir", default="./wedge_filtered_maps", help="Output directory")
    
    # Map params
    p.add_argument("--box", type=float, default=128.0, help="Box size in cMpc/h")
    p.add_argument("--ngrid", type=int, default=64, help="Grid size")
    p.add_argument("--nbins_par", type=int, default=10, help="Number of k_parallel bins for FFT plotting")
    p.add_argument("--nbins_perp", type=int, default=10, help="Number of k_perp bins for FFT plotting")
    p.add_argument("--n_show", type=int, default=5, help="Number of main realizations to plot")
    
    # Cosmology
    p.add_argument("--z", type=float, default=7.0, help="Redshift")
    p.add_argument("--omega_m", type=float, default=0.308)
    p.add_argument("--omega_l", type=float, default=0.692)
    
    return p.parse_args()

class Visualizer:
    def __init__(self, outdir, n_show=5):
        self.outdir = outdir
        self.n_show = n_show
        self.plot_dir = os.path.join(outdir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_fft_cylindrical_first_n(self, ds_name, seeds, amp2d_list, k_par_edges, k_perp_edges, C):
        """Plot first N wedge-removed Fourier-amplitude cylindrical maps in one row."""
        if len(amp2d_list) == 0:
            return

        n_show = len(amp2d_list)
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show + 1.5, 4), constrained_layout=True)
        if n_show == 1:
            axes = [axes]

        vmin_global = np.inf
        vmax_global = -np.inf
        for arr in amp2d_list:
            pos = arr[arr > 0]
            if pos.size > 0:
                vmin_global = min(vmin_global, np.min(pos))
                vmax_global = max(vmax_global, np.max(pos))
        if not np.isfinite(vmin_global) or not np.isfinite(vmax_global) or vmin_global >= vmax_global:
            vmin_global, vmax_global = 1e-12, 1.0

        im = None
        for idx, (seed, amp2d) in enumerate(zip(seeds, amp2d_list)):
            ax = axes[idx]
            amp_plot = amp2d.copy().astype(float)
            amp_plot[amp_plot <= 0] = np.nan

            im = ax.pcolormesh(
                k_perp_edges,
                k_par_edges,
                amp_plot,
                norm=LogNorm(vmin=vmin_global, vmax=vmax_global),
                cmap="inferno",
                shading="flat",
            )

            kperp_line = np.linspace(k_perp_edges[0], k_perp_edges[-1], 200)
            kpar_line = C * kperp_line
            ax.plot(kperp_line, kpar_line, "w--", linewidth=1.2, label=f"Wedge (C={C:.2f})")

            ax.set_xlim(k_perp_edges[0], k_perp_edges[-1])
            ax.set_ylim(k_par_edges[0], k_par_edges[-1])
            ax.set_xlabel(r"$k_\perp\ [h/\mathrm{cMpc}]$", fontsize=10)
            if idx == 0:
                ax.set_ylabel(r"$k_\parallel\ [h/\mathrm{cMpc}]$", fontsize=10)
                ax.legend(fontsize=7, loc="upper left")
            ax.set_title(f"Seed {seed}", fontsize=10)
            ax.tick_params(labelsize=8)

        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(r"$|\tilde{\delta T_b}(k_\parallel, k_\perp)|\ [\mathrm{mK}\ (\mathrm{cMpc}/h)^{3/2}]$", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(
            f"{ds_name.capitalize()} Main: Wedge-Removed Fourier Amplitude (first {n_show})",
            fontsize=13,
            weight="bold",
        )

        outfile = os.path.join(self.plot_dir, f"{ds_name}_main_fft_wedge_removed_first{n_show}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_realspace_first_n(self, ds_name, seeds, slice2d_list, box, yslice_idx):
        """Plot first N wedge-filtered real-space middle-y slices in one row."""
        if len(slice2d_list) == 0:
            return

        n_show = len(slice2d_list)
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show + 1.5, 4), constrained_layout=True)
        if n_show == 1:
            axes = [axes]

        vmin_global = min(np.min(slc) for slc in slice2d_list)
        vmax_global = max(np.max(slc) for slc in slice2d_list)
        if vmin_global == vmax_global:
            vmin_global -= 1e-6
            vmax_global += 1e-6

        im = None
        for idx, (seed, slc) in enumerate(zip(seeds, slice2d_list)):
            ax = axes[idx]
            im = ax.imshow(
                slc,
                origin="lower",
                extent=[0.0, box, 0.0, box],
                cmap="viridis",
                vmin=vmin_global,
                vmax=vmax_global,
                interpolation="bicubic",
                aspect="auto",
            )

            ax.set_xlabel(r"$x\ [\mathrm{cMpc}/h]$", fontsize=10)
            if idx == 0:
                ax.set_ylabel(r"$z\ [\mathrm{cMpc}/h]$", fontsize=10)
            ax.set_title(f"Seed {seed}", fontsize=10)
            ax.tick_params(labelsize=8)

        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(r"$\delta T_b\ [\mathrm{mK}]$", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(
            f"{ds_name.capitalize()} Main: Wedge-Filtered Real-Space Maps (first {n_show}, y-slice={yslice_idx})",
            fontsize=13,
            weight="bold",
        )

        outfile = os.path.join(self.plot_dir, f"{ds_name}_main_realspace_wedge_filtered_first{n_show}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

# ==============================================================
# MAIN
# ==============================================================

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    vis = Visualizer(args.outdir, n_show=args.n_show)
    
    print(f"Running Wedge Filtering (Real -> Fourier -> Mask -> Real)")
    print(f"Redshift: {args.z}")

    print(f"Box: {args.box} cMpc/h, Ngrid: {args.ngrid}")

    k_par_edges, k_perp_edges = setup_k_bins(args.ngrid, args.box, args.nbins_par, args.nbins_perp)
    print(f"k-bin setup: linear bins with nbins_par={args.nbins_par}, nbins_perp={args.nbins_perp}")

    # 1. Compute Slope
    C = compute_wedge_slope(args.z, args.omega_m, args.omega_l)
    print(f"Wedge Slope C: {C:.6f}")

    # 2. Build Mask
    print("Building 3D Wedge Mask...")
    k_par, k_perp = get_3d_k_grids(args.ngrid, args.box)
    
    # Keep modes where k_par > C * k_perp
    # Remove modes where k_par <= C * k_perp
    # Note: This removes DC mode (0,0) as 0 <= 0 is True.
    # If standard mean subtraction is assumed, DC is already 0.
    mask = k_par > C * k_perp
    
    kept_frac = np.count_nonzero(mask) / mask.size
    print(f"Mask created. Fraction of modes kept: {kept_frac:.2%}")

    # Collect first-N main realizations for combined FFT plots.
    fft_plot_data = {
        "training": {"seeds": [], "amp2d": []},
        "validation": {"seeds": [], "amp2d": []},
    }
    real_plot_data = {
        "training": {"seeds": [], "slice2d": []},
        "validation": {"seeds": [], "slice2d": []},
    }
    y_mid = args.ngrid // 2

    # 3. Process Datasets
    datasets = {
        'training': {
            'input_dir': args.training_input_dir,
            'files': {
                'main': 'training_temp_plus_noise_ms_main.npy',
                'QHII_plus': 'training_temp_plus_noise_ms_QHII_plus.npy',
                'QHII_minus': 'training_temp_plus_noise_ms_QHII_minus.npy',
                'Mmin_plus': 'training_temp_plus_noise_ms_Mmin_plus.npy',
                'Mmin_minus': 'training_temp_plus_noise_ms_Mmin_minus.npy',
            },
            'output_prefix': 'training',
        },
        'validation': {
            'input_dir': args.validation_input_dir,
            'files': {
                'main': 'validation_temp_plus_noise_ms_main.npy',
                'QHII_plus': 'validation_temp_plus_noise_ms_QHII_plus.npy',
                'QHII_minus': 'validation_temp_plus_noise_ms_QHII_minus.npy',
                'Mmin_plus': 'validation_temp_plus_noise_ms_Mmin_plus.npy',
                'Mmin_minus': 'validation_temp_plus_noise_ms_Mmin_minus.npy',
            },
            'output_prefix': 'validation',
        },
    }

    for ds_name, ds_info in datasets.items():
        print(f"\nProcessing {ds_name.upper()} datasets...")
        
        for case_key, filename in ds_info['files'].items():
            infile = os.path.join(ds_info['input_dir'], filename)
            if not os.path.exists(infile):
                print(f"  [MISSING] {infile}")
                continue
                
            print(f"  Loading {filename}...")
            # Load dictionary of maps {seed: map}
            maps_dict = np.load(infile, allow_pickle=True).item()
            filtered_dict = {}
            
            # Use seed counter for first-N main realizations
            seed_count = 0
            
            # Process each seed
            for seed, map_data in tqdm(maps_dict.items(), desc=f"  Filtering {case_key}", ncols=80):
                # FFT
                # Use norm='backward' (default) -> unscaled transform
                # We will handle scaling strictly
                ft_before = np.fft.fftn(map_data)
                ft = ft_before.copy()
                
                # Apply Mask
                ft[~mask] = 0.0
                
                if case_key == 'main' and seed_count < vis.n_show:
                    # Match normalization used in PS pipeline: deltak has units mK (cMpc/h)^(3/2)
                    deltak = ft * (np.sqrt(args.box**3) / args.ngrid**3)
                    amp3d = np.abs(deltak)
                    amp2d = bin_cylindrical_stat(amp3d, k_par, k_perp, k_par_edges, k_perp_edges)

                    fft_plot_data[ds_name]["seeds"].append(seed)
                    fft_plot_data[ds_name]["amp2d"].append(amp2d)
                
                # Inverse FFT
                # The output should be real if input was real, but numerical noise gives small complex part
                map_filtered = np.fft.ifftn(ft).real
                
                filtered_dict[seed] = map_filtered.astype(np.float32) # Save space

                if case_key == 'main' and seed_count < vis.n_show:
                    # Middle y-slice with x on horizontal axis and z on vertical axis.
                    # map[:, y_mid, :] gives (x, z), transpose -> (z, x) for imshow.
                    map_slice = map_filtered[:, y_mid, :].T
                    real_plot_data[ds_name]["seeds"].append(seed)
                    real_plot_data[ds_name]["slice2d"].append(map_slice)
                
                seed_count += 1
            
            # Save output
            out_filename = f"{ds_info['output_prefix']}_wedge_filtered_{case_key}.npy"
            out_path = os.path.join(args.outdir, out_filename)
            np.save(out_path, filtered_dict)
            print(f"  -> Saved to {out_path}")
            
            del maps_dict, filtered_dict

    print("\nGenerating combined FFT wedge-removed plots...")
    for ds_name in ["training", "validation"]:
        vis.plot_fft_cylindrical_first_n(
            ds_name=ds_name,
            seeds=fft_plot_data[ds_name]["seeds"],
            amp2d_list=fft_plot_data[ds_name]["amp2d"],
            k_par_edges=k_par_edges,
            k_perp_edges=k_perp_edges,
            C=C,
        )

    print("Generating combined real-space wedge-filtered plots...")
    for ds_name in ["training", "validation"]:
        vis.plot_realspace_first_n(
            ds_name=ds_name,
            seeds=real_plot_data[ds_name]["seeds"],
            slice2d_list=real_plot_data[ds_name]["slice2d"],
            box=args.box,
            yslice_idx=y_mid,
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
