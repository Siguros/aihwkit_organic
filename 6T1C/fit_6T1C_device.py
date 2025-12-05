#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6T1C_result.xlsx 데이터를 AIHWKit PiecewiseStepDevice로 Fitting

Based on xor_task/fit_current_data.py approach
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution

from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.configs import PiecewiseStepDevice, SingleRPUConfig
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()

# File path
EXCEL_FILE = "6T1C_result.xlsx"
SHEET_NAME = "가장&선형대칭"
N_SEGMENTS = 10


def load_6T1C_data(filename, sheet_name):
    """Load 6T1C data from Excel file.

    Returns:
        conductance: array of conductance values
        pulse_numbers: array of pulse numbers
    """
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    pulse_numbers = df[2].dropna().values
    conductance = df[3].dropna().values
    return conductance, pulse_numbers


def extract_cycles(conductance):
    """Extract individual UP and DOWN cycles from conductance data.

    Returns:
        up_cycles: list of UP half-cycle data arrays
        down_cycles: list of DOWN half-cycle data arrays
    """
    # Find peaks (max) and valleys (min)
    peaks, _ = find_peaks(conductance, distance=200, prominence=50)
    valleys, _ = find_peaks(-conductance, distance=200, prominence=50)

    print(f"Detected {len(peaks)} peaks and {len(valleys)} valleys")
    print(f"Peak positions: {peaks}")
    print(f"Valley positions: {valleys}")

    up_cycles = []
    down_cycles = []

    # First UP cycle: start -> first peak
    up_cycles.append(conductance[:peaks[0]+1])

    # Subsequent cycles
    for i, peak in enumerate(peaks):
        if i < len(valleys):
            # DOWN: peak -> valley
            down_cycles.append(conductance[peak:valleys[i]+1])

            if i + 1 < len(peaks):
                # UP: valley -> next peak
                up_cycles.append(conductance[valleys[i]:peaks[i+1]+1])

    # Last DOWN cycle (if data ends low)
    if len(peaks) > len(valleys):
        # Data ends going down after last peak
        down_cycles.append(conductance[peaks[-1]:])

    print(f"\nExtracted {len(up_cycles)} UP cycles and {len(down_cycles)} DOWN cycles")
    for i, cyc in enumerate(up_cycles):
        print(f"  UP cycle {i+1}: {len(cyc)} pulses, {cyc.min():.0f} -> {cyc.max():.0f}")
    for i, cyc in enumerate(down_cycles):
        print(f"  DOWN cycle {i+1}: {len(cyc)} pulses, {cyc.max():.0f} -> {cyc.min():.0f}")

    return up_cycles, down_cycles


def fit_device_parameters(up_cycles, down_cycles, n_segments=10, use_n_cycles=3):
    """Fit PiecewiseStepDevice parameters from cycle data.

    Args:
        up_cycles: list of UP half-cycle arrays
        down_cycles: list of DOWN half-cycle arrays
        n_segments: number of piecewise segments
        use_n_cycles: number of cycles to use for fitting

    Returns:
        dict with device parameters and metadata
    """
    # Use specified number of cycles
    up_data_list = up_cycles[:use_n_cycles]
    down_data_list = down_cycles[:use_n_cycles]

    # Get global min/max across all cycles for normalization
    all_up = np.concatenate(up_data_list)
    all_down = np.concatenate(down_data_list)

    g_min = min(all_up.min(), all_down.min())
    g_max = max(all_up.max(), all_down.max())

    print(f"\nGlobal conductance range: {g_min:.2f} ~ {g_max:.2f}")

    # Average cycle lengths for dw_min calculation
    avg_up_pulses = np.mean([len(c) for c in up_data_list])
    avg_down_pulses = np.mean([len(c) for c in down_data_list])
    avg_pulses = (avg_up_pulses + avg_down_pulses) / 2

    # dw_min: average step size (normalized weight range is 2: from -1 to 1)
    dw_min = 2.0 / avg_pulses

    print(f"\nAverage UP pulses: {avg_up_pulses:.0f}")
    print(f"Average DOWN pulses: {avg_down_pulses:.0f}")
    print(f"dw_min = 2 / {avg_pulses:.0f} = {dw_min:.6f}")

    # Normalize each cycle to [-1, 1] and calculate piecewise values
    def normalize_cycle(data, g_min, g_max):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    def get_piecewise_from_histogram(norm_data, n_segments, dw_min_val):
        """Calculate piecewise values from histogram of normalized data."""
        counts, edges = np.histogram(norm_data, bins=n_segments, range=[-1, 1])
        segment_size = 2.0 / n_segments
        # Step size in each segment = segment_size / counts
        step_sizes = np.where(counts > 0, segment_size / counts, 1.0)
        # Piecewise = step_size / dw_min
        piecewise = step_sizes / dw_min_val
        return piecewise, counts

    # Aggregate piecewise values across cycles
    all_up_piecewise = []
    all_down_piecewise = []

    for up_cyc in up_data_list:
        up_norm = normalize_cycle(up_cyc, g_min, g_max)
        pw, counts = get_piecewise_from_histogram(up_norm, n_segments, dw_min)
        all_up_piecewise.append(pw)

    for down_cyc in down_data_list:
        down_norm = normalize_cycle(down_cyc, g_min, g_max)
        pw, counts = get_piecewise_from_histogram(down_norm, n_segments, dw_min)
        all_down_piecewise.append(pw)

    # Average piecewise values across cycles
    piecewise_up = np.mean(all_up_piecewise, axis=0)
    piecewise_down = np.mean(all_down_piecewise, axis=0)

    print(f"\nInitial piecewise_up: {piecewise_up}")
    print(f"Initial piecewise_down: {piecewise_down}")

    # Calculate up_down asymmetry
    center = n_segments // 2
    up_down = (piecewise_up[center] - piecewise_down[center]) * dw_min

    # Calculate noise from step variance
    all_noise = []
    for up_cyc in up_data_list:
        up_norm = normalize_cycle(up_cyc, g_min, g_max)
        up_diff = np.diff(up_norm)
        all_noise.append(np.std(up_diff))
    for down_cyc in down_data_list:
        down_norm = normalize_cycle(down_cyc, g_min, g_max)
        down_diff = -np.diff(down_norm)
        all_noise.append(np.std(down_diff))

    noise_std = np.mean(all_noise)

    print(f"\nup_down: {up_down:.6f}")
    print(f"write_noise_std: {noise_std:.6f}")

    return {
        'dw_min': dw_min,
        'up_down': up_down,
        'write_noise_std': noise_std,
        'piecewise_up': piecewise_up.tolist(),
        'piecewise_down': piecewise_down.tolist(),
        'g_min': g_min,
        'g_max': g_max,
        'avg_up_pulses': avg_up_pulses,
        'avg_down_pulses': avg_down_pulses,
        'n_segments': n_segments,
        'n_cycles_used': use_n_cycles,
    }


def simulate_piecewise_response(piecewise_values, dw_min_val, n_steps, direction='up'):
    """Simulate pulse response using piecewise step model."""
    w_min, w_max = -1.0, 1.0
    n_segments = len(piecewise_values)

    if direction == 'up':
        w = w_min
        weights = [w]
        for _ in range(n_steps - 1):
            seg_idx = int((w - w_min) / (w_max - w_min) * n_segments)
            seg_idx = max(0, min(n_segments - 1, seg_idx))
            dw = dw_min_val * piecewise_values[seg_idx]
            w = min(w_max, w + dw)
            weights.append(w)
    else:
        w = w_max
        weights = [w]
        for _ in range(n_steps - 1):
            seg_idx = int((w - w_min) / (w_max - w_min) * n_segments)
            seg_idx = max(0, min(n_segments - 1, seg_idx))
            dw = dw_min_val * piecewise_values[seg_idx]
            w = max(w_min, w - dw)
            weights.append(w)

    return np.array(weights)


def optimize_piecewise_params(up_cycles, down_cycles, params, n_cycles=3):
    """Optimize piecewise parameters using differential evolution."""

    n_segments = params['n_segments']
    g_min = params['g_min']
    g_max = params['g_max']

    # Prepare normalized data for fitting
    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    up_norm_list = [normalize(c) for c in up_cycles[:n_cycles]]
    down_norm_list = [normalize(c) for c in down_cycles[:n_cycles]]

    def objective(x):
        """Objective: minimize negative sum of R² values."""
        pw_up = x[:n_segments]
        pw_down = x[n_segments:]

        total_r2 = 0
        n_fits = 0

        # Evaluate UP cycles
        for up_norm in up_norm_list:
            n_up = len(up_norm)
            dw_up = 2.0 / n_up
            sim = simulate_piecewise_response(pw_up, dw_up, n_up, 'up')

            residual = up_norm - sim
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((up_norm - np.mean(up_norm))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            total_r2 += r2
            n_fits += 1

        # Evaluate DOWN cycles
        for down_norm in down_norm_list:
            n_down = len(down_norm)
            dw_down = 2.0 / n_down
            sim = simulate_piecewise_response(pw_down, dw_down, n_down, 'down')

            residual = down_norm - sim
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((down_norm - np.mean(down_norm))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            total_r2 += r2
            n_fits += 1

        return -total_r2 / n_fits  # Negative for minimization

    # Initial guess from histogram-based parameters
    x0 = np.array(params['piecewise_up'] + params['piecewise_down'])

    # Bounds
    bounds = [(0.1, 3.0)] * (2 * n_segments)

    print("\nOptimizing piecewise parameters...")
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=500,
        tol=1e-6,
        disp=False,
        workers=1
    )

    optimized_pw_up = result.x[:n_segments].tolist()
    optimized_pw_down = result.x[n_segments:].tolist()

    # Calculate final metrics
    final_r2_up = []
    final_r2_down = []
    final_rmse_up = []
    final_rmse_down = []

    for up_norm in up_norm_list:
        n_up = len(up_norm)
        dw_up = 2.0 / n_up
        sim = simulate_piecewise_response(optimized_pw_up, dw_up, n_up, 'up')

        residual = up_norm - sim
        rmse = np.sqrt(np.mean(residual**2))
        r2 = 1 - np.sum(residual**2) / np.sum((up_norm - np.mean(up_norm))**2)
        final_r2_up.append(r2)
        final_rmse_up.append(rmse)

    for down_norm in down_norm_list:
        n_down = len(down_norm)
        dw_down = 2.0 / n_down
        sim = simulate_piecewise_response(optimized_pw_down, dw_down, n_down, 'down')

        residual = down_norm - sim
        rmse = np.sqrt(np.mean(residual**2))
        r2 = 1 - np.sum(residual**2) / np.sum((down_norm - np.mean(down_norm))**2)
        final_r2_down.append(r2)
        final_rmse_down.append(rmse)

    avg_r2_up = np.mean(final_r2_up)
    avg_r2_down = np.mean(final_r2_down)
    avg_rmse_up = np.mean(final_rmse_up)
    avg_rmse_down = np.mean(final_rmse_down)

    print(f"\nOptimization Results:")
    print(f"  UP   - Avg R²: {avg_r2_up:.6f}, Avg RMSE: {avg_rmse_up:.6f}")
    print(f"  DOWN - Avg R²: {avg_r2_down:.6f}, Avg RMSE: {avg_rmse_down:.6f}")

    return {
        'piecewise_up': optimized_pw_up,
        'piecewise_down': optimized_pw_down,
        'r2_up': avg_r2_up,
        'r2_down': avg_r2_down,
        'rmse_up': avg_rmse_up,
        'rmse_down': avg_rmse_down,
    }


def create_visualization(up_cycles, down_cycles, params, optimized, output_prefix):
    """Create visualization plots."""

    g_min = params['g_min']
    g_max = params['g_max']
    n_segments = params['n_segments']

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    # Plot 1: Model vs Actual Data (first cycle)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # UP comparison
    ax1 = axes[0, 0]
    up_norm = normalize(up_cycles[0])
    n_up = len(up_norm)
    dw_up = 2.0 / n_up
    sim_up = simulate_piecewise_response(optimized['piecewise_up'], dw_up, n_up, 'up')

    ax1.plot(up_norm, 'r-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax1.plot(sim_up, 'k--', linewidth=2, label=f'Model (R²={optimized["r2_up"]:.4f})')
    ax1.set_xlabel('Pulse Number')
    ax1.set_ylabel('Normalized Conductance')
    ax1.set_title('UP Pulse: Model vs Actual Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # DOWN comparison
    ax2 = axes[0, 1]
    down_norm = normalize(down_cycles[0])
    n_down = len(down_norm)
    dw_down = 2.0 / n_down
    sim_down = simulate_piecewise_response(optimized['piecewise_down'], dw_down, n_down, 'down')

    ax2.plot(down_norm, 'b-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax2.plot(sim_down, 'k--', linewidth=2, label=f'Model (R²={optimized["r2_down"]:.4f})')
    ax2.set_xlabel('Pulse Number')
    ax2.set_ylabel('Normalized Conductance')
    ax2.set_title('DOWN Pulse: Model vs Actual Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Piecewise parameters
    ax3 = axes[1, 0]
    x_seg = np.arange(n_segments)
    width = 0.35
    ax3.bar(x_seg - width/2, optimized['piecewise_up'], width, label='UP', alpha=0.8, color='red')
    ax3.bar(x_seg + width/2, optimized['piecewise_down'], width, label='DOWN', alpha=0.8, color='blue')
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Piecewise Value')
    ax3.set_title('Optimized Piecewise Parameters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_seg)

    # Residual error
    ax4 = axes[1, 1]
    res_up = up_norm - sim_up
    res_down = down_norm - sim_down
    ax4.plot(res_up, 'r-', alpha=0.7, linewidth=1, label=f'UP Error (RMSE={optimized["rmse_up"]:.4f})')
    ax4.plot(res_down, 'b-', alpha=0.7, linewidth=1, label=f'DOWN Error (RMSE={optimized["rmse_down"]:.4f})')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Pulse Number')
    ax4.set_ylabel('Residual (Actual - Model)')
    ax4.set_title('Residual Error Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'6T1C PiecewiseStepDevice Fitting Results\n'
                 f'UP: R²={optimized["r2_up"]:.4f}, RMSE={optimized["rmse_up"]:.4f} | '
                 f'DOWN: R²={optimized["r2_down"]:.4f}, RMSE={optimized["rmse_down"]:.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_fitting_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_fitting_result.png")

    # Plot 2: All cycles comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, (up_cyc, down_cyc) in enumerate(zip(up_cycles[:3], down_cycles[:3])):
        up_norm = normalize(up_cyc)
        down_norm = normalize(down_cyc)

        n_up = len(up_norm)
        n_down = len(down_norm)
        dw_up = 2.0 / n_up
        dw_down = 2.0 / n_down

        sim_up = simulate_piecewise_response(optimized['piecewise_up'], dw_up, n_up, 'up')
        sim_down = simulate_piecewise_response(optimized['piecewise_down'], dw_down, n_down, 'down')

        # UP
        axes[0, i].plot(up_norm, 'r-', alpha=0.8, linewidth=1, label='Actual')
        axes[0, i].plot(sim_up, 'k--', linewidth=1.5, label='Model')
        axes[0, i].set_title(f'UP Cycle {i+1}')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)

        # DOWN
        axes[1, i].plot(down_norm, 'b-', alpha=0.8, linewidth=1, label='Actual')
        axes[1, i].plot(sim_down, 'k--', linewidth=1.5, label='Model')
        axes[1, i].set_title(f'DOWN Cycle {i+1}')
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle('6T1C All Cycles: Model vs Actual Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_all_cycles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_all_cycles.png")


if __name__ == "__main__":
    print("=" * 80)
    print("6T1C Device Fitting to PiecewiseStepDevice")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading data from {EXCEL_FILE}, sheet '{SHEET_NAME}'...")
    conductance, pulse_numbers = load_6T1C_data(EXCEL_FILE, SHEET_NAME)
    print(f"Loaded {len(conductance)} data points")
    print(f"Conductance range: {conductance.min():.0f} ~ {conductance.max():.0f}")

    # Extract cycles
    print("\n" + "=" * 80)
    print("Extracting UP/DOWN cycles...")
    print("=" * 80)
    up_cycles, down_cycles = extract_cycles(conductance)

    # Fit initial parameters
    print("\n" + "=" * 80)
    print("Fitting initial device parameters...")
    print("=" * 80)
    params = fit_device_parameters(up_cycles, down_cycles, n_segments=N_SEGMENTS, use_n_cycles=3)

    # Optimize parameters
    print("\n" + "=" * 80)
    print("Optimizing piecewise parameters...")
    print("=" * 80)
    optimized = optimize_piecewise_params(up_cycles, down_cycles, params, n_cycles=3)

    # Create PiecewiseStepDevice
    print("\n" + "=" * 80)
    print("Creating PiecewiseStepDevice...")
    print("=" * 80)

    device = PiecewiseStepDevice(
        w_min=-1,
        w_max=1,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        dw_min_std=0.0,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        dw_min=params['dw_min'],
        up_down=params['up_down'],
        write_noise_std=params['write_noise_std'],
        piecewise_up=optimized['piecewise_up'],
        piecewise_down=optimized['piecewise_down'],
        apply_write_noise_on_set=True,
    )

    print("\nGenerated PiecewiseStepDevice:")
    print(device)

    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    create_visualization(up_cycles, down_cycles, params, optimized, "6T1C")

    # Visualize device response using AIHWKit
    fig = plot_device_compact(
        device,
        w_noise=0.0,
        n_steps=int(params['avg_up_pulses']),
        use_cuda=USE_CUDA
    )
    plt.savefig('6T1C_device_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 6T1C_device_response.png")

    # Save device configuration
    print("\n" + "=" * 80)
    print("Saving device configuration...")
    print("=" * 80)

    device_config = {
        'w_min': -1,
        'w_max': 1,
        'dw_min': params['dw_min'],
        'up_down': params['up_down'],
        'write_noise_std': params['write_noise_std'],
        'piecewise_up': optimized['piecewise_up'],
        'piecewise_down': optimized['piecewise_down'],
        'w_min_dtod': 0.0,
        'w_max_dtod': 0.0,
        'dw_min_dtod': 0.0,
        'dw_min_std': 0.0,
        'up_down_dtod': 0.0,
        'apply_write_noise_on_set': True,
        '_metadata': {
            'n_segments': N_SEGMENTS,
            'n_cycles_used': 3,
            'pulses_per_half_cycle': int(params['avg_up_pulses']),
            'g_min_original': params['g_min'],
            'g_max_original': params['g_max'],
            'metrics': {
                'r2_up': optimized['r2_up'],
                'r2_down': optimized['r2_down'],
                'rmse_up': optimized['rmse_up'],
                'rmse_down': optimized['rmse_down'],
            }
        }
    }

    with open('6T1C_device_config.json', 'w') as f:
        json.dump(device_config, f, indent=2)
    print("Saved: 6T1C_device_config.json")

    # Print summary
    print("\n" + "=" * 80)
    print("FITTING COMPLETE!")
    print("=" * 80)
    print(f"\nDevice Parameters:")
    print(f"  dw_min: {params['dw_min']:.6f} ({int(2.0/params['dw_min'])} states)")
    print(f"  up_down: {params['up_down']:.6f}")
    print(f"  write_noise_std: {params['write_noise_std']:.6f}")
    print(f"\nFitting Quality:")
    print(f"  UP   - R²: {optimized['r2_up']:.6f}, RMSE: {optimized['rmse_up']:.6f}")
    print(f"  DOWN - R²: {optimized['r2_down']:.6f}, RMSE: {optimized['rmse_down']:.6f}")
    print(f"\nGenerated Files:")
    print(f"  1. 6T1C_device_config.json - Device configuration")
    print(f"  2. 6T1C_fitting_result.png - Model vs Data comparison")
    print(f"  3. 6T1C_all_cycles.png - All cycles comparison")
    print(f"  4. 6T1C_device_response.png - AIHWKit device response")
