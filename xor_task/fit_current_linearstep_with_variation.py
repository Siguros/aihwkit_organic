#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current.xlsx Data Fitting to LinearStepDevice with Cycle-to-Cycle Variation

This script:
1. Fits each cycle individually to get parameter distributions
2. Calculates mean and std for each parameter
3. Uses std as dtod (device-to-device variation) parameters
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution, minimize

from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()
EXCEL_FILE = "Current.xlsx"


# ============================================================
# Data Loading
# ============================================================
def load_current_data(filename):
    df = pd.read_excel(filename, sheet_name=0, header=0)
    current = df['Current'].values
    return current


def extract_cycles(current_data):
    peaks, _ = find_peaks(current_data, distance=800,
                         prominence=(current_data.max()-current_data.min())*0.3)
    valleys, _ = find_peaks(-current_data, distance=800,
                           prominence=(current_data.max()-current_data.min())*0.3)

    up_cycles = []
    down_cycles = []

    if len(peaks) > 0:
        up_cycles.append(current_data[:peaks[0]+1])

    for i, peak in enumerate(peaks):
        if i < len(valleys):
            down_cycles.append(current_data[peak:valleys[i]+1])
            if i + 1 < len(peaks):
                up_cycles.append(current_data[valleys[i]:peaks[i+1]+1])

    if len(peaks) > len(valleys) and peaks[-1] < len(current_data) - 1:
        down_cycles.append(current_data[peaks[-1]:])

    return up_cycles, down_cycles, peaks, valleys


# ============================================================
# LinearStepDevice Simulation
# ============================================================
def linear_step_simulate(gamma, dw_min, n_steps, direction='up'):
    w_min, w_max = -1.0, 1.0
    if direction == 'up':
        w = w_min
        weights = [w]
        for _ in range(n_steps - 1):
            step = dw_min * (1 + gamma * w)
            step = max(0, step)
            w = min(w_max, w + step)
            weights.append(w)
    else:
        w = w_max
        weights = [w]
        for _ in range(n_steps - 1):
            step = dw_min * (1 + gamma * w)
            step = max(0, step)
            w = max(w_min, w - step)
            weights.append(w)
    return np.array(weights)


# ============================================================
# Fit Individual Cycles
# ============================================================
def fit_single_cycle(cycle_data, g_min, g_max, direction='up'):
    """Fit a single cycle to get individual parameters."""

    # Normalize
    cycle_norm = (cycle_data - g_min) / (g_max - g_min) * 2 - 1
    n_pulses = len(cycle_norm)
    dw_min_init = 2.0 / n_pulses

    def objective(params):
        gamma, dw_min_scale = params
        dw_min = dw_min_init * dw_min_scale

        sim = linear_step_simulate(gamma, dw_min, n_pulses, direction)

        residual = cycle_norm - sim
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((cycle_norm - np.mean(cycle_norm))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return -r2  # Minimize negative R²

    # Optimize
    bounds = [(-0.99, 0.99), (0.5, 2.0)]
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=300, tol=1e-7,
        workers=1, disp=False
    )

    gamma_opt, dw_min_scale_opt = result.x
    dw_min_opt = dw_min_init * dw_min_scale_opt

    # Calculate metrics
    sim = linear_step_simulate(gamma_opt, dw_min_opt, n_pulses, direction)
    residual = cycle_norm - sim
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((cycle_norm - np.mean(cycle_norm))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residual**2))

    return {
        'gamma': gamma_opt,
        'dw_min': dw_min_opt,
        'r2': r2,
        'rmse': rmse,
        'n_pulses': n_pulses
    }


def fit_all_cycles_individually(up_cycles, down_cycles, n_cycles=10):
    """Fit each cycle individually and get parameter distributions."""

    # Get global normalization
    all_data = np.concatenate(up_cycles[:n_cycles] + down_cycles[:n_cycles])
    g_min = all_data.min()
    g_max = all_data.max()

    print(f"\n  Fitting each cycle individually...")

    # Fit UP cycles
    up_params = []
    for i, up_cycle in enumerate(up_cycles[:n_cycles]):
        result = fit_single_cycle(up_cycle, g_min, g_max, direction='up')
        up_params.append(result)
        print(f"    UP Cycle {i+1}: gamma={result['gamma']:+.4f}, dw_min={result['dw_min']:.6f}, R²={result['r2']:.4f}")

    # Fit DOWN cycles
    down_params = []
    for i, down_cycle in enumerate(down_cycles[:n_cycles]):
        result = fit_single_cycle(down_cycle, g_min, g_max, direction='down')
        down_params.append(result)
        print(f"    DOWN Cycle {i+1}: gamma={result['gamma']:+.4f}, dw_min={result['dw_min']:.6f}, R²={result['r2']:.4f}")

    # Calculate statistics
    gamma_up_values = [p['gamma'] for p in up_params]
    gamma_down_values = [p['gamma'] for p in down_params]
    dw_min_up_values = [p['dw_min'] for p in up_params]
    dw_min_down_values = [p['dw_min'] for p in down_params]

    gamma_up_mean = np.mean(gamma_up_values)
    gamma_up_std = np.std(gamma_up_values)
    gamma_down_mean = np.mean(gamma_down_values)
    gamma_down_std = np.std(gamma_down_values)

    dw_min_up_mean = np.mean(dw_min_up_values)
    dw_min_up_std = np.std(dw_min_up_values)
    dw_min_down_mean = np.mean(dw_min_down_values)
    dw_min_down_std = np.std(dw_min_down_values)

    # Overall dw_min
    dw_min_mean = np.mean(dw_min_up_values + dw_min_down_values)
    dw_min_std = np.std(dw_min_up_values + dw_min_down_values)

    # Calculate dtod as relative std (CV = std/mean)
    gamma_up_dtod = gamma_up_std / abs(gamma_up_mean) if abs(gamma_up_mean) > 0 else 0
    gamma_down_dtod = gamma_down_std / abs(gamma_down_mean) if abs(gamma_down_mean) > 0 else 0
    dw_min_dtod = dw_min_std / dw_min_mean if dw_min_mean > 0 else 0

    # Calculate write_noise_std from residuals
    all_residuals = []
    for i, up_cycle in enumerate(up_cycles[:n_cycles]):
        cycle_norm = (up_cycle - g_min) / (g_max - g_min) * 2 - 1
        sim = linear_step_simulate(up_params[i]['gamma'], up_params[i]['dw_min'],
                                   len(cycle_norm), 'up')
        all_residuals.extend(cycle_norm - sim)

    for i, down_cycle in enumerate(down_cycles[:n_cycles]):
        cycle_norm = (down_cycle - g_min) / (g_max - g_min) * 2 - 1
        sim = linear_step_simulate(down_params[i]['gamma'], down_params[i]['dw_min'],
                                   len(cycle_norm), 'down')
        all_residuals.extend(cycle_norm - sim)

    write_noise_std = np.std(all_residuals)

    # Overall metrics
    r2_up_mean = np.mean([p['r2'] for p in up_params])
    r2_up_std = np.std([p['r2'] for p in up_params])
    r2_down_mean = np.mean([p['r2'] for p in down_params])
    r2_down_std = np.std([p['r2'] for p in down_params])

    rmse_up_mean = np.mean([p['rmse'] for p in up_params])
    rmse_down_mean = np.mean([p['rmse'] for p in down_params])

    return {
        'gamma_up_mean': gamma_up_mean,
        'gamma_up_std': gamma_up_std,
        'gamma_up_dtod': gamma_up_dtod,
        'gamma_down_mean': gamma_down_mean,
        'gamma_down_std': gamma_down_std,
        'gamma_down_dtod': gamma_down_dtod,
        'dw_min_mean': dw_min_mean,
        'dw_min_std': dw_min_std,
        'dw_min_dtod': dw_min_dtod,
        'write_noise_std': write_noise_std,
        'g_min': g_min,
        'g_max': g_max,
        'n_cycles': n_cycles,
        'up_params': up_params,
        'down_params': down_params,
        'gamma_up_values': gamma_up_values,
        'gamma_down_values': gamma_down_values,
        'dw_min_values': dw_min_up_values + dw_min_down_values,
        'metrics': {
            'r2_up_mean': r2_up_mean,
            'r2_up_std': r2_up_std,
            'r2_down_mean': r2_down_mean,
            'r2_down_std': r2_down_std,
            'rmse_up_mean': rmse_up_mean,
            'rmse_down_mean': rmse_down_mean,
        }
    }


# ============================================================
# Visualization
# ============================================================
def create_variation_plots(params, output_prefix):
    """Create plots showing parameter variations across cycles."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_cycles = params['n_cycles']

    # Plot 1: gamma values
    ax1 = axes[0, 0]
    x = np.arange(1, n_cycles + 1)
    ax1.plot(x, params['gamma_up_values'], 'ro-', label='UP', markersize=8, linewidth=2)
    ax1.plot(x, params['gamma_down_values'], 'bo-', label='DOWN', markersize=8, linewidth=2)
    ax1.axhline(y=params['gamma_up_mean'], color='r', linestyle='--', alpha=0.5,
                label=f'UP mean={params["gamma_up_mean"]:.3f}±{params["gamma_up_std"]:.3f}')
    ax1.axhline(y=params['gamma_down_mean'], color='b', linestyle='--', alpha=0.5,
                label=f'DOWN mean={params["gamma_down_mean"]:.3f}±{params["gamma_down_std"]:.3f}')
    ax1.set_xlabel('Cycle Number', fontsize=11)
    ax1.set_ylabel('gamma', fontsize=11)
    ax1.set_title('Gamma Values Across Cycles', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)

    # Plot 2: dw_min values
    ax2 = axes[0, 1]
    ax2.plot(x, [p['dw_min'] for p in params['up_params']], 'ro-',
             label='UP', markersize=8, linewidth=2)
    ax2.plot(x, [p['dw_min'] for p in params['down_params']], 'bo-',
             label='DOWN', markersize=8, linewidth=2)
    ax2.axhline(y=params['dw_min_mean'], color='k', linestyle='--', alpha=0.5,
                label=f'Overall mean={params["dw_min_mean"]:.4f}±{params["dw_min_std"]:.4f}')
    ax2.set_xlabel('Cycle Number', fontsize=11)
    ax2.set_ylabel('dw_min', fontsize=11)
    ax2.set_title('dw_min Values Across Cycles', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)

    # Plot 3: R² values
    ax3 = axes[1, 0]
    ax3.plot(x, [p['r2'] for p in params['up_params']], 'ro-',
             label='UP', markersize=8, linewidth=2)
    ax3.plot(x, [p['r2'] for p in params['down_params']], 'bo-',
             label='DOWN', markersize=8, linewidth=2)
    ax3.axhline(y=params['metrics']['r2_up_mean'], color='r', linestyle='--', alpha=0.5,
                label=f'UP mean={params["metrics"]["r2_up_mean"]:.4f}')
    ax3.axhline(y=params['metrics']['r2_down_mean'], color='b', linestyle='--', alpha=0.5,
                label=f'DOWN mean={params["metrics"]["r2_down_mean"]:.4f}')
    ax3.set_xlabel('Cycle Number', fontsize=11)
    ax3.set_ylabel('R²', fontsize=11)
    ax3.set_title('Fitting Quality (R²) Across Cycles', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x)
    ax3.set_ylim([0.95, 1.0])

    # Plot 4: Parameter distribution histograms
    ax4 = axes[1, 1]
    ax4.hist(params['gamma_up_values'], bins=8, alpha=0.6, color='red', label='gamma_up')
    ax4.hist(params['gamma_down_values'], bins=8, alpha=0.6, color='blue', label='gamma_down')
    ax4.axvline(x=params['gamma_up_mean'], color='red', linestyle='--', linewidth=2)
    ax4.axvline(x=params['gamma_down_mean'], color='blue', linestyle='--', linewidth=2)
    ax4.set_xlabel('gamma value', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Gamma Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Cycle-to-Cycle Variation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cycle_variation.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Current.xlsx LinearStepDevice with Cycle-to-Cycle Variation")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    current_data = load_current_data(EXCEL_FILE)
    print(f"  Current data: {len(current_data)} points")

    print("\n[2/4] Extracting cycles...")
    up_cycles, down_cycles, peaks, valleys = extract_cycles(current_data)
    print(f"  Found {len(up_cycles)} UP cycles, {len(down_cycles)} DOWN cycles")

    print("\n[3/4] Fitting each cycle individually...")
    n_cycles = min(10, len(up_cycles), len(down_cycles))
    params = fit_all_cycles_individually(up_cycles, down_cycles, n_cycles=n_cycles)

    print(f"\n{'='*70}")
    print("PARAMETER STATISTICS (Cycle-to-Cycle Variation)")
    print(f"{'='*70}")

    print(f"\ngamma_up:")
    print(f"  Mean:  {params['gamma_up_mean']:+.6f}")
    print(f"  Std:   {params['gamma_up_std']:.6f}")
    print(f"  dtod (CV): {params['gamma_up_dtod']:.6f} ({params['gamma_up_dtod']*100:.2f}%)")

    print(f"\ngamma_down:")
    print(f"  Mean:  {params['gamma_down_mean']:+.6f}")
    print(f"  Std:   {params['gamma_down_std']:.6f}")
    print(f"  dtod (CV): {params['gamma_down_dtod']:.6f} ({params['gamma_down_dtod']*100:.2f}%)")

    print(f"\ndw_min:")
    print(f"  Mean:  {params['dw_min_mean']:.6f}")
    print(f"  Std:   {params['dw_min_std']:.6f}")
    print(f"  dtod (CV): {params['dw_min_dtod']:.6f} ({params['dw_min_dtod']*100:.2f}%)")

    print(f"\nwrite_noise_std: {params['write_noise_std']:.6f}")

    print(f"\nFitting Quality:")
    print(f"  R² (UP):   {params['metrics']['r2_up_mean']:.6f} ± {params['metrics']['r2_up_std']:.6f}")
    print(f"  R² (DOWN): {params['metrics']['r2_down_mean']:.6f} ± {params['metrics']['r2_down_std']:.6f}")
    print(f"  RMSE (UP):   {params['metrics']['rmse_up_mean']:.6f}")
    print(f"  RMSE (DOWN): {params['metrics']['rmse_down_mean']:.6f}")

    print("\n[4/4] Creating device and saving...")

    # Create LinearStepDevice with dtod parameters
    device = LinearStepDevice(
        w_min=-1.0,
        w_max=1.0,
        dw_min=params['dw_min_mean'],
        dw_min_dtod=params['dw_min_dtod'],
        dw_min_std=0.3,
        gamma_up=params['gamma_up_mean'],
        gamma_up_dtod=params['gamma_up_dtod'],
        gamma_down=params['gamma_down_mean'],
        gamma_down_dtod=params['gamma_down_dtod'],
        write_noise_std=params['write_noise_std'],
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        mult_noise=True,
        mean_bound_reference=True,
    )

    print("\n  Created LinearStepDevice with cycle variation:")
    print(device)

    # Save configuration
    config = {
        'w_min': -1.0,
        'w_max': 1.0,
        'dw_min': params['dw_min_mean'],
        'dw_min_dtod': params['dw_min_dtod'],
        'dw_min_std': 0.3,
        'gamma_up': params['gamma_up_mean'],
        'gamma_up_dtod': params['gamma_up_dtod'],
        'gamma_down': params['gamma_down_mean'],
        'gamma_down_dtod': params['gamma_down_dtod'],
        'write_noise_std': params['write_noise_std'],
        'w_min_dtod': 0.0,
        'w_max_dtod': 0.0,
        'mult_noise': True,
        'mean_bound_reference': True,
        '_metadata': {
            'device_type': 'LinearStepDevice_with_CycleToCycleVariation',
            'source': 'Current.xlsx',
            'n_cycles_fitted': n_cycles,
            'fitting_method': 'individual_cycle_fitting',
            'parameter_statistics': {
                'gamma_up': {
                    'mean': params['gamma_up_mean'],
                    'std': params['gamma_up_std'],
                    'values': params['gamma_up_values'],
                },
                'gamma_down': {
                    'mean': params['gamma_down_mean'],
                    'std': params['gamma_down_std'],
                    'values': params['gamma_down_values'],
                },
                'dw_min': {
                    'mean': params['dw_min_mean'],
                    'std': params['dw_min_std'],
                    'values': params['dw_min_values'],
                }
            },
            'metrics': params['metrics'],
            'current_range': {
                'min': float(params['g_min']),
                'max': float(params['g_max']),
            }
        }
    }

    with open('Current_LinearStepDevice_with_variation_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("  Saved: Current_LinearStepDevice_with_variation_config.json")

    # Create visualizations
    print("\n  Creating visualizations...")
    create_variation_plots(params, 'Current_LinearStep_variation')
    print("    Saved: Current_LinearStep_variation_cycle_variation.png")

    # Plot device response
    try:
        avg_pulses = np.mean([p['n_pulses'] for p in params['up_params']])
        fig = plot_device_compact(device, w_noise=0.0,
                                   n_steps=int(avg_pulses),
                                   use_cuda=USE_CUDA)
        plt.savefig('Current_LinearStep_variation_device_response.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: Current_LinearStep_variation_device_response.png")
    except Exception as e:
        print(f"    Warning: Could not create device response plot: {e}")

    print("\n" + "=" * 70)
    print("FITTING COMPLETE!")
    print("=" * 70)

    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║    LinearStepDevice with Cycle-to-Cycle Variation Modeling      ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  MEAN PARAMETERS (from {n_cycles} cycles)                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    dw_min:       {params['dw_min_mean']:.6f} ± {params['dw_min_std']:.6f} (dtod={params['dw_min_dtod']:.3f})    ║
    ║    gamma_up:     {params['gamma_up_mean']:+.6f} ± {params['gamma_up_std']:.6f} (dtod={params['gamma_up_dtod']:.3f})    ║
    ║    gamma_down:   {params['gamma_down_mean']:+.6f} ± {params['gamma_down_std']:.6f} (dtod={params['gamma_down_dtod']:.3f})    ║
    ║    noise_std:    {params['write_noise_std']:.6f}                               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  FITTING QUALITY                                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    R² (UP):      {params['metrics']['r2_up_mean']:.6f} ± {params['metrics']['r2_up_std']:.6f}                    ║
    ║    R² (DOWN):    {params['metrics']['r2_down_mean']:.6f} ± {params['metrics']['r2_down_std']:.6f}                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  GENERATED FILES                                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    1. Current_LinearStepDevice_with_variation_config.json        ║
    ║    2. Current_LinearStep_variation_cycle_variation.png           ║
    ║    3. Current_LinearStep_variation_device_response.png           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
