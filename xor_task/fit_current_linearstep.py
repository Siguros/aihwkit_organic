#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current.xlsx Data Fitting to AIHWKit LinearStepDevice

This script fits Current.xlsx data to LinearStepDevice model.
Based on 6T1C LinearStepDevice fitting approach.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution

from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()

# File paths
EXCEL_FILE = "Current.xlsx"


# ============================================================
# PART 1: Load and Prepare Data
# ============================================================
def load_current_data(filename):
    """Load current data from Excel file."""
    df = pd.read_excel(filename, sheet_name=0, header=0)
    current = df['Current'].values
    return current


def extract_cycles(current_data):
    """Extract UP and DOWN cycles from current data."""
    # Find peaks and valleys
    peaks, _ = find_peaks(current_data, distance=800,
                         prominence=(current_data.max()-current_data.min())*0.3)
    valleys, _ = find_peaks(-current_data, distance=800,
                           prominence=(current_data.max()-current_data.min())*0.3)

    up_cycles = []
    down_cycles = []

    # First UP: start -> first peak
    if len(peaks) > 0:
        up_cycles.append(current_data[:peaks[0]+1])

    # Extract alternating UP and DOWN cycles
    for i, peak in enumerate(peaks):
        if i < len(valleys):
            down_cycles.append(current_data[peak:valleys[i]+1])
            if i + 1 < len(peaks):
                up_cycles.append(current_data[valleys[i]:peaks[i+1]+1])

    # Last DOWN cycle if available
    if len(peaks) > len(valleys) and peaks[-1] < len(current_data) - 1:
        down_cycles.append(current_data[peaks[-1]:])

    return up_cycles, down_cycles, peaks, valleys


# ============================================================
# PART 2: LinearStepDevice Update Model
# ============================================================
def linear_step_simulate(gamma, dw_min, n_steps, direction='up'):
    """
    Simulate LinearStepDevice pulse response.

    LinearStepDevice update formula:
        w_new = w - dw_min * (1 + gamma * w / w_max) * (1 + noise)

    For normalized weights [-1, 1]:
        step_size = dw_min * (1 + gamma * w)
    """
    w_min, w_max = -1.0, 1.0

    if direction == 'up':
        w = w_min
        weights = [w]
        for _ in range(n_steps - 1):
            # gamma is typically negative for decreasing step size
            step = dw_min * (1 + gamma * w)
            step = max(0, step)  # step size cannot be negative
            w = min(w_max, w + step)
            weights.append(w)
    else:  # down
        w = w_max
        weights = [w]
        for _ in range(n_steps - 1):
            step = dw_min * (1 + gamma * w)
            step = max(0, step)
            w = max(w_min, w - step)
            weights.append(w)

    return np.array(weights)


def fit_update_parameters(up_cycles, down_cycles, n_cycles=3):
    """
    Fit LinearStepDevice parameters from cycling data.

    Parameters to fit:
        - gamma_up: slope for up direction
        - gamma_down: slope for down direction
        - dw_min: base step size
        - write_noise_std: cycle-to-cycle noise
    """
    # Get global normalization range
    all_data = np.concatenate(up_cycles[:n_cycles] + down_cycles[:n_cycles])
    g_min = all_data.min()
    g_max = all_data.max()

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    # Prepare normalized data
    up_norm_list = [normalize(c) for c in up_cycles[:n_cycles]]
    down_norm_list = [normalize(c) for c in down_cycles[:n_cycles]]

    # Calculate average pulse counts
    avg_up_pulses = np.mean([len(c) for c in up_norm_list])
    avg_down_pulses = np.mean([len(c) for c in down_norm_list])

    # Initial dw_min estimate
    dw_min_init = 2.0 / ((avg_up_pulses + avg_down_pulses) / 2)

    def objective(params):
        """Objective function: minimize negative R²."""
        gamma_up, gamma_down, dw_min_scale = params
        dw_min = dw_min_init * dw_min_scale

        total_r2 = 0
        n_fits = 0

        for up_norm in up_norm_list:
            n_up = len(up_norm)
            # Adjust dw_min for this cycle length
            dw_up = 2.0 / n_up * dw_min_scale
            sim = linear_step_simulate(gamma_up, dw_up, n_up, 'up')

            residual = up_norm - sim
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((up_norm - np.mean(up_norm))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            total_r2 += r2
            n_fits += 1

        for down_norm in down_norm_list:
            n_down = len(down_norm)
            dw_down = 2.0 / n_down * dw_min_scale
            sim = linear_step_simulate(gamma_down, dw_down, n_down, 'down')

            residual = down_norm - sim
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((down_norm - np.mean(down_norm))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            total_r2 += r2
            n_fits += 1

        return -total_r2 / n_fits

    # Optimize
    print("  Optimizing parameters...")
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.5, 2.0)]
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=500, tol=1e-7,
        workers=1,
        disp=False
    )

    gamma_up_opt, gamma_down_opt, dw_min_scale_opt = result.x
    dw_min_opt = dw_min_init * dw_min_scale_opt

    # Calculate noise from residuals
    all_residuals_up = []
    all_residuals_down = []

    for up_norm in up_norm_list:
        n_up = len(up_norm)
        dw_up = 2.0 / n_up * dw_min_scale_opt
        sim = linear_step_simulate(gamma_up_opt, dw_up, n_up, 'up')
        all_residuals_up.extend(up_norm - sim)

    for down_norm in down_norm_list:
        n_down = len(down_norm)
        dw_down = 2.0 / n_down * dw_min_scale_opt
        sim = linear_step_simulate(gamma_down_opt, dw_down, n_down, 'down')
        all_residuals_down.extend(down_norm - sim)

    noise_std = np.std(all_residuals_up + all_residuals_down)

    # Calculate final metrics
    metrics = calculate_update_metrics(
        up_norm_list, down_norm_list,
        gamma_up_opt, gamma_down_opt, dw_min_scale_opt
    )

    return {
        'gamma_up': gamma_up_opt,
        'gamma_down': gamma_down_opt,
        'dw_min': dw_min_opt,
        'write_noise_std': noise_std,
        'g_min': g_min,
        'g_max': g_max,
        'avg_up_pulses': avg_up_pulses,
        'avg_down_pulses': avg_down_pulses,
        'metrics': metrics,
    }


def calculate_update_metrics(up_norm_list, down_norm_list, gamma_up, gamma_down, dw_scale):
    """Calculate R² and RMSE for update fitting."""
    r2_up_list = []
    r2_down_list = []
    rmse_up_list = []
    rmse_down_list = []

    for up_norm in up_norm_list:
        n_up = len(up_norm)
        dw_up = 2.0 / n_up * dw_scale
        sim = linear_step_simulate(gamma_up, dw_up, n_up, 'up')

        residual = up_norm - sim
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((up_norm - np.mean(up_norm))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residual**2))

        r2_up_list.append(r2)
        rmse_up_list.append(rmse)

    for down_norm in down_norm_list:
        n_down = len(down_norm)
        dw_down = 2.0 / n_down * dw_scale
        sim = linear_step_simulate(gamma_down, dw_down, n_down, 'down')

        residual = down_norm - sim
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((down_norm - np.mean(down_norm))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residual**2))

        r2_down_list.append(r2)
        rmse_down_list.append(rmse)

    return {
        'r2_up': np.mean(r2_up_list),
        'r2_down': np.mean(r2_down_list),
        'rmse_up': np.mean(rmse_up_list),
        'rmse_down': np.mean(rmse_down_list),
    }


# ============================================================
# PART 3: Visualization
# ============================================================
def create_update_plots(up_cycles, down_cycles, params, output_prefix):
    """Create update characteristic plots."""
    g_min = params['g_min']
    g_max = params['g_max']
    gamma_up = params['gamma_up']
    gamma_down = params['gamma_down']

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(min(3, len(up_cycles))):
        up_norm = normalize(up_cycles[i])
        down_norm = normalize(down_cycles[i])

        n_up = len(up_norm)
        n_down = len(down_norm)

        dw_up = 2.0 / n_up
        dw_down = 2.0 / n_down

        sim_up = linear_step_simulate(gamma_up, dw_up, n_up, 'up')
        sim_down = linear_step_simulate(gamma_down, dw_down, n_down, 'down')

        # UP plot
        axes[0, i].plot(up_norm, 'r-', alpha=0.8, linewidth=1, label='Actual')
        axes[0, i].plot(sim_up, 'k--', linewidth=1.5, label='Model')
        axes[0, i].set_title(f'UP Cycle {i+1}')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlabel('Pulse')
        axes[0, i].set_ylabel('Weight')

        # DOWN plot
        axes[1, i].plot(down_norm, 'b-', alpha=0.8, linewidth=1, label='Actual')
        axes[1, i].plot(sim_down, 'k--', linewidth=1.5, label='Model')
        axes[1, i].set_title(f'DOWN Cycle {i+1}')
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlabel('Pulse')
        axes[1, i].set_ylabel('Weight')

    metrics = params['metrics']
    fig.suptitle(f'LinearStepDevice Update Fitting\n'
                 f'gamma_up={gamma_up:.4f}, gamma_down={gamma_down:.4f}\n'
                 f'R²(UP)={metrics["r2_up"]:.4f}, R²(DOWN)={metrics["r2_down"]:.4f}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_update_fitting.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_combined_summary_plot(up_cycles, down_cycles, current_data,
                                  peaks, valleys, update_params, output_prefix):
    """Create a combined summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    g_min = update_params['g_min']
    g_max = update_params['g_max']
    gamma_up = update_params['gamma_up']
    gamma_down = update_params['gamma_down']

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    # Plot 1: Full current data with peaks/valleys
    ax1 = axes[0, 0]
    ax1.plot(current_data, 'b-', alpha=0.7, linewidth=0.5)
    ax1.plot(peaks, current_data[peaks], 'r^', markersize=8, label=f'Peaks ({len(peaks)})')
    ax1.plot(valleys, current_data[valleys], 'gv', markersize=8, label=f'Valleys ({len(valleys)})')
    ax1.set_xlabel('Pulse Number', fontsize=10)
    ax1.set_ylabel('Current (A)', fontsize=10)
    ax1.set_title('Full Current Data', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: UP pulse response
    ax2 = axes[0, 1]
    up_norm = normalize(up_cycles[0])
    n_up = len(up_norm)
    dw_up = 2.0 / n_up
    sim_up = linear_step_simulate(gamma_up, dw_up, n_up, 'up')

    ax2.plot(up_norm, 'r-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax2.plot(sim_up, 'k--', linewidth=2,
             label=f'Model (γ={gamma_up:.3f}, R²={update_params["metrics"]["r2_up"]:.4f})')
    ax2.set_xlabel('Pulse Number', fontsize=10)
    ax2.set_ylabel('Normalized Weight', fontsize=10)
    ax2.set_title('UP Pulse Response', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: DOWN pulse response
    ax3 = axes[1, 0]
    down_norm = normalize(down_cycles[0])
    n_down = len(down_norm)
    dw_down = 2.0 / n_down
    sim_down = linear_step_simulate(gamma_down, dw_down, n_down, 'down')

    ax3.plot(down_norm, 'b-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax3.plot(sim_down, 'k--', linewidth=2,
             label=f'Model (γ={gamma_down:.3f}, R²={update_params["metrics"]["r2_down"]:.4f})')
    ax3.set_xlabel('Pulse Number', fontsize=10)
    ax3.set_ylabel('Normalized Weight', fontsize=10)
    ax3.set_title('DOWN Pulse Response', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Parameter summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    ════════════════════════════════════════
         Current.xlsx LinearStepDevice
              Fitted Parameters
    ════════════════════════════════════════

    UPDATE CHARACTERISTICS:
    ────────────────────────────────────────
      dw_min:           {update_params['dw_min']:.6f}
      gamma_up:         {gamma_up:.6f}
      gamma_down:       {gamma_down:.6f}
      write_noise_std:  {update_params['write_noise_std']:.6f}

      Fitting Quality:
        R² (UP):   {update_params['metrics']['r2_up']:.4f}
        R² (DOWN): {update_params['metrics']['r2_down']:.4f}
        RMSE (UP):   {update_params['metrics']['rmse_up']:.4f}
        RMSE (DOWN): {update_params['metrics']['rmse_down']:.4f}

    DATA STATISTICS:
    ────────────────────────────────────────
      Current range: {g_min:.2e} ~ {g_max:.2e} A
      UP cycles: {len(up_cycles)}
      DOWN cycles: {len(down_cycles)}
      Avg UP pulses: {update_params['avg_up_pulses']:.0f}
      Avg DOWN pulses: {update_params['avg_down_pulses']:.0f}

    ════════════════════════════════════════
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Current.xlsx Device Characterization Summary',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# PART 4: Main Fitting Pipeline
# ============================================================
def main():
    print("=" * 70)
    print("Current.xlsx LinearStepDevice Fitting")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Load Data
    # --------------------------------------------------------
    print("\n[1/4] Loading data...")

    current_data = load_current_data(EXCEL_FILE)

    print(f"  Current data: {len(current_data)} points")
    print(f"  Range: {current_data.min():.2e} ~ {current_data.max():.2e} A")

    # --------------------------------------------------------
    # Step 2: Extract Cycles
    # --------------------------------------------------------
    print("\n[2/4] Extracting cycles...")

    up_cycles, down_cycles, peaks, valleys = extract_cycles(current_data)

    print(f"  Found {len(up_cycles)} UP cycles, {len(down_cycles)} DOWN cycles")
    for i, (up, down) in enumerate(zip(up_cycles[:3], down_cycles[:3])):
        print(f"    Cycle {i+1}: UP={len(up)} pulses, DOWN={len(down)} pulses")

    # --------------------------------------------------------
    # Step 3: Fit Update Parameters
    # --------------------------------------------------------
    print("\n[3/4] Fitting update parameters...")

    update_params = fit_update_parameters(up_cycles, down_cycles, n_cycles=3)

    print(f"\n  Update Parameters:")
    print(f"    dw_min:          {update_params['dw_min']:.6f}")
    print(f"    gamma_up:        {update_params['gamma_up']:.6f}")
    print(f"    gamma_down:      {update_params['gamma_down']:.6f}")
    print(f"    write_noise_std: {update_params['write_noise_std']:.6f}")
    print(f"\n  Fitting Quality:")
    print(f"    R² (UP):   {update_params['metrics']['r2_up']:.6f}")
    print(f"    R² (DOWN): {update_params['metrics']['r2_down']:.6f}")
    print(f"    RMSE (UP):   {update_params['metrics']['rmse_up']:.6f}")
    print(f"    RMSE (DOWN): {update_params['metrics']['rmse_down']:.6f}")

    # --------------------------------------------------------
    # Step 4: Create Device and Save
    # --------------------------------------------------------
    print("\n[4/4] Creating device and saving configuration...")

    # Create LinearStepDevice
    device = LinearStepDevice(
        # Weight bounds
        w_min=-1.0,
        w_max=1.0,

        # Update parameters
        dw_min=update_params['dw_min'],
        gamma_up=update_params['gamma_up'],
        gamma_down=update_params['gamma_down'],

        # Noise
        write_noise_std=update_params['write_noise_std'],
        dw_min_std=0.3,  # default cycle-to-cycle variation

        # Other settings
        mult_noise=True,
        mean_bound_reference=True,

        # Device-to-device variation
        dw_min_dtod=0.1,
        w_min_dtod=0.05,
        w_max_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
    )

    print("\n  Created LinearStepDevice:")
    print(device)

    # Create RPU config
    rpu_config = SingleRPUConfig(device=device)

    # Save configuration
    config = {
        # Weight bounds
        'w_min': -1.0,
        'w_max': 1.0,

        # Update parameters
        'dw_min': update_params['dw_min'],
        'gamma_up': update_params['gamma_up'],
        'gamma_down': update_params['gamma_down'],
        'write_noise_std': update_params['write_noise_std'],

        # Other settings
        'mult_noise': True,
        'mean_bound_reference': True,
        'dw_min_std': 0.3,
        'dw_min_dtod': 0.1,
        'w_min_dtod': 0.05,
        'w_max_dtod': 0.05,
        'gamma_up_dtod': 0.05,
        'gamma_down_dtod': 0.05,

        # Metadata
        '_metadata': {
            'device_type': 'LinearStepDevice',
            'source': 'Current.xlsx',
            'update_metrics': update_params['metrics'],
            'current_range': {
                'min': float(update_params['g_min']),
                'max': float(update_params['g_max']),
            },
            'avg_pulses_per_cycle': {
                'up': float(update_params['avg_up_pulses']),
                'down': float(update_params['avg_down_pulses']),
            },
        }
    }

    with open('Current_LinearStepDevice_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("\n  Saved: Current_LinearStepDevice_config.json")

    # Create visualizations
    print("\n  Creating visualizations...")

    create_update_plots(up_cycles, down_cycles, update_params, 'Current_LinearStep')
    print("    Saved: Current_LinearStep_update_fitting.png")

    create_combined_summary_plot(up_cycles, down_cycles, current_data,
                                  peaks, valleys, update_params, 'Current_LinearStep')
    print("    Saved: Current_LinearStep_summary.png")

    # Plot device using AIHWKit visualization
    try:
        fig = plot_device_compact(device, w_noise=0.0,
                                   n_steps=int(update_params['avg_up_pulses']),
                                   use_cuda=USE_CUDA)
        plt.savefig('Current_LinearStep_device_response.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: Current_LinearStep_device_response.png")
    except Exception as e:
        print(f"    Warning: Could not create device response plot: {e}")

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("FITTING COMPLETE!")
    print("=" * 70)

    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          Current.xlsx LinearStepDevice Final Parameters          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  UPDATE CHARACTERISTICS                                          ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    dw_min:          {update_params['dw_min']:.6f}                               ║
    ║    gamma_up:        {update_params['gamma_up']:+.6f}                               ║
    ║    gamma_down:      {update_params['gamma_down']:+.6f}                               ║
    ║    write_noise_std: {update_params['write_noise_std']:.6f}                               ║
    ║    R² (UP/DOWN):    {update_params['metrics']['r2_up']:.4f} / {update_params['metrics']['r2_down']:.4f}                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  GENERATED FILES                                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    1. Current_LinearStepDevice_config.json                       ║
    ║    2. Current_LinearStep_update_fitting.png                      ║
    ║    3. Current_LinearStep_summary.png                             ║
    ║    4. Current_LinearStep_device_response.png                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
