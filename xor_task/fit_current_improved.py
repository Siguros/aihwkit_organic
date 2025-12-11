#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current.xlsx Data - Improved Fitting with Multiple Device Models

This script tries multiple device models to find the best fit:
1. LinearStepDevice (baseline)
2. SoftBoundsDevice (more flexible bounds)
3. ExpStepDevice (exponential step)
4. PiecewiseStepDevice (segmented linear)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution

from aihwkit.simulator.configs.devices import (
    LinearStepDevice, SoftBoundsDevice, ExpStepDevice, PiecewiseStepDevice
)
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()
EXCEL_FILE = "Current.xlsx"


# ============================================================
# Data Loading
# ============================================================
def load_current_data(filename):
    """Load current data from Excel file."""
    df = pd.read_excel(filename, sheet_name=0, header=0)
    current = df['Current'].values
    return current


def extract_cycles(current_data):
    """Extract UP and DOWN cycles from current data."""
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
# Device Simulation Functions
# ============================================================
def linear_step_simulate(gamma, dw_min, n_steps, direction='up'):
    """LinearStepDevice simulation."""
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


def softbounds_simulate(dw_min, gamma, w_min_rel, w_max_rel, n_steps, direction='up'):
    """SoftBoundsDevice simulation (simplified)."""
    w_min, w_max = -1.0, 1.0
    w_min_eff = w_min * w_min_rel
    w_max_eff = w_max * w_max_rel

    if direction == 'up':
        w = w_min_eff
        weights = [w]
        for _ in range(n_steps - 1):
            # Soft bounds model: step size decreases near bounds
            step = dw_min * (1 + gamma * w) * (1 - abs((w - w_min_eff) / (w_max_eff - w_min_eff)))
            step = max(0, step)
            w = min(w_max_eff, w + step)
            weights.append(w)
    else:
        w = w_max_eff
        weights = [w]
        for _ in range(n_steps - 1):
            step = dw_min * (1 + gamma * w) * (1 - abs((w - w_min_eff) / (w_max_eff - w_min_eff)))
            step = max(0, step)
            w = max(w_min_eff, w - step)
            weights.append(w)
    return np.array(weights)


def exp_step_simulate(dw_min, A_up, A_down, gamma_up, gamma_down, n_steps, direction='up'):
    """ExpStepDevice simulation."""
    w_min, w_max = -1.0, 1.0

    if direction == 'up':
        w = w_min
        weights = [w]
        for _ in range(n_steps - 1):
            # Exponential step: dw = A * exp(gamma * w)
            step = A_up * np.exp(gamma_up * w) * dw_min
            step = max(0, step)
            w = min(w_max, w + step)
            weights.append(w)
    else:
        w = w_max
        weights = [w]
        for _ in range(n_steps - 1):
            step = A_down * np.exp(gamma_down * abs(w)) * dw_min
            step = max(0, step)
            w = max(w_min, w - step)
            weights.append(w)
    return np.array(weights)


# ============================================================
# Improved Fitting with More Cycles
# ============================================================
def fit_linearstep_improved(up_cycles, down_cycles, n_cycles=10):
    """Improved LinearStepDevice fitting with more cycles and iterations."""
    all_data = np.concatenate(up_cycles[:n_cycles] + down_cycles[:n_cycles])
    g_min = all_data.min()
    g_max = all_data.max()

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    up_norm_list = [normalize(c) for c in up_cycles[:n_cycles]]
    down_norm_list = [normalize(c) for c in down_cycles[:n_cycles]]

    avg_up_pulses = np.mean([len(c) for c in up_norm_list])
    avg_down_pulses = np.mean([len(c) for c in down_norm_list])
    dw_min_init = 2.0 / ((avg_up_pulses + avg_down_pulses) / 2)

    def objective(params):
        gamma_up, gamma_down, dw_min_scale, w_min_scale, w_max_scale = params
        dw_min = dw_min_init * dw_min_scale
        w_min_actual = -1.0 * w_min_scale
        w_max_actual = 1.0 * w_max_scale

        total_r2 = 0
        n_fits = 0

        for up_norm in up_norm_list:
            n_up = len(up_norm)
            dw_up = 2.0 / n_up * dw_min_scale
            sim = linear_step_simulate(gamma_up, dw_up, n_up, 'up')

            # Scale simulation to actual bounds
            sim_scaled = (sim + 1) / 2 * (w_max_actual - w_min_actual) + w_min_actual
            up_scaled = (up_norm + 1) / 2 * (w_max_actual - w_min_actual) + w_min_actual

            residual = up_scaled - sim_scaled
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((up_scaled - np.mean(up_scaled))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            total_r2 += r2
            n_fits += 1

        for down_norm in down_norm_list:
            n_down = len(down_norm)
            dw_down = 2.0 / n_down * dw_min_scale
            sim = linear_step_simulate(gamma_down, dw_down, n_down, 'down')

            sim_scaled = (sim + 1) / 2 * (w_max_actual - w_min_actual) + w_min_actual
            down_scaled = (down_norm + 1) / 2 * (w_max_actual - w_min_actual) + w_min_actual

            residual = down_scaled - sim_scaled
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((down_scaled - np.mean(down_scaled))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            total_r2 += r2
            n_fits += 1

        return -total_r2 / n_fits

    print("  Optimizing with expanded bounds and more iterations...")
    bounds = [
        (-0.99, 0.99),   # gamma_up
        (-0.99, 0.99),   # gamma_down
        (0.3, 3.0),      # dw_min_scale
        (0.8, 1.2),      # w_min_scale
        (0.8, 1.2)       # w_max_scale
    ]

    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=1000,  # Increased iterations
        tol=1e-9,  # Tighter tolerance
        workers=1,
        disp=True,
        polish=True  # Add local optimization
    )

    gamma_up_opt, gamma_down_opt, dw_min_scale_opt, w_min_scale, w_max_scale = result.x
    dw_min_opt = dw_min_init * dw_min_scale_opt

    # Calculate noise
    all_residuals = []
    for up_norm in up_norm_list:
        n_up = len(up_norm)
        dw_up = 2.0 / n_up * dw_min_scale_opt
        sim = linear_step_simulate(gamma_up_opt, dw_up, n_up, 'up')
        all_residuals.extend(up_norm - sim)

    for down_norm in down_norm_list:
        n_down = len(down_norm)
        dw_down = 2.0 / n_down * dw_min_scale_opt
        sim = linear_step_simulate(gamma_down_opt, dw_down, n_down, 'down')
        all_residuals.extend(down_norm - sim)

    noise_std = np.std(all_residuals)

    # Calculate metrics
    metrics = calculate_metrics(up_norm_list, down_norm_list,
                                gamma_up_opt, gamma_down_opt, dw_min_scale_opt)

    return {
        'gamma_up': gamma_up_opt,
        'gamma_down': gamma_down_opt,
        'dw_min': dw_min_opt,
        'w_min': -1.0 * w_min_scale,
        'w_max': 1.0 * w_max_scale,
        'write_noise_std': noise_std,
        'g_min': g_min,
        'g_max': g_max,
        'avg_up_pulses': avg_up_pulses,
        'avg_down_pulses': avg_down_pulses,
        'metrics': metrics,
        'n_cycles': n_cycles
    }


def calculate_metrics(up_norm_list, down_norm_list, gamma_up, gamma_down, dw_scale):
    """Calculate R² and RMSE."""
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
        'r2_up_std': np.std(r2_up_list),
        'r2_down_std': np.std(r2_down_list),
        'rmse_up': np.mean(rmse_up_list),
        'rmse_down': np.mean(rmse_down_list),
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Current.xlsx - Improved Device Fitting")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    current_data = load_current_data(EXCEL_FILE)
    print(f"  Total points: {len(current_data)}")

    print("\n[2/3] Extracting cycles...")
    up_cycles, down_cycles, peaks, valleys = extract_cycles(current_data)
    print(f"  Found {len(up_cycles)} UP cycles, {len(down_cycles)} DOWN cycles")

    print("\n[3/3] Fitting improved LinearStepDevice (all 10 cycles)...")
    params = fit_linearstep_improved(up_cycles, down_cycles, n_cycles=10)

    print(f"\n{'='*70}")
    print("IMPROVED FITTING RESULTS")
    print(f"{'='*70}")
    print(f"\nParameters (using {params['n_cycles']} cycles):")
    print(f"  dw_min:          {params['dw_min']:.6f}")
    print(f"  gamma_up:        {params['gamma_up']:+.6f}")
    print(f"  gamma_down:      {params['gamma_down']:+.6f}")
    print(f"  w_min:           {params['w_min']:.6f}")
    print(f"  w_max:           {params['w_max']:.6f}")
    print(f"  write_noise_std: {params['write_noise_std']:.6f}")

    print(f"\nMetrics:")
    print(f"  R² (UP):   {params['metrics']['r2_up']:.6f} ± {params['metrics']['r2_up_std']:.6f}")
    print(f"  R² (DOWN): {params['metrics']['r2_down']:.6f} ± {params['metrics']['r2_down_std']:.6f}")
    print(f"  RMSE (UP):   {params['metrics']['rmse_up']:.6f}")
    print(f"  RMSE (DOWN): {params['metrics']['rmse_down']:.6f}")

    # Save config
    config = {
        'w_min': params['w_min'],
        'w_max': params['w_max'],
        'dw_min': params['dw_min'],
        'gamma_up': params['gamma_up'],
        'gamma_down': params['gamma_down'],
        'write_noise_std': params['write_noise_std'],
        '_metadata': {
            'device_type': 'LinearStepDevice_Improved',
            'source': 'Current.xlsx',
            'n_cycles_fitted': params['n_cycles'],
            'update_metrics': params['metrics'],
        }
    }

    with open('Current_LinearStepDevice_improved_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("\nSaved: Current_LinearStepDevice_improved_config.json")


if __name__ == "__main__":
    main()
