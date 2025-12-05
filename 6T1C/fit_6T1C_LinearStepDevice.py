#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6T1C Device Comprehensive Fitting to AIHWKit LinearStepDevice

This script fits both:
1. Update characteristics (gamma_up, gamma_down, dw_min, noise)
2. Retention characteristics (lifetime, reset)

Based on 6T1C_result.xlsx experimental data.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution, minimize, curve_fit
from math import exp

from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()

# File paths
EXCEL_FILE = "6T1C_result.xlsx"
UPDATE_SHEET = "가장&선형대칭"
RETENTION_SHEET = "Retention_결과"


# ============================================================
# PART 1: Load and Prepare Data
# ============================================================
def load_update_data(filename, sheet_name):
    """Load conductance cycling data for update fitting."""
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    conductance = df[3].dropna().values
    return conductance


def load_retention_data(filename, sheet_name):
    """Load retention data for decay fitting."""
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    time_min = df[8].iloc[1:].astype(float).values
    vcap = df[9].iloc[1:].astype(float).values
    return time_min, vcap


def extract_cycles(conductance):
    """Extract UP and DOWN cycles from conductance data."""
    peaks, _ = find_peaks(conductance, distance=200, prominence=50)
    valleys, _ = find_peaks(-conductance, distance=200, prominence=50)

    up_cycles = []
    down_cycles = []

    # First UP: start -> first peak
    up_cycles.append(conductance[:peaks[0]+1])

    for i, peak in enumerate(peaks):
        if i < len(valleys):
            down_cycles.append(conductance[peak:valleys[i]+1])
            if i + 1 < len(peaks):
                up_cycles.append(conductance[valleys[i]:peaks[i+1]+1])

    if len(peaks) > len(valleys):
        down_cycles.append(conductance[peaks[-1]:])

    return up_cycles, down_cycles


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
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.5, 2.0)]
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=500, tol=1e-7,
        workers=1
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
# PART 3: Retention (Decay) Model Fitting
# ============================================================
def fit_retention_parameters(time_min, vcap):
    """
    Fit retention parameters from Vcap decay data.

    Physical model: Vcap(t) = Vcap_0 * exp(-t/tau)

    AIHWKit model: w(t) = (w_0 - b) * (1 - delta)^n + b
                   where delta = 1/lifetime

    For capacitor: decay towards 0V, so b = -1 (normalized)
    """
    time_sec = time_min * 60

    # Fit exponential decay: Vcap(t) = V0 * exp(-t/tau)
    def exp_decay(t, V0, tau):
        return V0 * np.exp(-t / tau)

    popt, _ = curve_fit(exp_decay, time_sec, vcap, p0=[vcap[0], 50000])
    V0_fit, tau_fit = popt

    vcap_fit = exp_decay(time_sec, V0_fit, tau_fit)
    residual = vcap - vcap_fit
    r2 = 1 - np.sum(residual**2) / np.sum((vcap - np.mean(vcap))**2)
    rmse = np.sqrt(np.mean(residual**2))

    # Also try stretched exponential for comparison
    def stretched_exp(t, V0, tau, beta):
        return V0 * np.exp(-(t / tau) ** beta)

    try:
        popt_se, _ = curve_fit(stretched_exp, time_sec, vcap,
                               p0=[vcap[0], 50000, 1.0],
                               bounds=([0, 0, 0.1], [2, 1e7, 2.0]))
        V0_se, tau_se, beta_se = popt_se
        vcap_se = stretched_exp(time_sec, V0_se, tau_se, beta_se)
        r2_se = 1 - np.sum((vcap - vcap_se)**2) / np.sum((vcap - np.mean(vcap))**2)
    except:
        r2_se = 0
        tau_se = tau_fit
        beta_se = 1.0

    # Weight normalization: w = 2 * Vcap - 1 (0V -> -1, 1V -> +1)
    # For 6T1C: actual Vcap range is ~0 to ~1V
    # Decay target: 0V -> w = -1

    return {
        'tau_sec': tau_fit,
        'tau_min': tau_fit / 60,
        'V0': V0_fit,
        'reset': -1.0,  # decay target (0V normalized)
        'r2_exp': r2,
        'rmse_exp': rmse,
        'r2_stretched': r2_se,
        'tau_stretched': tau_se,
        'beta_stretched': beta_se,
    }


def calculate_lifetime(tau_sec, dt_batch_sec=1.0):
    """
    Calculate AIHWKit lifetime from physical tau.

    lifetime = 1 / delta
    delta = 1 - exp(-dt_batch / tau)
    """
    delta = 1 - exp(-dt_batch_sec / tau_sec)
    lifetime = 1 / delta if delta > 0 else float('inf')
    return lifetime, delta


# ============================================================
# PART 4: Visualization
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


def create_retention_plots(time_min, vcap, retention_params, output_prefix):
    """Create retention characteristic plots."""
    time_sec = time_min * 60
    tau_sec = retention_params['tau_sec']
    V0 = retention_params['V0']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Vcap vs time
    ax1 = axes[0, 0]
    ax1.scatter(time_min, vcap, s=100, c='blue', marker='o', label='Measured', zorder=5)

    time_dense = np.linspace(0, time_min.max() * 1.5, 200)
    vcap_model = V0 * np.exp(-time_dense * 60 / tau_sec)
    ax1.plot(time_dense, vcap_model, 'r-', linewidth=2,
             label=f'Model (τ={tau_sec/60:.0f}min)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Vcap (V)')
    ax1.set_title('Capacitor Retention')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized weight
    ax2 = axes[0, 1]
    w_actual = 2 * vcap - 1
    w_0 = w_actual[0]
    b = -1.0

    ax2.scatter(time_min, w_actual, s=100, c='blue', marker='o', label='Actual', zorder=5)
    w_model = (w_0 - b) * np.exp(-time_dense * 60 / tau_sec) + b
    ax2.plot(time_dense, w_model, 'r-', linewidth=2, label=f'AIHWKit Model (b={b})')
    ax2.axhline(y=b, color='gray', linestyle='--', alpha=0.5, label='Decay target')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Normalized Weight')
    ax2.set_title('Weight Domain Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1.2, 1.0])

    # Plot 3: Residual
    ax3 = axes[1, 0]
    vcap_fit = V0 * np.exp(-time_sec / tau_sec)
    residual = (vcap - vcap_fit) * 1000
    ax3.scatter(time_min, residual, s=80, c='red', marker='s')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Residual (mV)')
    ax3.set_title(f'Residual (RMSE={retention_params["rmse_exp"]*1000:.1f}mV)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Log scale
    ax4 = axes[1, 1]
    ax4.scatter(time_min, np.log(vcap), s=100, c='blue', marker='o')
    ax4.plot(time_dense, np.log(V0 * np.exp(-time_dense * 60 / tau_sec)), 'r-', linewidth=2)
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('log(Vcap)')
    ax4.set_title('Log Scale (linearity check)')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'6T1C Retention Fitting\n'
                 f'τ={tau_sec/60:.1f}min, R²={retention_params["r2_exp"]:.4f}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_retention_fitting.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_combined_summary_plot(up_cycles, down_cycles, time_min, vcap,
                                  update_params, retention_params, output_prefix):
    """Create a combined summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    g_min = update_params['g_min']
    g_max = update_params['g_max']
    gamma_up = update_params['gamma_up']
    gamma_down = update_params['gamma_down']

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    # Plot 1: UP pulse response
    ax1 = axes[0, 0]
    up_norm = normalize(up_cycles[0])
    n_up = len(up_norm)
    dw_up = 2.0 / n_up
    sim_up = linear_step_simulate(gamma_up, dw_up, n_up, 'up')

    ax1.plot(up_norm, 'r-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax1.plot(sim_up, 'k--', linewidth=2,
             label=f'Model (γ={gamma_up:.3f}, R²={update_params["metrics"]["r2_up"]:.4f})')
    ax1.set_xlabel('Pulse Number', fontsize=11)
    ax1.set_ylabel('Normalized Weight', fontsize=11)
    ax1.set_title('UP Pulse Response', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: DOWN pulse response
    ax2 = axes[0, 1]
    down_norm = normalize(down_cycles[0])
    n_down = len(down_norm)
    dw_down = 2.0 / n_down
    sim_down = linear_step_simulate(gamma_down, dw_down, n_down, 'down')

    ax2.plot(down_norm, 'b-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax2.plot(sim_down, 'k--', linewidth=2,
             label=f'Model (γ={gamma_down:.3f}, R²={update_params["metrics"]["r2_down"]:.4f})')
    ax2.set_xlabel('Pulse Number', fontsize=11)
    ax2.set_ylabel('Normalized Weight', fontsize=11)
    ax2.set_title('DOWN Pulse Response', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Retention
    ax3 = axes[1, 0]
    tau_sec = retention_params['tau_sec']
    V0 = retention_params['V0']

    ax3.scatter(time_min, vcap, s=100, c='blue', marker='o', label='Measured', zorder=5)
    time_dense = np.linspace(0, time_min.max() * 1.2, 200)
    vcap_model = V0 * np.exp(-time_dense * 60 / tau_sec)
    ax3.plot(time_dense, vcap_model, 'r-', linewidth=2,
             label=f'Model (τ={tau_sec/60:.0f}min, R²={retention_params["r2_exp"]:.4f})')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Decay target')
    ax3.set_xlabel('Time (min)', fontsize=11)
    ax3.set_ylabel('Vcap (V)', fontsize=11)
    ax3.set_title('Capacitor Retention', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Parameter summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate lifetime for different dt_batch
    lifetime_1s, _ = calculate_lifetime(tau_sec, 1.0)
    lifetime_1m, _ = calculate_lifetime(tau_sec, 60.0)

    summary_text = f"""
    ══════════════════════════════════════════
           6T1C LinearStepDevice Parameters
    ══════════════════════════════════════════

    UPDATE CHARACTERISTICS:
    ───────────────────────────────────────────
      dw_min:           {update_params['dw_min']:.6f}
      gamma_up:         {gamma_up:.6f}
      gamma_down:       {gamma_down:.6f}
      write_noise_std:  {update_params['write_noise_std']:.6f}

      Fitting Quality:
        R² (UP):   {update_params['metrics']['r2_up']:.4f}
        R² (DOWN): {update_params['metrics']['r2_down']:.4f}

    RETENTION CHARACTERISTICS:
    ───────────────────────────────────────────
      τ (time constant): {tau_sec:.0f} sec ({tau_sec/60:.1f} min)
      reset (b):         {retention_params['reset']:.1f}

      lifetime (dt=1sec):  {lifetime_1s:.0f}
      lifetime (dt=1min):  {lifetime_1m:.0f}

      Fitting Quality:
        R²: {retention_params['r2_exp']:.4f}

    ══════════════════════════════════════════
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('6T1C Device Characterization Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# PART 5: Main Fitting Pipeline
# ============================================================
def main():
    print("=" * 70)
    print("6T1C LinearStepDevice Comprehensive Fitting")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Load Data
    # --------------------------------------------------------
    print("\n[1/5] Loading data...")

    conductance = load_update_data(EXCEL_FILE, UPDATE_SHEET)
    time_min, vcap = load_retention_data(EXCEL_FILE, RETENTION_SHEET)

    print(f"  Update data: {len(conductance)} points")
    print(f"  Retention data: {len(time_min)} points ({time_min.min():.0f}-{time_min.max():.0f} min)")

    # --------------------------------------------------------
    # Step 2: Extract Cycles
    # --------------------------------------------------------
    print("\n[2/5] Extracting cycles...")

    up_cycles, down_cycles = extract_cycles(conductance)

    print(f"  Found {len(up_cycles)} UP cycles, {len(down_cycles)} DOWN cycles")
    for i, (up, down) in enumerate(zip(up_cycles[:3], down_cycles[:3])):
        print(f"    Cycle {i+1}: UP={len(up)} pulses, DOWN={len(down)} pulses")

    # --------------------------------------------------------
    # Step 3: Fit Update Parameters
    # --------------------------------------------------------
    print("\n[3/5] Fitting update parameters...")

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
    # Step 4: Fit Retention Parameters
    # --------------------------------------------------------
    print("\n[4/5] Fitting retention parameters...")

    retention_params = fit_retention_parameters(time_min, vcap)

    print(f"\n  Retention Parameters:")
    print(f"    τ (time constant): {retention_params['tau_sec']:.1f} sec = {retention_params['tau_min']:.1f} min")
    print(f"    V0:               {retention_params['V0']:.4f} V")
    print(f"    reset (b):        {retention_params['reset']:.1f}")
    print(f"\n  Fitting Quality:")
    print(f"    R² (exponential):  {retention_params['r2_exp']:.6f}")
    print(f"    R² (stretched):    {retention_params['r2_stretched']:.6f}")

    # Calculate lifetime for common dt_batch values
    print(f"\n  Lifetime calculations:")
    for dt_name, dt_sec in [('1 sec', 1.0), ('10 sec', 10.0), ('1 min', 60.0), ('10 min', 600.0)]:
        lifetime, delta = calculate_lifetime(retention_params['tau_sec'], dt_sec)
        print(f"    dt_batch = {dt_name:6s} -> lifetime = {lifetime:,.0f}")

    # --------------------------------------------------------
    # Step 5: Create Device and Save
    # --------------------------------------------------------
    print("\n[5/5] Creating device and saving configuration...")

    # Default dt_batch = 1 second
    dt_batch_default = 1.0
    lifetime_default, _ = calculate_lifetime(retention_params['tau_sec'], dt_batch_default)

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

        # Retention
        lifetime=lifetime_default,
        lifetime_dtod=0.1,  # 10% device-to-device variation
        reset=retention_params['reset'],
        reset_dtod=0.0,

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

        # Retention parameters
        'lifetime': lifetime_default,
        'lifetime_dtod': 0.1,
        'reset': retention_params['reset'],
        'reset_dtod': 0.0,

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
            'source': '6T1C_result.xlsx',
            'dt_batch_assumed': dt_batch_default,
            'physical_tau_sec': retention_params['tau_sec'],
            'physical_tau_min': retention_params['tau_min'],
            'update_metrics': update_params['metrics'],
            'retention_metrics': {
                'r2': retention_params['r2_exp'],
                'rmse': retention_params['rmse_exp'],
            },
            'conductance_range': {
                'g_min': update_params['g_min'],
                'g_max': update_params['g_max'],
            },
            'avg_pulses_per_cycle': {
                'up': update_params['avg_up_pulses'],
                'down': update_params['avg_down_pulses'],
            },
        }
    }

    with open('6T1C_LinearStepDevice_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("\n  Saved: 6T1C_LinearStepDevice_config.json")

    # Create visualizations
    print("\n  Creating visualizations...")

    create_update_plots(up_cycles, down_cycles, update_params, '6T1C_LinearStep')
    print("    Saved: 6T1C_LinearStep_update_fitting.png")

    create_retention_plots(time_min, vcap, retention_params, '6T1C_LinearStep')
    print("    Saved: 6T1C_LinearStep_retention_fitting.png")

    create_combined_summary_plot(up_cycles, down_cycles, time_min, vcap,
                                  update_params, retention_params, '6T1C_LinearStep')
    print("    Saved: 6T1C_LinearStep_summary.png")

    # Plot device using AIHWKit visualization
    try:
        fig = plot_device_compact(device, w_noise=0.0,
                                   n_steps=int(update_params['avg_up_pulses']),
                                   use_cuda=USE_CUDA)
        plt.savefig('6T1C_LinearStep_device_response.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    Saved: 6T1C_LinearStep_device_response.png")
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
    ║              6T1C LinearStepDevice Final Parameters              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  UPDATE CHARACTERISTICS                                          ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    dw_min:          {update_params['dw_min']:.6f}                               ║
    ║    gamma_up:        {update_params['gamma_up']:+.6f}                               ║
    ║    gamma_down:      {update_params['gamma_down']:+.6f}                               ║
    ║    write_noise_std: {update_params['write_noise_std']:.6f}                               ║
    ║    R² (UP/DOWN):    {update_params['metrics']['r2_up']:.4f} / {update_params['metrics']['r2_down']:.4f}                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  RETENTION CHARACTERISTICS                                       ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    τ (physical):    {retention_params['tau_sec']:.0f} sec ({retention_params['tau_min']:.1f} min)                   ║
    ║    reset (b):       {retention_params['reset']:.1f}                                        ║
    ║    lifetime:        {lifetime_default:.0f} (dt_batch=1sec)                    ║
    ║    R²:              {retention_params['r2_exp']:.4f}                                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  GENERATED FILES                                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    1. 6T1C_LinearStepDevice_config.json                          ║
    ║    2. 6T1C_LinearStep_update_fitting.png                         ║
    ║    3. 6T1C_LinearStep_retention_fitting.png                      ║
    ║    4. 6T1C_LinearStep_summary.png                                ║
    ║    5. 6T1C_LinearStep_device_response.png                        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
