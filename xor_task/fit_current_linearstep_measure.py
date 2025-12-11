#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current.xlsx Data Fitting to AIHWKit LinearStepDevice using fit_measurements

This script uses AIHWKit's built-in fit_measurements function to fit
Current.xlsx data to LinearStepDevice model.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.utils.fitting import fit_measurements
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


def prepare_pulse_response_data(up_cycles, down_cycles, n_cycles=3):
    """
    Prepare pulse and response data for fit_measurements.

    Returns:
        pulse_data: tuple of pulse arrays (+1 for up, -1 for down)
        response_data: tuple of normalized conductance arrays
        g_min, g_max: normalization range
    """
    # Get global normalization range
    all_data = np.concatenate(up_cycles[:n_cycles] + down_cycles[:n_cycles])
    g_min = all_data.min()
    g_max = all_data.max()

    def normalize(data):
        return (data - g_min) / (g_max - g_min) * 2 - 1

    pulse_data_list = []
    response_data_list = []

    # UP cycles: positive pulses
    for up_cycle in up_cycles[:n_cycles]:
        n_up = len(up_cycle)
        # Pulses: +1 for each step
        pulses = np.ones(n_up)
        # Response: normalized conductance
        response = normalize(up_cycle)

        pulse_data_list.append(pulses)
        response_data_list.append(response)

    # DOWN cycles: negative pulses
    for down_cycle in down_cycles[:n_cycles]:
        n_down = len(down_cycle)
        # Pulses: -1 for each step
        pulses = -np.ones(n_down)
        # Response: normalized conductance
        response = normalize(down_cycle)

        pulse_data_list.append(pulses)
        response_data_list.append(response)

    return tuple(pulse_data_list), tuple(response_data_list), g_min, g_max


# ============================================================
# PART 2: Fit LinearStepDevice using fit_measurements
# ============================================================
def fit_linearstep_device(pulse_data, response_data):
    """
    Fit LinearStepDevice parameters using aihwkit.utils.fitting.fit_measurements.

    Args:
        pulse_data: tuple of pulse arrays
        response_data: tuple of response arrays

    Returns:
        fit_result: lmfit result object
        fitted_device: fitted LinearStepDevice
        model_responses: model predictions
    """
    # Create base device configuration
    base_device = LinearStepDevice(
        w_min=-1.0,
        w_max=1.0,
    )

    # Define parameters to fit
    # Format: {param_name: (initial_value, min_value, max_value)}
    parameters = {
        'dw_min': (0.002, 0.0001, 0.1),
        'gamma_up': (0.0, -0.9, 0.9),
        'gamma_down': (0.0, -0.9, 0.9),
        'write_noise_std': (0.01, 0.0, 0.5),
    }

    print("  Fitting parameters using fit_measurements...")
    print(f"  Parameters to fit: {list(parameters.keys())}")

    # Fit the device
    fit_result, fitted_device, model_responses = fit_measurements(
        parameters=parameters,
        pulse_data=pulse_data,
        response_data=response_data,
        device_config=base_device,
        suppress_device_noise=True,
        max_pulses=1,
        n_traces=5,  # simulate 5 traces for averaging
        method='powell',
        verbose=True
    )

    return fit_result, fitted_device, model_responses


def calculate_fit_metrics(response_data, model_responses):
    """Calculate R² and RMSE for the fit."""
    r2_list = []
    rmse_list = []

    for i, (response, model) in enumerate(zip(response_data, model_responses)):
        # Ensure same length
        min_len = min(len(response), len(model))
        resp = response[:min_len].flatten() if response.ndim > 1 else response[:min_len]
        mod = model[:min_len].flatten() if model.ndim > 1 else model[:min_len]

        residual = resp - mod
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((resp - np.mean(resp))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residual**2))

        r2_list.append(r2)
        rmse_list.append(rmse)

    # Separate UP and DOWN metrics
    n_cycles = len(response_data) // 2
    r2_up = np.mean(r2_list[:n_cycles])
    r2_down = np.mean(r2_list[n_cycles:])
    rmse_up = np.mean(rmse_list[:n_cycles])
    rmse_down = np.mean(rmse_list[n_cycles:])

    return {
        'r2_up': r2_up,
        'r2_down': r2_down,
        'r2_all': np.mean(r2_list),
        'rmse_up': rmse_up,
        'rmse_down': rmse_down,
        'rmse_all': np.mean(rmse_list),
    }


# ============================================================
# PART 3: Visualization
# ============================================================
def create_fitting_plots(response_data, model_responses, fitted_device,
                         current_data, peaks, valleys, g_min, g_max, metrics,
                         output_prefix):
    """Create comprehensive fitting visualization."""

    n_cycles = len(response_data) // 2
    up_responses = response_data[:n_cycles]
    down_responses = response_data[n_cycles:]
    up_models = model_responses[:n_cycles]
    down_models = model_responses[n_cycles:]

    # Plot 1: Full data with cycles
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Full current data
    ax1 = axes[0, 0]
    ax1.plot(current_data, 'b-', alpha=0.7, linewidth=0.5)
    ax1.plot(peaks, current_data[peaks], 'r^', markersize=8, label=f'Peaks ({len(peaks)})')
    ax1.plot(valleys, current_data[valleys], 'gv', markersize=8, label=f'Valleys ({len(valleys)})')
    ax1.set_xlabel('Pulse Number', fontsize=10)
    ax1.set_ylabel('Current (A)', fontsize=10)
    ax1.set_title('Full Current Data', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # First UP cycle
    ax2 = axes[0, 1]
    resp_up = up_responses[0].flatten() if up_responses[0].ndim > 1 else up_responses[0]
    mod_up = up_models[0].flatten() if up_models[0].ndim > 1 else up_models[0]
    min_len = min(len(resp_up), len(mod_up))

    ax2.plot(resp_up[:min_len], 'r-', alpha=0.8, linewidth=1.5, label='Measured')
    ax2.plot(mod_up[:min_len], 'k--', linewidth=2, label='Model')
    ax2.set_xlabel('Pulse Number', fontsize=10)
    ax2.set_ylabel('Normalized Weight', fontsize=10)
    ax2.set_title(f'UP Cycle 1 (R²={metrics["r2_up"]:.4f})', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # First DOWN cycle
    ax3 = axes[1, 0]
    resp_down = down_responses[0].flatten() if down_responses[0].ndim > 1 else down_responses[0]
    mod_down = down_models[0].flatten() if down_models[0].ndim > 1 else down_models[0]
    min_len = min(len(resp_down), len(mod_down))

    ax3.plot(resp_down[:min_len], 'b-', alpha=0.8, linewidth=1.5, label='Measured')
    ax3.plot(mod_down[:min_len], 'k--', linewidth=2, label='Model')
    ax3.set_xlabel('Pulse Number', fontsize=10)
    ax3.set_ylabel('Normalized Weight', fontsize=10)
    ax3.set_title(f'DOWN Cycle 1 (R²={metrics["r2_down"]:.4f})', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Parameter summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    params_text = f"""
    ═══════════════════════════════════════════
      Current.xlsx LinearStepDevice Fitting
    ═══════════════════════════════════════════

    FITTED PARAMETERS:
    ───────────────────────────────────────────
      dw_min:          {fitted_device.dw_min:.6f}
      gamma_up:        {fitted_device.gamma_up:+.6f}
      gamma_down:      {fitted_device.gamma_down:+.6f}
      write_noise_std: {fitted_device.write_noise_std:.6f}

    FITTING QUALITY:
    ───────────────────────────────────────────
      R² (UP):     {metrics['r2_up']:.4f}
      R² (DOWN):   {metrics['r2_down']:.4f}
      R² (Overall): {metrics['r2_all']:.4f}

      RMSE (UP):   {metrics['rmse_up']:.4f}
      RMSE (DOWN): {metrics['rmse_down']:.4f}
      RMSE (Overall): {metrics['rmse_all']:.4f}

    DATA INFO:
    ───────────────────────────────────────────
      Current range: {g_min:.2e} ~ {g_max:.2e} A
      Cycles fitted: {n_cycles} UP, {n_cycles} DOWN

    ═══════════════════════════════════════════
    """

    ax4.text(0.05, 0.95, params_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Current.xlsx LinearStepDevice Fitting (fit_measurements)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_fitting_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_prefix}_fitting_summary.png")

    # Plot 2: All cycles comparison
    n_plots = min(3, n_cycles)
    fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_plots):
        # UP cycle
        resp_up = up_responses[i].flatten() if up_responses[i].ndim > 1 else up_responses[i]
        mod_up = up_models[i].flatten() if up_models[i].ndim > 1 else up_models[i]
        min_len = min(len(resp_up), len(mod_up))

        axes[0, i].plot(resp_up[:min_len], 'r-', alpha=0.8, linewidth=1, label='Measured')
        axes[0, i].plot(mod_up[:min_len], 'k--', linewidth=1.5, label='Model')
        axes[0, i].set_title(f'UP Cycle {i+1}')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlabel('Pulse')
        axes[0, i].set_ylabel('Weight')

        # DOWN cycle
        resp_down = down_responses[i].flatten() if down_responses[i].ndim > 1 else down_responses[i]
        mod_down = down_models[i].flatten() if down_models[i].ndim > 1 else down_models[i]
        min_len = min(len(resp_down), len(mod_down))

        axes[1, i].plot(resp_down[:min_len], 'b-', alpha=0.8, linewidth=1, label='Measured')
        axes[1, i].plot(mod_down[:min_len], 'k--', linewidth=1.5, label='Model')
        axes[1, i].set_title(f'DOWN Cycle {i+1}')
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlabel('Pulse')
        axes[1, i].set_ylabel('Weight')

    fig.suptitle(f'LinearStepDevice Fitting - All Cycles\n'
                 f'R²(UP)={metrics["r2_up"]:.4f}, R²(DOWN)={metrics["r2_down"]:.4f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_all_cycles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_prefix}_all_cycles.png")


# ============================================================
# PART 4: Main Fitting Pipeline
# ============================================================
def main():
    print("=" * 70)
    print("Current.xlsx LinearStepDevice Fitting using fit_measurements")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Load Data
    # --------------------------------------------------------
    print("\n[1/5] Loading data...")

    current_data = load_current_data(EXCEL_FILE)

    print(f"  Current data: {len(current_data)} points")
    print(f"  Range: {current_data.min():.2e} ~ {current_data.max():.2e} A")

    # --------------------------------------------------------
    # Step 2: Extract Cycles
    # --------------------------------------------------------
    print("\n[2/5] Extracting cycles...")

    up_cycles, down_cycles, peaks, valleys = extract_cycles(current_data)

    print(f"  Found {len(up_cycles)} UP cycles, {len(down_cycles)} DOWN cycles")
    for i, (up, down) in enumerate(zip(up_cycles[:3], down_cycles[:3])):
        print(f"    Cycle {i+1}: UP={len(up)} pulses, DOWN={len(down)} pulses")

    # --------------------------------------------------------
    # Step 3: Prepare Data for fit_measurements
    # --------------------------------------------------------
    print("\n[3/5] Preparing data for fit_measurements...")

    n_cycles = 3  # Use first 3 cycles for fitting
    pulse_data, response_data, g_min, g_max = prepare_pulse_response_data(
        up_cycles, down_cycles, n_cycles=n_cycles
    )

    print(f"  Prepared {len(pulse_data)} traces ({n_cycles} UP + {n_cycles} DOWN)")
    print(f"  Normalization range: {g_min:.2e} ~ {g_max:.2e} A")

    # --------------------------------------------------------
    # Step 4: Fit LinearStepDevice
    # --------------------------------------------------------
    print("\n[4/5] Fitting LinearStepDevice...")

    fit_result, fitted_device, model_responses = fit_linearstep_device(
        pulse_data, response_data
    )

    # Get fitted parameters
    fitted_params = fit_result.params.valuesdict()
    print(f"\n  Fitted Parameters:")
    for param, value in fitted_params.items():
        print(f"    {param}: {value:.6f}")

    # Calculate metrics
    metrics = calculate_fit_metrics(response_data, model_responses)

    print(f"\n  Fitting Quality:")
    print(f"    R² (UP):      {metrics['r2_up']:.6f}")
    print(f"    R² (DOWN):    {metrics['r2_down']:.6f}")
    print(f"    R² (Overall): {metrics['r2_all']:.6f}")
    print(f"    RMSE (UP):    {metrics['rmse_up']:.6f}")
    print(f"    RMSE (DOWN):  {metrics['rmse_down']:.6f}")

    # --------------------------------------------------------
    # Step 5: Save and Visualize
    # --------------------------------------------------------
    print("\n[5/5] Saving results and creating visualizations...")

    # Save configuration
    config = {
        # Weight bounds
        'w_min': -1.0,
        'w_max': 1.0,

        # Fitted parameters
        'dw_min': fitted_device.dw_min,
        'gamma_up': fitted_device.gamma_up,
        'gamma_down': fitted_device.gamma_down,
        'write_noise_std': fitted_device.write_noise_std,

        # Other device settings
        'mult_noise': fitted_device.mult_noise,
        'mean_bound_reference': fitted_device.mean_bound_reference,
        'dw_min_std': fitted_device.dw_min_std,

        # Metadata
        '_metadata': {
            'device_type': 'LinearStepDevice',
            'source': 'Current.xlsx',
            'fitting_method': 'fit_measurements (aihwkit.utils.fitting)',
            'n_cycles_fitted': n_cycles,
            'metrics': metrics,
            'current_range': {
                'min': float(g_min),
                'max': float(g_max),
            },
        }
    }

    with open('Current_LinearStepDevice_fitted_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("  Saved: Current_LinearStepDevice_fitted_config.json")

    # Create visualizations
    print("\n  Creating visualizations...")
    create_fitting_plots(
        response_data, model_responses, fitted_device,
        current_data, peaks, valleys, g_min, g_max, metrics,
        'Current_LinearStep_measured'
    )

    # Plot device response using AIHWKit
    try:
        avg_pulses = np.mean([len(up_cycles[i]) for i in range(min(3, len(up_cycles)))])
        fig = plot_device_compact(fitted_device, w_noise=0.0,
                                   n_steps=int(avg_pulses),
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
    ║       Current.xlsx LinearStepDevice Final Parameters             ║
    ║               (fitted using fit_measurements)                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  FITTED PARAMETERS                                               ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    dw_min:          {fitted_device.dw_min:.6f}                               ║
    ║    gamma_up:        {fitted_device.gamma_up:+.6f}                               ║
    ║    gamma_down:      {fitted_device.gamma_down:+.6f}                               ║
    ║    write_noise_std: {fitted_device.write_noise_std:.6f}                               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  FITTING QUALITY                                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    R² (UP):        {metrics['r2_up']:.4f}                                    ║
    ║    R² (DOWN):      {metrics['r2_down']:.4f}                                    ║
    ║    R² (Overall):   {metrics['r2_all']:.4f}                                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  GENERATED FILES                                                 ║
    ║  ────────────────────────────────────────────────────────────────║
    ║    1. Current_LinearStepDevice_fitted_config.json                ║
    ║    2. Current_LinearStep_measured_fitting_summary.png            ║
    ║    3. Current_LinearStep_measured_all_cycles.png                 ║
    ║    4. Current_LinearStep_device_response.png                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
