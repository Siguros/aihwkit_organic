# -*- coding: utf-8 -*-

"""Verify 6T1C Retention Fitting Against Actual Measured Data.

This script compares:
1. Actual 6T1C retention data from 6T1C_result.xlsx
2. Fitted exponential model (τ = 775.1 min)
3. AIHWKit decay simulation with fitted lifetime parameter

Goal: Verify that AIHWKit correctly reproduces the physical 6T1C retention behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit

from preset_6T1C import SixT1CPresetDevice, get_lifetime_for_dt_batch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig


def load_retention_data():
    """Load actual 6T1C retention data from Excel."""
    df = pd.read_excel('6T1C_result.xlsx', sheet_name='Retention_결과')

    time_min = df['Time(min)'].values
    vcap = df['Vcap'].values

    print("Loaded 6T1C Retention Data:")
    print("-" * 40)
    for t, v in zip(time_min, vcap):
        print(f"  Time: {t:4.0f} min, Vcap: {v:.4f} V")
    print("-" * 40)

    return time_min, vcap


def fit_exponential_decay(time_min, vcap):
    """Fit exponential decay model to data."""

    # Model: V(t) = V0 * exp(-t/τ)
    # Since we know decay target is 0V, we use simple exponential
    def exp_decay(t, V0, tau):
        return V0 * np.exp(-t / tau)

    # Initial guess
    p0 = [vcap[0], 500]

    # Fit
    popt, pcov = curve_fit(exp_decay, time_min, vcap, p0=p0)
    V0_fit, tau_fit = popt

    # Calculate R²
    vcap_pred = exp_decay(time_min, V0_fit, tau_fit)
    ss_res = np.sum((vcap - vcap_pred) ** 2)
    ss_tot = np.sum((vcap - np.mean(vcap)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nExponential Fit Results:")
    print(f"  V0 = {V0_fit:.4f} V")
    print(f"  τ = {tau_fit:.1f} min ({tau_fit*60:.0f} sec)")
    print(f"  R² = {r_squared:.4f}")

    return V0_fit, tau_fit, r_squared


def simulate_aihwkit_decay(V0, tau_min, time_points_min):
    """Simulate decay using AIHWKit with fitted parameters."""

    # Convert τ from minutes to seconds
    tau_sec = tau_min * 60

    # For simulation, we'll use dt_batch = 1 minute (60 sec)
    # This means each decay_weights() call = 1 minute of physical time
    dt_batch_sec = 60  # 1 minute per step

    # Calculate lifetime for AIHWKit
    lifetime = get_lifetime_for_dt_batch(dt_batch_sec)

    print(f"\nAIHWKit Simulation Parameters:")
    print(f"  Physical τ: {tau_min:.1f} min ({tau_sec:.0f} sec)")
    print(f"  dt_batch: {dt_batch_sec} sec (1 min)")
    print(f"  AIHWKit lifetime: {lifetime:.0f}")

    # Create device with this lifetime
    device = SixT1CPresetDevice()
    device.lifetime = lifetime
    device.reset = 0.0  # Decay toward 0

    rpu_config = SingleRPUConfig(device=device)
    layer = AnalogLinear(5, 5, bias=False, rpu_config=rpu_config)
    tile = list(layer.analog_tiles())[0]

    # Set initial weight to V0 (in voltage domain, not normalized)
    # We'll track the raw value
    tile.set_weights(torch.full((5, 5), V0))

    # Simulate decay at each time point
    simulated_vcap = []
    current_time = 0

    for target_time in time_points_min:
        # Apply decay for the elapsed time (in 1-minute steps)
        steps_needed = int(target_time - current_time)
        for _ in range(steps_needed):
            tile.decay_weights(alpha=1.0)
        current_time = target_time

        # Record current weight
        current_v = tile.get_weights()[0].mean().item()
        simulated_vcap.append(current_v)

    return np.array(simulated_vcap), lifetime


def calculate_theoretical_decay(V0, tau_min, time_points_min):
    """Calculate theoretical exponential decay."""
    return V0 * np.exp(-time_points_min / tau_min)


def plot_comparison(time_min, vcap_actual, vcap_fit, vcap_aihwkit, tau_min, lifetime):
    """Create comparison plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('6T1C Retention: Actual Data vs Fitted Model vs AIHWKit',
                 fontsize=14, fontweight='bold')

    # Plot 1: Vcap comparison
    ax1 = axes[0, 0]
    ax1.plot(time_min, vcap_actual, 'bo', markersize=10, label='Actual 6T1C Data')
    ax1.plot(time_min, vcap_fit, 'r-', linewidth=2, label=f'Exponential Fit (τ={tau_min:.0f}min)')
    ax1.plot(time_min, vcap_aihwkit, 'g--', linewidth=2, label=f'AIHWKit (lifetime={lifetime:.0f})')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Decay target (0V)')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Vcap (V)')
    ax1.set_title('Capacitor Voltage Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals (Actual - Fit)
    ax2 = axes[0, 1]
    residual_fit = vcap_actual - vcap_fit
    residual_aihwkit = vcap_actual - vcap_aihwkit
    ax2.bar(np.array(time_min) - 10, residual_fit * 1000, width=18,
            label='Fit Residual', color='red', alpha=0.7)
    ax2.bar(np.array(time_min) + 10, residual_aihwkit * 1000, width=18,
            label='AIHWKit Residual', color='green', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Residual (mV)')
    ax2.set_title('Fitting Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Normalized comparison
    ax3 = axes[1, 0]
    v0 = vcap_actual[0]
    ax3.plot(time_min, vcap_actual / v0, 'bo', markersize=10, label='Actual (normalized)')
    ax3.plot(time_min, vcap_fit / v0, 'r-', linewidth=2, label='Exponential Fit')
    ax3.plot(time_min, vcap_aihwkit / v0, 'g--', linewidth=2, label='AIHWKit')
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Vcap / V0')
    ax3.set_title('Normalized Decay (V/V0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Log scale verification
    ax4 = axes[1, 1]
    # For exponential decay: ln(V) = ln(V0) - t/τ (linear in log scale)
    ax4.semilogy(time_min, vcap_actual, 'bo', markersize=10, label='Actual')
    ax4.semilogy(time_min, vcap_fit, 'r-', linewidth=2, label='Exponential Fit')
    ax4.semilogy(time_min, vcap_aihwkit, 'g--', linewidth=2, label='AIHWKit')
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Vcap (V) - Log Scale')
    ax4.set_title('Log Scale Verification\n(should be linear for pure exponential)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('6T1C_retention_fitting_verification.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: 6T1C_retention_fitting_verification.png")
    plt.close()


def print_comparison_table(time_min, vcap_actual, vcap_fit, vcap_aihwkit):
    """Print detailed comparison table."""

    print("\n" + "=" * 80)
    print("DETAILED COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Time(min)':>10} {'Actual(V)':>12} {'Fit(V)':>12} {'AIHWKit(V)':>12} "
          f"{'Fit Err(mV)':>12} {'AIHWKit Err(mV)':>16}")
    print("-" * 80)

    for t, va, vf, vs in zip(time_min, vcap_actual, vcap_fit, vcap_aihwkit):
        err_fit = (va - vf) * 1000
        err_aihwkit = (va - vs) * 1000
        print(f"{t:>10.0f} {va:>12.4f} {vf:>12.4f} {vs:>12.4f} "
              f"{err_fit:>12.2f} {err_aihwkit:>16.2f}")

    print("-" * 80)

    # Calculate RMSE
    rmse_fit = np.sqrt(np.mean((vcap_actual - vcap_fit) ** 2)) * 1000
    rmse_aihwkit = np.sqrt(np.mean((vcap_actual - vcap_aihwkit) ** 2)) * 1000

    print(f"{'RMSE':>10} {' ':>12} {' ':>12} {' ':>12} {rmse_fit:>12.2f} {rmse_aihwkit:>16.2f}")
    print("=" * 80)


def main():
    print("=" * 70)
    print("6T1C RETENTION FITTING VERIFICATION")
    print("=" * 70)
    print("\nComparing actual 6T1C retention data with:")
    print("  1. Exponential decay fit (τ = fitted value)")
    print("  2. AIHWKit decay simulation (using fitted lifetime)")

    # Load actual data
    time_min, vcap_actual = load_retention_data()

    # Fit exponential decay
    V0_fit, tau_fit, r_squared = fit_exponential_decay(time_min, vcap_actual)

    # Calculate theoretical fit
    vcap_fit = calculate_theoretical_decay(V0_fit, tau_fit, time_min)

    # Simulate with AIHWKit
    vcap_aihwkit, lifetime = simulate_aihwkit_decay(V0_fit, tau_fit, time_min)

    # Print comparison table
    print_comparison_table(time_min, vcap_actual, vcap_fit, vcap_aihwkit)

    # Calculate R² for AIHWKit
    ss_res = np.sum((vcap_actual - vcap_aihwkit) ** 2)
    ss_tot = np.sum((vcap_actual - np.mean(vcap_actual)) ** 2)
    r_squared_aihwkit = 1 - (ss_res / ss_tot)

    # Plot comparison
    plot_comparison(time_min, vcap_actual, vcap_fit, vcap_aihwkit, tau_fit, lifetime)

    # Summary
    print("\n" + "=" * 70)
    print("FITTING SUMMARY")
    print("=" * 70)
    print(f"""
    Actual 6T1C Data:
    ─────────────────────────────────────────
    - Time range: 0 ~ {time_min[-1]:.0f} min ({time_min[-1]/60:.1f} hours)
    - Vcap range: {vcap_actual[0]:.4f} V → {vcap_actual[-1]:.4f} V
    - Decay: {(1 - vcap_actual[-1]/vcap_actual[0])*100:.1f}%

    Exponential Fit:
    ─────────────────────────────────────────
    - V0 = {V0_fit:.4f} V
    - τ = {tau_fit:.1f} min ({tau_fit*60:.0f} sec)
    - R² = {r_squared:.4f}
    - RMSE = {np.sqrt(np.mean((vcap_actual - vcap_fit)**2))*1000:.2f} mV

    AIHWKit Simulation:
    ─────────────────────────────────────────
    - lifetime = {lifetime:.0f} (dt_batch = 60s)
    - R² = {r_squared_aihwkit:.4f}
    - RMSE = {np.sqrt(np.mean((vcap_actual - vcap_aihwkit)**2))*1000:.2f} mV

    Conclusion:
    ─────────────────────────────────────────
    {'✓ AIHWKit accurately reproduces 6T1C retention behavior!' if r_squared_aihwkit > 0.99 else '⚠ Some deviation observed'}
    {'✓ R² > 0.99 confirms excellent fit' if r_squared_aihwkit > 0.99 else ''}
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
