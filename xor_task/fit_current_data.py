#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fitting Current.xlsx data to AIHWKit PiecewiseStepDevice
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Imports from aihwkit
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.configs import PiecewiseStepDevice
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()

# CSV file path
FILE_NAME = "device_conductance_data.csv"

def read_from_file(filename, from_pulse_response=True, n_segments=10, skip_rows=0, use_first_cycle_only=True):
    """Read the update steps from file and convert to the required device input format.

    Here the CSV file has two columns, one for the up and the second
    for the down pulses. The conductance values should be consecutive.

    Args:
        filename: CSV file name to read from
        from_pulse_response: whether to load from pulse response
            data. Otherwise the up/down pulse directly given for each segment
        n_segments: the number of segments
        skip_rows: initial rows to skip (to skip column names)
        use_first_cycle_only: if True, only use the first half-cycle for fitting

    Returns:
        piecewise_up: scaled vector of up pulses in the range w_min to w_max
        piecewise_down: scaled vector of down pulses in the range w_min to w_max
        dw_min: minimal dw at zero
        up_down: bias at zero for up versus down direction
        noise_std: update noise estimate
        up_data: up data read from file
        down_data: down data read from file
    """
    from scipy.signal import find_peaks

    up_data = []
    down_data = []

    # Import CSV file to determine up_pulse, down_pulse, and n_points
    with open(filename, newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["up_data", "down_data"]
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue

            up_data.append(float(row["up_data"]))
            down_data.append(float(row["down_data"]))

    up_data = np.array(up_data)
    down_data = np.array(down_data)

    print(f"Loaded data: {len(up_data)} up pulses, {len(down_data)} down pulses")
    print(f"Up data range: {up_data.min():.2e} ~ {up_data.max():.2e}")
    print(f"Down data range: {down_data.min():.2e} ~ {down_data.max():.2e}")

    # Extract first half-cycle only if requested
    if use_first_cycle_only:
        # Find first peak for UP (starts low, goes high)
        peaks_up, _ = find_peaks(up_data, distance=100, prominence=(up_data.max()-up_data.min())*0.3)
        # Find first valley for DOWN (starts high, goes low)
        valleys_down, _ = find_peaks(-down_data, distance=100, prominence=(down_data.max()-down_data.min())*0.3)

        first_up_end = peaks_up[0] if len(peaks_up) > 0 else len(up_data) - 1
        first_down_end = valleys_down[0] if len(valleys_down) > 0 else len(down_data) - 1

        print(f"\n*** Using first cycle only ***")
        print(f"UP first half-cycle: 0 ~ {first_up_end} ({first_up_end + 1} pulses)")
        print(f"DOWN first half-cycle: 0 ~ {first_down_end} ({first_down_end + 1} pulses)")

        up_data = up_data[:first_up_end + 1]
        down_data = down_data[:first_down_end + 1]

    if from_pulse_response:
        g_max = min(up_data.max(), down_data.max())
        g_min = max(up_data.min(), down_data.min())

        print(f"g_min: {g_min:.2e}, g_max: {g_max:.2e}")

        # Normalize data to [-1, 1] first
        up_norm = (up_data - g_min) / (g_max - g_min) * 2 - 1
        down_norm = (down_data - g_min) / (g_max - g_min) * 2 - 1

        # ================================================================
        # CORRECTED: dw_min based on actual pulse count (not histogram)
        # ================================================================
        n_up_pulses = len(up_data)
        n_down_pulses = len(down_data)
        dw_min = 2.0 / ((n_up_pulses + n_down_pulses) / 2)

        print(f"\n[CORRECTED] dw_min calculation:")
        print(f"  UP pulses: {n_up_pulses}")
        print(f"  DOWN pulses: {n_down_pulses}")
        print(f"  dw_min = 2 / avg_pulses = {dw_min:.6f}")
        print(f"  Expected states: {2.0/dw_min:.0f}")

        # ================================================================
        # Piecewise values: step_size_per_segment / dw_min
        # ================================================================
        def get_piecewise_values(norm_data, dw_min_val, direction='up'):
            # Count pulses in each segment
            mean, edges = np.histogram(norm_data, bins=n_segments, range=[-1, 1])

            # Step size in each segment = segment_size / pulse_count
            segment_size = 2.0 / n_segments
            step_sizes = segment_size / mean  # actual step size per segment

            # Piecewise = step_size / dw_min (relative to dw_min)
            piecewise = step_sizes / dw_min_val

            return piecewise, mean

        up_piecewise, up_counts = get_piecewise_values(up_norm, dw_min, 'up')
        down_piecewise, down_counts = get_piecewise_values(down_norm, dw_min, 'down')

        print(f"\n[Histogram pulse distribution]")
        print(f"  UP counts per segment:   {up_counts.tolist()}")
        print(f"  DOWN counts per segment: {down_counts.tolist()}")

        # ================================================================
        # up_down: asymmetry between up and down at center
        # ================================================================
        center = n_segments // 2
        # up_down = (up_step - down_step) at center, normalized by dw_min
        up_down = (up_piecewise[center] - down_piecewise[center]) * dw_min

        # ================================================================
        # CORRECTED: noise_std as absolute std (not CV)
        # AIHWKit: w_apparent = w + write_noise_std * xi
        # ================================================================
        up_diff = np.diff(up_norm)
        down_diff = -np.diff(down_norm)

        # All steps (not just positive) - to capture true variability
        noise_abs_up = np.std(up_diff)
        noise_abs_down = np.std(down_diff)
        noise_abs_avg = (noise_abs_up + noise_abs_down) / 2

        # Relative to dw_min for proper scaling
        # But AIHWKit write_noise_std is multiplied by xi, so it's absolute
        # We set it relative to weight range for reasonable behavior
        noise_std = noise_abs_avg  # absolute std of step sizes

        # Also calculate CV for reference
        pos_up = up_diff[up_diff > 0]
        pos_down = down_diff[down_diff > 0]
        cv_up = np.std(pos_up) / np.mean(pos_up) if len(pos_up) > 0 else 0
        cv_down = np.std(pos_down) / np.mean(pos_down) if len(pos_down) > 0 else 0

        print(f"\n[CORRECTED] Noise estimation:")
        print(f"  Step std (UP):   {noise_abs_up:.6f}")
        print(f"  Step std (DOWN): {noise_abs_down:.6f}")
        print(f"  Absolute noise_std: {noise_std:.6f}")
        print(f"  (For reference) CV: UP={cv_up:.2%}, DOWN={cv_down:.2%}")

        # Update normalized data for return
        up_data = up_norm
        down_data = down_norm

        print(f"\nNormalized data range:")
        print(f"  UP:   {up_data.min():.3f} ~ {up_data.max():.3f}")
        print(f"  DOWN: {down_data.min():.3f} ~ {down_data.max():.3f}")
    else:
        # directly given
        up_piecewise = up_data
        down_piecewise = down_data
        dw_min = 0.002  # default
        up_down = 0.0
        noise_std = 0.0

    print(f"\n[Final Device Parameters]")
    print(f"  dw_min: {dw_min:.6f} ({2.0/dw_min:.0f} states)")
    print(f"  up_down: {up_down:.6f}")
    print(f"  write_noise_std: {noise_std:.6f}")
    print(f"  n_segments: {n_segments}")

    return (
        up_piecewise.tolist() if hasattr(up_piecewise, 'tolist') else up_piecewise,
        down_piecewise.tolist() if hasattr(down_piecewise, 'tolist') else down_piecewise,
        dw_min,
        up_down,
        noise_std,
        up_data,
        down_data,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Current.xlsx 데이터를 PiecewiseStepDevice로 Fitting")
    print("=" * 80)
    print()

    # Define the device response from values in the CSV file
    n_segments = 10
    print(f"Segments: {n_segments}\n")

    piecewise_up, piecewise_down, dw_min, up_down, noise_std, up_data, down_data = read_from_file(
        FILE_NAME, from_pulse_response=True, n_segments=n_segments
    )

    # Create PiecewiseStepDevice
    my_device = PiecewiseStepDevice(
        w_min=-1,
        w_max=1,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        dw_min_std=0.0,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        dw_min=dw_min,
        up_down=up_down,
        write_noise_std=noise_std,
        piecewise_up=piecewise_up,
        piecewise_down=piecewise_down,
    )

    print("\n" + "=" * 80)
    print("생성된 PiecewiseStepDevice 설정:")
    print("=" * 80)
    print(my_device)

    # Plot the pulse response
    print("\n시각화 중...")

    fig = plot_device_compact(
        my_device,
        w_noise=0.0,
        n_steps=len(up_data),
        use_cuda=USE_CUDA
    )

    plt.savefig('fitted_device_response.png', dpi=150, bbox_inches='tight')
    print("✓ Fitted device response를 'fitted_device_response.png'에 저장했습니다.")

    # Additional visualization: Compare original data with fitted model
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Original normalized data
    axes[0].plot(up_data, 'r-', alpha=0.7, linewidth=1, label='Up data (original)')
    axes[0].plot(down_data, 'b-', alpha=0.7, linewidth=1, label='Down data (original)')
    axes[0].set_xlabel('Pulse Number')
    axes[0].set_ylabel('Normalized Conductance')
    axes[0].set_title('Original Data (Normalized)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Piecewise pulse values
    axes[1].plot(piecewise_up, 'ro-', markersize=8, linewidth=2, label='Piecewise Up')
    axes[1].plot(piecewise_down, 'bo-', markersize=8, linewidth=2, label='Piecewise Down')
    axes[1].set_xlabel('Segment Index')
    axes[1].set_ylabel('Scaled Pulse Value')
    axes[1].set_title(f'Piecewise Step Model ({n_segments} segments)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_vs_model.png', dpi=150, bbox_inches='tight')
    print("✓ 데이터 비교 그래프를 'data_vs_model.png'에 저장했습니다.")

    print("\n" + "=" * 80)
    print("Fitting 완료!")
    print("=" * 80)
    print(f"\n생성된 파일:")
    print(f"1. fitted_device_response.png - AIHWKit device response 시각화")
    print(f"2. data_vs_model.png - 원본 데이터 vs 피팅 모델 비교")

    # Save device configuration
    import json
    device_config = {
        'w_min': -1,
        'w_max': 1,
        'dw_min': dw_min,
        'up_down': up_down,
        'write_noise_std': noise_std,
        'n_segments': n_segments,
        'piecewise_up': piecewise_up,
        'piecewise_down': piecewise_down,
    }

    with open('device_config.json', 'w') as f:
        json.dump(device_config, f, indent=2)
    print(f"3. device_config.json - 디바이스 설정 파일")

    # ========================================================================
    # NEW: Compare model simulation with actual data overlay
    # ========================================================================
    print("\n" + "=" * 80)
    print("Model Simulation vs Actual Data Comparison")
    print("=" * 80)

    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.tiles import AnalogTile
    import torch

    # Simulate pulse response using piecewise model directly
    def simulate_piecewise_response(piecewise_values, dw_min_val, n_steps, direction='up'):
        """Simulate pulse response using piecewise step model"""
        w_min, w_max = -1.0, 1.0
        n_segments = len(piecewise_values)

        if direction == 'up':
            w = w_min
            weights = [w]
            for _ in range(n_steps - 1):
                # Determine segment index
                seg_idx = int((w - w_min) / (w_max - w_min) * n_segments)
                seg_idx = max(0, min(n_segments - 1, seg_idx))
                # Calculate step size
                dw = dw_min_val * piecewise_values[seg_idx]
                w = min(w_max, w + dw)
                weights.append(w)
        else:  # down
            w = w_max
            weights = [w]
            for _ in range(n_steps - 1):
                seg_idx = int((w - w_min) / (w_max - w_min) * n_segments)
                seg_idx = max(0, min(n_segments - 1, seg_idx))
                dw = dw_min_val * piecewise_values[seg_idx]
                w = max(w_min, w - dw)
                weights.append(w)

        return np.array(weights)

    # up_data and down_data are already the first cycle (extracted in read_from_file)
    # Use the entire data directly
    up_cycle_actual = up_data
    down_cycle_actual = down_data

    n_up_pulses = len(up_cycle_actual)
    n_down_pulses = len(down_cycle_actual)

    print(f"UP cycle: {n_up_pulses} pulses")
    print(f"DOWN cycle: {n_down_pulses} pulses")

    # Calculate dw_min based on actual cycle length
    # Total range is 2 (-1 to 1), so dw_min = 2 / n_pulses (average)
    avg_up_step = 2.0 / n_up_pulses
    avg_down_step = 2.0 / n_down_pulses

    print(f"Estimated avg UP step: {avg_up_step:.6f}")
    print(f"Estimated avg DOWN step: {avg_down_step:.6f}")

    # Simulate using piecewise model
    simulated_up = simulate_piecewise_response(piecewise_up, avg_up_step, n_up_pulses, 'up')
    simulated_down = simulate_piecewise_response(piecewise_down, avg_down_step, n_down_pulses, 'down')

    # Normalize actual data to [-1, 1] range
    # UP: starts from min (-1) and goes to max (1)
    up_cycle_norm = (up_cycle_actual - up_cycle_actual.min()) / (up_cycle_actual.max() - up_cycle_actual.min()) * 2 - 1
    # DOWN: starts from max (1) and goes to min (-1)
    # Same normalization: high value -> 1, low value -> -1
    down_cycle_norm = (down_cycle_actual - down_cycle_actual.min()) / (down_cycle_actual.max() - down_cycle_actual.min()) * 2 - 1

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: UP pulse comparison (single half-cycle)
    ax1 = axes[0, 0]
    ax1.plot(up_cycle_norm, 'r-', alpha=0.7, linewidth=2, label='Actual Data (Up)')
    ax1.plot(simulated_up, 'k--', linewidth=2, label='Model Simulation (Up)')
    ax1.set_xlabel('Pulse Number')
    ax1.set_ylabel('Normalized Conductance')
    ax1.set_title(f'UP Pulse (Single Half-Cycle, {n_up_pulses} pulses): Actual vs Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: DOWN pulse comparison (single half-cycle)
    ax2 = axes[0, 1]
    ax2.plot(down_cycle_norm, 'b-', alpha=0.7, linewidth=2, label='Actual Data (Down)')
    ax2.plot(simulated_down, 'k--', linewidth=2, label='Model Simulation (Down)')
    ax2.set_xlabel('Pulse Number')
    ax2.set_ylabel('Normalized Conductance')
    ax2.set_title(f'DOWN Pulse (Single Half-Cycle, {n_down_pulses} pulses): Actual vs Model')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Full first cycle data
    ax3 = axes[1, 0]
    ax3.plot(up_data, 'r-', alpha=0.7, linewidth=1, label=f'UP ({n_up_pulses} pulses)')
    ax3.plot(down_data, 'b-', alpha=0.7, linewidth=1, label=f'DOWN ({n_down_pulses} pulses)')
    ax3.set_xlabel('Pulse Number')
    ax3.set_ylabel('Normalized Conductance')
    ax3.set_title('First Cycle Data (Used for Fitting)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residual (error) plot
    ax4 = axes[1, 1]
    residual_up = up_cycle_norm - simulated_up
    residual_down = down_cycle_norm - simulated_down
    ax4.plot(residual_up, 'r-', alpha=0.7, linewidth=1, label=f'Up Error (std={np.std(residual_up):.4f})')
    ax4.plot(residual_down, 'b-', alpha=0.7, linewidth=1, label=f'Down Error (std={np.std(residual_down):.4f})')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Pulse Number')
    ax4.set_ylabel('Residual (Actual - Model)')
    ax4.set_title('Error Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Calculate fit metrics
    rmse_up = np.sqrt(np.mean(residual_up**2))
    rmse_down = np.sqrt(np.mean(residual_down**2))
    r2_up = 1 - np.sum(residual_up**2) / np.sum((up_cycle_norm - np.mean(up_cycle_norm))**2)
    r2_down = 1 - np.sum(residual_down**2) / np.sum((down_cycle_norm - np.mean(down_cycle_norm))**2)

    fig.suptitle(f'PiecewiseStepDevice Fitting Result\n'
                 f'UP - RMSE: {rmse_up:.4f}, R2: {r2_up:.4f} | '
                 f'DOWN - RMSE: {rmse_down:.4f}, R2: {r2_down:.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_vs_data_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved model vs data comparison to 'model_vs_data_comparison.png'")

    print(f"\nFitting Quality Metrics:")
    print(f"  UP   - RMSE: {rmse_up:.6f}, R2: {r2_up:.6f}")
    print(f"  DOWN - RMSE: {rmse_down:.6f}, R2: {r2_down:.6f}")
    print(f"\n4. model_vs_data_comparison.png - Model vs Actual Data Comparison")

    # ========================================================================
    # OPTIMIZATION: Improve fitting by optimizing piecewise parameters
    # ========================================================================
    print("\n" + "=" * 80)
    print("Parameter Optimization")
    print("=" * 80)

    from scipy.optimize import minimize, differential_evolution

    def objective_function(params, actual_up, actual_down, n_segments):
        """Objective function to minimize (negative R² sum)"""
        # Split params into piecewise_up and piecewise_down
        pw_up = params[:n_segments]
        pw_down = params[n_segments:]

        n_up = len(actual_up)
        n_down = len(actual_down)
        dw_up = 2.0 / n_up
        dw_down = 2.0 / n_down

        # Simulate UP
        sim_up = simulate_piecewise_response(pw_up, dw_up, n_up, 'up')
        # Simulate DOWN
        sim_down = simulate_piecewise_response(pw_down, dw_down, n_down, 'down')

        # Calculate R² for both
        res_up = actual_up - sim_up
        res_down = actual_down - sim_down

        ss_res_up = np.sum(res_up**2)
        ss_tot_up = np.sum((actual_up - np.mean(actual_up))**2)
        r2_up = 1 - ss_res_up / ss_tot_up if ss_tot_up > 0 else 0

        ss_res_down = np.sum(res_down**2)
        ss_tot_down = np.sum((actual_down - np.mean(actual_down))**2)
        r2_down = 1 - ss_res_down / ss_tot_down if ss_tot_down > 0 else 0

        # Return negative of combined R² (we want to maximize R²)
        return -(r2_up + r2_down)

    # Initial parameters (current piecewise values)
    initial_params = np.array(piecewise_up + piecewise_down)

    print(f"Initial R² (UP): {r2_up:.4f}")
    print(f"Initial R² (DOWN): {r2_down:.4f}")
    print(f"Initial Combined R²: {r2_up + r2_down:.4f}")

    # Bounds for parameters (positive values, reasonable range)
    bounds = [(0.1, 3.0)] * (2 * n_segments)

    print("\nOptimizing parameters...")

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective_function,
        bounds,
        args=(up_cycle_norm, down_cycle_norm, n_segments),
        seed=42,
        maxiter=500,
        tol=1e-6,
        disp=False,
        workers=1
    )

    optimized_params = result.x
    optimized_pw_up = optimized_params[:n_segments].tolist()
    optimized_pw_down = optimized_params[n_segments:].tolist()

    # Calculate optimized metrics
    sim_up_opt = simulate_piecewise_response(optimized_pw_up, avg_up_step, n_up_pulses, 'up')
    sim_down_opt = simulate_piecewise_response(optimized_pw_down, avg_down_step, n_down_pulses, 'down')

    res_up_opt = up_cycle_norm - sim_up_opt
    res_down_opt = down_cycle_norm - sim_down_opt

    rmse_up_opt = np.sqrt(np.mean(res_up_opt**2))
    rmse_down_opt = np.sqrt(np.mean(res_down_opt**2))
    r2_up_opt = 1 - np.sum(res_up_opt**2) / np.sum((up_cycle_norm - np.mean(up_cycle_norm))**2)
    r2_down_opt = 1 - np.sum(res_down_opt**2) / np.sum((down_cycle_norm - np.mean(down_cycle_norm))**2)

    print(f"\nOptimized Results:")
    print(f"  UP   - RMSE: {rmse_up_opt:.6f}, R2: {r2_up_opt:.6f} (was {r2_up:.4f})")
    print(f"  DOWN - RMSE: {rmse_down_opt:.6f}, R2: {r2_down_opt:.6f} (was {r2_down:.4f})")
    print(f"  Combined R²: {r2_up_opt + r2_down_opt:.4f} (was {r2_up + r2_down:.4f})")

    print(f"\nOptimized piecewise_up: {[f'{v:.4f}' for v in optimized_pw_up]}")
    print(f"Optimized piecewise_down: {[f'{v:.4f}' for v in optimized_pw_down]}")

    # Create clean optimized comparison plot (Optimized Model vs Actual Data only)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: UP Pulse - Optimized Model vs Actual Data
    ax1 = axes[0, 0]
    ax1.plot(up_cycle_norm, 'r-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax1.plot(sim_up_opt, 'k--', linewidth=2, label=f'Optimized Model (R²={r2_up_opt:.4f})')
    ax1.set_xlabel('Pulse Number')
    ax1.set_ylabel('Normalized Conductance')
    ax1.set_title(f'UP Pulse: Optimized Model vs Actual Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: DOWN Pulse - Optimized Model vs Actual Data
    ax2 = axes[0, 1]
    ax2.plot(down_cycle_norm, 'b-', alpha=0.8, linewidth=1.5, label='Actual Data')
    ax2.plot(sim_down_opt, 'k--', linewidth=2, label=f'Optimized Model (R²={r2_down_opt:.4f})')
    ax2.set_xlabel('Pulse Number')
    ax2.set_ylabel('Normalized Conductance')
    ax2.set_title(f'DOWN Pulse: Optimized Model vs Actual Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Optimized Piecewise Parameters
    ax3 = axes[1, 0]
    x_seg = np.arange(n_segments)
    width = 0.35
    ax3.bar(x_seg - width/2, optimized_pw_up, width, label='Optimized UP', alpha=0.8, color='red')
    ax3.bar(x_seg + width/2, optimized_pw_down, width, label='Optimized DOWN', alpha=0.8, color='blue')
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Piecewise Value')
    ax3.set_title('Optimized Piecewise Parameters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_seg)

    # Plot 4: Residual Error
    ax4 = axes[1, 1]
    ax4.plot(res_up_opt, 'r-', alpha=0.7, linewidth=1, label=f'UP Error (RMSE={rmse_up_opt:.4f})')
    ax4.plot(res_down_opt, 'b-', alpha=0.7, linewidth=1, label=f'DOWN Error (RMSE={rmse_down_opt:.4f})')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Pulse Number')
    ax4.set_ylabel('Residual (Actual - Model)')
    ax4.set_title('Residual Error Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Optimized PiecewiseStepDevice Fitting Results\n'
                 f'UP: R²={r2_up_opt:.4f}, RMSE={rmse_up_opt:.4f} | '
                 f'DOWN: R²={r2_down_opt:.4f}, RMSE={rmse_down_opt:.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('optimized_fitting.png', dpi=150, bbox_inches='tight')
    print("\nSaved optimized comparison to 'optimized_fitting.png'")

    # Save optimized device configuration
    optimized_device_config = {
        'w_min': -1,
        'w_max': 1,
        'dw_min': dw_min,
        'up_down': up_down,
        'write_noise_std': noise_std,
        'n_segments': n_segments,
        'piecewise_up': optimized_pw_up,
        'piecewise_down': optimized_pw_down,
        'metrics': {
            'r2_up': r2_up_opt,
            'r2_down': r2_down_opt,
            'rmse_up': rmse_up_opt,
            'rmse_down': rmse_down_opt
        }
    }

    with open('optimized_device_config.json', 'w') as f:
        json.dump(optimized_device_config, f, indent=2)
    print("Saved optimized config to 'optimized_device_config.json'")

    # Create optimized PiecewiseStepDevice
    optimized_device = PiecewiseStepDevice(
        w_min=-1,
        w_max=1,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        dw_min_std=0.0,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        dw_min=dw_min,
        up_down=up_down,
        write_noise_std=noise_std,
        piecewise_up=optimized_pw_up,
        piecewise_down=optimized_pw_down,
    )

    print("\n" + "=" * 80)
    print("Optimized PiecewiseStepDevice:")
    print("=" * 80)
    print(optimized_device)
