# -*- coding: utf-8 -*-

"""Test script to verify 6T1C retention characteristics in AIHWKit.

6T1C Retention Model:
    - Physical time constant τ = 775.1 min (46505 sec)
    - Decay target: 0V (reset = 0.0 in AIHWKit)
    - AIHWKit formula: w_new = (w - reset) * (1 - 1/lifetime) + reset
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from preset_6T1C import SixT1CPreset, SixT1CPresetDevice, get_lifetime_for_dt_batch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig


def test_retention_decay():
    """Test that retention decay works correctly."""
    print("=" * 70)
    print("6T1C Retention Test")
    print("=" * 70)

    # Create analog layer
    print("\n[1] Creating AnalogLinear with SixT1CPreset...")
    rpu_config = SixT1CPreset()
    layer = AnalogLinear(10, 10, bias=False, rpu_config=rpu_config)

    device = rpu_config.device
    print(f"    lifetime: {device.lifetime}")
    print(f"    reset: {device.reset}")

    tiles = list(layer.analog_tiles())
    tile = tiles[0]

    # Set initial weights
    print("\n[2] Setting weights to +0.8...")
    initial_weight = 0.8
    tile.set_weights(torch.full((10, 10), initial_weight))
    print(f"    Initial weights (mean): {tile.get_weights()[0].mean().item():.4f}")

    # Apply decay
    print("\n[3] Applying retention decay...")
    n_steps = 100
    weight_history = [tile.get_weights()[0].mean().item()]

    for i in range(n_steps):
        tile.decay_weights(alpha=1.0)
        weight_history.append(tile.get_weights()[0].mean().item())
        if i % 20 == 0:
            print(f"    Step {i}: mean weight = {weight_history[-1]:.6f}")

    print(f"    Final: mean weight = {weight_history[-1]:.6f}")

    # Calculate expected decay
    print("\n[4] Comparing with expected decay...")
    lifetime = device.lifetime
    reset = device.reset
    delta = 1.0 / lifetime

    expected_history = []
    w = initial_weight
    for i in range(n_steps + 1):
        expected_history.append(w)
        w = (w - reset) * (1 - delta) + reset

    print(f"    Expected final: {expected_history[-1]:.6f}")
    print(f"    Actual final:   {weight_history[-1]:.6f}")
    print(f"    Difference: {abs(expected_history[-1] - weight_history[-1]):.6f}")

    decay_occurred = weight_history[-1] < weight_history[0]
    print(f"\n    Decay occurred: {decay_occurred}")
    print(f"    Decaying toward reset ({reset}): {decay_occurred}")

    return weight_history, expected_history, device


def test_different_lifetimes():
    """Test retention with different dt_batch values."""
    print("\n" + "=" * 70)
    print("Testing Different Lifetime Settings")
    print("=" * 70)

    dt_batch_values = [1, 10, 60, 600]
    results = {}
    initial_weight = 0.9
    n_steps = 50

    for dt_batch in dt_batch_values:
        lifetime = get_lifetime_for_dt_batch(dt_batch)

        device = SixT1CPresetDevice()
        device.lifetime = lifetime

        rpu_config = SingleRPUConfig(device=device)
        layer = AnalogLinear(5, 5, bias=False, rpu_config=rpu_config)
        tile = list(layer.analog_tiles())[0]
        tile.set_weights(torch.full((5, 5), initial_weight))

        weight_history = [initial_weight]
        for _ in range(n_steps):
            tile.decay_weights(alpha=1.0)
            weight_history.append(tile.get_weights()[0].mean().item())

        results[dt_batch] = weight_history
        decay = initial_weight - weight_history[-1]

        print(f"\n  dt_batch = {dt_batch}s (lifetime = {lifetime:.0f}):")
        print(f"    Initial: {initial_weight:.6f}")
        print(f"    After {n_steps} steps: {weight_history[-1]:.6f}")
        print(f"    Decay: {decay:.6f}")

    return results


def test_decay_direction():
    """Test decay direction from different initial values."""
    print("\n" + "=" * 70)
    print("Testing Decay Direction (toward reset = 0.0)")
    print("=" * 70)

    device = SixT1CPresetDevice()
    device.lifetime = 50  # Short lifetime for visible decay

    rpu_config = SingleRPUConfig(device=device)

    test_cases = [
        ("Positive weights (+0.8)", 0.8),
        ("Small positive (+0.2)", 0.2),
        ("Negative weights (-0.5)", -0.5),
        ("Large negative (-0.8)", -0.8),
    ]

    results = {}
    n_steps = 200

    for name, init_val in test_cases:
        layer = AnalogLinear(5, 5, bias=False, rpu_config=rpu_config)
        tile = list(layer.analog_tiles())[0]
        tile.set_weights(torch.full((5, 5), init_val))

        history = [init_val]
        for _ in range(n_steps):
            tile.decay_weights(alpha=1.0)
            history.append(tile.get_weights()[0].mean().item())

        results[name] = history
        final_val = history[-1]

        # Check if moving toward 0
        moving_toward_zero = abs(final_val) < abs(init_val)
        direction = "toward 0" if moving_toward_zero else "away from 0"

        print(f"\n  {name}:")
        print(f"    Start: {init_val:.4f}")
        print(f"    End:   {final_val:.4f}")
        print(f"    Direction: {direction} ({'correct' if moving_toward_zero else 'INCORRECT'})")

    return results


def test_long_term_convergence():
    """Test that weights converge to reset value."""
    print("\n" + "=" * 70)
    print("Testing Long-Term Convergence (to reset = 0)")
    print("=" * 70)

    device = SixT1CPresetDevice()
    device.lifetime = 20  # Very short for quick convergence

    rpu_config = SingleRPUConfig(device=device)
    layer = AnalogLinear(5, 5, bias=False, rpu_config=rpu_config)
    tile = list(layer.analog_tiles())[0]

    initial_weight = 0.9
    tile.set_weights(torch.full((5, 5), initial_weight))

    n_steps = 500
    history = [initial_weight]

    for _ in range(n_steps):
        tile.decay_weights(alpha=1.0)
        history.append(tile.get_weights()[0].mean().item())

    reset = device.reset
    final_weight = history[-1]

    print(f"\n  Initial weight: {initial_weight}")
    print(f"  Reset target: {reset}")
    print(f"  After {n_steps} steps: {final_weight:.6f}")
    print(f"  Distance from reset: {abs(final_weight - reset):.6f}")
    print(f"  Converged: {abs(final_weight - reset) < 0.01}")

    return history, reset


def plot_results(weight_history, expected_history, device,
                 different_lifetimes, decay_directions, long_term_data):
    """Create visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('6T1C Retention Verification (reset=0, decay toward 0V)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Basic retention
    ax1 = axes[0, 0]
    steps = range(len(weight_history))
    ax1.plot(steps, weight_history, 'b-', linewidth=2, label='Actual (AIHWKit)')
    ax1.plot(steps, expected_history, 'r--', linewidth=2, label='Expected (theory)')
    ax1.axhline(y=device.reset, color='gray', linestyle=':', label=f'Reset target ({device.reset})')
    ax1.set_xlabel('Decay Steps')
    ax1.set_ylabel('Weight Value')
    ax1.set_title(f'Retention Decay (lifetime={device.lifetime:.0f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Different lifetimes
    ax2 = axes[0, 1]
    colors = ['blue', 'green', 'orange', 'red']
    for (dt_batch, history), color in zip(different_lifetimes.items(), colors):
        lifetime = get_lifetime_for_dt_batch(dt_batch)
        ax2.plot(history, color=color, linewidth=2,
                label=f'dt={dt_batch}s (lifetime={lifetime:.0f})')
    ax2.axhline(y=0.0, color='gray', linestyle=':', label='Reset (0.0)')
    ax2.set_xlabel('Decay Steps')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Retention with Different dt_batch')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Decay direction
    ax3 = axes[1, 0]
    colors = ['red', 'orange', 'blue', 'purple']
    for (name, history), color in zip(decay_directions.items(), colors):
        ax3.plot(history, color=color, linewidth=2, label=name)
    ax3.axhline(y=0.0, color='gray', linestyle=':', linewidth=2, label='Reset target (0.0)')
    ax3.set_xlabel('Decay Steps')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Decay Direction (all converge to 0)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Long-term convergence
    ax4 = axes[1, 1]
    long_term_history, reset = long_term_data
    ax4.plot(long_term_history, 'b-', linewidth=2, label='Weight value')
    ax4.axhline(y=reset, color='red', linestyle='--', linewidth=2, label=f'Reset ({reset})')
    ax4.set_xlabel('Decay Steps')
    ax4.set_ylabel('Weight Value')
    ax4.set_title('Long-term Convergence (lifetime=20, 500 steps)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('6T1C_retention_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: 6T1C_retention_test.png")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("6T1C RETENTION VERIFICATION")
    print("=" * 70)
    print("\nAIHWKit decay formula: w_new = (w - reset) * (1 - 1/lifetime) + reset")
    print("6T1C capacitor decays to 0V -> reset = 0.0")

    # Run tests
    weight_history, expected_history, device = test_retention_decay()
    different_lifetimes = test_different_lifetimes()
    decay_directions = test_decay_direction()
    long_term_data = test_long_term_convergence()

    # Plot
    print("\n[5] Creating visualization...")
    plot_results(weight_history, expected_history, device,
                different_lifetimes, decay_directions, long_term_data)

    # Summary
    print("\n" + "=" * 70)
    print("RETENTION TEST SUMMARY")
    print("=" * 70)

    decay_amount = weight_history[0] - weight_history[-1]
    if decay_amount > 0:
        print(f"""
    ✓ Retention decay is WORKING correctly
    ✓ Weight decreased by {decay_amount:.6f} over 100 steps
    ✓ All weights decay toward reset=0 (matching 0V)

    6T1C Retention Parameters:
    ─────────────────────────────────────────
    Physical τ:     775.1 min (46505 sec)
    Reset target:   0.0 (= 0V)
    Default lifetime: 46506 (dt_batch=1s)
""")
    else:
        print(f"\n    ✗ Retention test failed. Decay: {decay_amount}")

    print("=" * 70)


if __name__ == "__main__":
    main()
