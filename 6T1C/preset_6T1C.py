# -*- coding: utf-8 -*-

"""6T1C Device Preset Configuration for AIHWKit.

This module provides preset device configurations for 6T1C (6 Transistors, 1 Capacitor)
memory device based on experimental measurements from 6T1C_result.xlsx.

The device parameters were fitted using LinearStepDevice model with:
- Update characteristics: R² = 0.999 (UP/DOWN)
- Retention characteristics: R² = 0.994

Usage:
    from preset_6T1C import SixT1CPresetDevice, SixT1CPreset

    # Single device configuration
    model = AnalogLinear(784, 128, rpu_config=SixT1CPreset())

    # Or use device directly
    from aihwkit.simulator.configs import SingleRPUConfig
    config = SingleRPUConfig(device=SixT1CPresetDevice())
"""

from dataclasses import dataclass, field

from aihwkit.simulator.configs.devices import LinearStepDevice, PulsedDevice
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import VectorUnitCell, TransferCompound
from aihwkit.simulator.parameters.enums import (
    VectorUnitCellUpdatePolicy,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


# =============================================================================
# 6T1C Device Preset (based on LinearStepDevice)
# =============================================================================

@dataclass
class SixT1CPresetDevice(LinearStepDevice):
    """Preset configuration for 6T1C (6 Transistors, 1 Capacitor) memory device.

    Fit of the model :class:`LinearStepDevice` to experimental data from
    6T1C capacitor-based synaptic device measurements.

    Device Characteristics:
        - ~1000 conductance states per direction
        - Capacitor-based weight storage with exponential decay
        - Time constant τ ≈ 775 min (12.9 hours)
        - Decay target: 0V (normalized to w = -1)

    Update Model:
        LinearStepDevice formula: w_new = w + dw_min * (1 + gamma * w) * (1 + noise)

        - gamma_up = -0.168 (slight saturation at high weights)
        - gamma_down = +0.141 (near-linear behavior)

    Retention Model:
        AIHWKit decay formula: w_new = (w - b) * (1 - 1/lifetime) + b

        - lifetime = 46506 (at dt_batch = 1 sec)
        - reset (b) = 0.0 (decays toward 0V)

    Caution:
        The ``lifetime`` parameter depends on the assumed time per mini-batch
        (dt_batch). The default assumes dt_batch = 1 second. Adjust according to
        your training setup:

        - dt_batch = 1 sec  -> lifetime = 46506
        - dt_batch = 10 sec -> lifetime = 4651
        - dt_batch = 1 min  -> lifetime = 776
        - dt_batch = 10 min -> lifetime = 78

        To calculate lifetime for different dt_batch:
            lifetime = 1 / (1 - exp(-dt_batch / τ))
            where τ = 46505 seconds (775.1 minutes)

    Note:
        Parameters were fitted from 6T1C_result.xlsx experimental data with:
        - Update fitting: R² > 0.999
        - Retention fitting: R² = 0.9935
    """

    # =========================================================================
    # Core update parameters (fitted from 6T1C conductance cycling data)
    # =========================================================================

    dw_min: float = 0.001981  # Minimum weight update step size
    up_down: float = 0.0  # Symmetry point (calibrated to zero)

    w_max: float = 1.0  # Maximum weight (normalized)
    w_min: float = -1.0  # Minimum weight (normalized)

    # LinearStepDevice nonlinearity parameters
    # gamma < 0: step size decreases as weight approaches bounds (saturation)
    # gamma > 0: step size increases as weight approaches bounds
    gamma_up: float = -0.1678  # Fitted: slight saturation effect
    gamma_down: float = 0.1410  # Fitted: near-linear behavior

    mult_noise: bool = True  # Use multiplicative noise model

    # =========================================================================
    # Device-to-device variation (D2D)
    # =========================================================================

    dw_min_dtod: float = 0.1  # D2D variation of dw_min
    up_down_dtod: float = 0.01  # D2D variation of asymmetry

    w_max_dtod: float = 0.05  # D2D variation of w_max
    w_min_dtod: float = 0.05  # D2D variation of w_min

    gamma_up_dtod: float = 0.05  # D2D variation of gamma_up
    gamma_down_dtod: float = 0.05  # D2D variation of gamma_down

    # =========================================================================
    # Cycle-to-cycle variation (C2C)
    # =========================================================================

    dw_min_std: float = 0.3  # C2C variation of update step
    write_noise_std: float = 0.0182  # Fitted write noise

    # =========================================================================
    # LinearStepDevice specific settings
    # =========================================================================

    # Slope does not depend on the actual device-specific bound
    mean_bound_reference: bool = True

    # =========================================================================
    # Retention / Capacitor leakage (fitted from 6T1C retention data)
    # =========================================================================

    # Lifetime in mini-batches (assuming dt_batch = 1 sec)
    # Physical time constant τ = 775.1 min = 46505 sec
    # lifetime = 1 / (1 - exp(-dt_batch/τ)) ≈ 46506 for dt_batch = 1 sec
    lifetime: float = 46506.0
    lifetime_dtod: float = 0.1  # D2D variation of lifetime

    # Decay target (reset value)
    # 6T1C capacitor decays to 0V
    # In AIHWKit, reset=0 means weights decay toward 0 (matching 0V)
    # Note: This is different from our normalized w = 2*Vcap - 1 mapping
    # AIHWKit uses a simpler mapping where w=0 corresponds to Vcap=0V
    reset: float = 0.0
    reset_dtod: float = 0.0  # Assuming uniform decay target


@dataclass
class SixT1CPresetDeviceNoRetention(LinearStepDevice):
    """6T1C device preset without retention effects.

    Use this for training scenarios where retention is not considered
    or when modeling ideal behavior without capacitor leakage.
    """

    dw_min: float = 0.001981
    up_down: float = 0.0

    w_max: float = 1.0
    w_min: float = -1.0

    gamma_up: float = -0.1678
    gamma_down: float = 0.1410

    mult_noise: bool = True

    dw_min_dtod: float = 0.1
    up_down_dtod: float = 0.01
    w_max_dtod: float = 0.05
    w_min_dtod: float = 0.05
    gamma_up_dtod: float = 0.05
    gamma_down_dtod: float = 0.05

    dw_min_std: float = 0.3
    write_noise_std: float = 0.0182

    mean_bound_reference: bool = True

    # No retention (infinite lifetime)
    lifetime: float = 0.0  # 0 means no decay
    reset: float = 0.0


# =============================================================================
# Single Device RPU Configurations
# =============================================================================

@dataclass
class SixT1CPreset(SingleRPUConfig):
    """Preset configuration using a single 6T1C device.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses, including capacitor retention effects.

    Example:
        >>> from aihwkit.nn import AnalogLinear
        >>> from preset_6T1C import SixT1CPreset
        >>>
        >>> # Create analog layer with 6T1C device
        >>> layer = AnalogLinear(784, 128, rpu_config=SixT1CPreset())
    """

    device: PulsedDevice = field(default_factory=SixT1CPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class SixT1CPresetNoRetention(SingleRPUConfig):
    """Preset configuration using 6T1C device without retention.

    Use this for ideal training without capacitor leakage effects.
    """

    device: PulsedDevice = field(default_factory=SixT1CPresetDeviceNoRetention)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# =============================================================================
# Multi-Device (Vector Unit Cell) Configurations
# =============================================================================

@dataclass
class SixT1C2Preset(UnitCellRPUConfig):
    """Preset configuration using two 6T1C devices per cross-point.

    Both devices are updated with random selection policy.
    This can improve weight precision and reduce noise effects.

    Example:
        >>> from aihwkit.nn import AnalogLinear
        >>> from preset_6T1C import SixT1C2Preset
        >>>
        >>> layer = AnalogLinear(784, 128, rpu_config=SixT1C2Preset())
    """

    device: VectorUnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[SixT1CPresetDevice(), SixT1CPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class SixT1C4Preset(UnitCellRPUConfig):
    """Preset configuration using four 6T1C devices per cross-point.

    Four devices per cross-point for higher weight precision.
    """

    device: VectorUnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                SixT1CPresetDevice(),
                SixT1CPresetDevice(),
                SixT1CPresetDevice(),
                SixT1CPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# =============================================================================
# Tiki-Taka Configuration
# =============================================================================

@dataclass
class TikiTakaSixT1CPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka optimizer with 6T1C devices.

    Tiki-taka is a hardware-aware optimizer that uses two crossbar arrays
    for improved gradient accumulation and weight updates.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    Example:
        >>> from aihwkit.nn import AnalogLinear
        >>> from preset_6T1C import TikiTakaSixT1CPreset
        >>>
        >>> layer = AnalogLinear(784, 128, rpu_config=TikiTakaSixT1CPreset())
    """

    device: TransferCompound = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[SixT1CPresetDevice(), SixT1CPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# =============================================================================
# Utility Functions
# =============================================================================

def get_lifetime_for_dt_batch(dt_batch_sec: float) -> float:
    """Calculate AIHWKit lifetime parameter for a given dt_batch.

    The 6T1C capacitor has a physical time constant τ = 46505 seconds.
    The AIHWKit lifetime parameter depends on dt_batch assumption.

    Args:
        dt_batch_sec: Assumed time per mini-batch in seconds.

    Returns:
        Lifetime parameter for AIHWKit configuration.

    Example:
        >>> # For 1 second per batch
        >>> lifetime = get_lifetime_for_dt_batch(1.0)
        >>> print(f"lifetime = {lifetime:.0f}")
        lifetime = 46506

        >>> # For 1 minute per batch
        >>> lifetime = get_lifetime_for_dt_batch(60.0)
        >>> print(f"lifetime = {lifetime:.0f}")
        lifetime = 776
    """
    import math
    TAU_SEC = 46505.0  # Physical time constant in seconds
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    return 1.0 / delta


def create_6t1c_device(
    dt_batch_sec: float = 1.0,
    include_retention: bool = True,
    include_noise: bool = True,
) -> LinearStepDevice:
    """Create a customized 6T1C device configuration.

    Args:
        dt_batch_sec: Assumed time per mini-batch in seconds.
        include_retention: Whether to include retention effects.
        include_noise: Whether to include cycle-to-cycle noise.

    Returns:
        Configured LinearStepDevice for 6T1C.

    Example:
        >>> from aihwkit.simulator.configs import SingleRPUConfig
        >>>
        >>> # Create device with 10 second batch time
        >>> device = create_6t1c_device(dt_batch_sec=10.0)
        >>> config = SingleRPUConfig(device=device)
    """
    lifetime = get_lifetime_for_dt_batch(dt_batch_sec) if include_retention else 0.0
    write_noise = 0.0182 if include_noise else 0.0
    dw_min_std = 0.3 if include_noise else 0.0

    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=dw_min_std,
        write_noise_std=write_noise,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.1 if include_retention else 0.0,
        reset=0.0,  # Decay toward 0V
        reset_dtod=0.0,
    )


# =============================================================================
# Device Info / Summary
# =============================================================================

def print_device_info():
    """Print summary of 6T1C device parameters."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    6T1C PresetDevice Summary                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Device Type: 6 Transistors, 1 Capacitor (6T1C) Synaptic Memory      ║
║  Model: LinearStepDevice                                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  UPDATE CHARACTERISTICS                                              ║
║  ────────────────────────────────────────────────────────────────────║
║    dw_min:          0.001981  (weight update step size)              ║
║    gamma_up:        -0.1678   (UP nonlinearity, slight saturation)   ║
║    gamma_down:      +0.1410   (DOWN nonlinearity, near-linear)       ║
║    ~1000 states per direction                                        ║
║    Fitting Quality: R² > 0.999                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  RETENTION CHARACTERISTICS                                           ║
║  ────────────────────────────────────────────────────────────────────║
║    Physical τ:      775.1 min (46505 sec)                            ║
║    Decay target:    0V (reset = 0.0)                                 ║
║    lifetime:        46506 (at dt_batch = 1 sec)                      ║
║    Fitting Quality: R² = 0.9935                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  AVAILABLE PRESETS                                                   ║
║  ────────────────────────────────────────────────────────────────────║
║    SixT1CPreset           - Single device with retention             ║
║    SixT1CPresetNoRetention - Single device without retention         ║
║    SixT1C2Preset          - Two devices per cross-point              ║
║    SixT1C4Preset          - Four devices per cross-point             ║
║    TikiTakaSixT1CPreset   - Tiki-taka optimizer                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_device_info()

    # Verify device creation
    print("\nVerifying device creation...")
    device = SixT1CPresetDevice()
    print(f"Device: {device}")

    print("\nVerifying preset creation...")
    preset = SixT1CPreset()
    print(f"Preset: {preset}")
