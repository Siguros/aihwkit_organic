# XOR Task Device Comparison

## Overview

This document compares three device types used in the XOR task training:
1. **Ideal Device** - Perfect floating-point behavior
2. **Linear Device** - Pulse-based with uniform steps (no noise)
3. **Fitted Device** - Real device characteristics from experimental data

---

## 1. Ideal Device

### Configuration
```python
from aihwkit.simulator.configs.devices import IdealDevice
ideal_device = IdealDevice()
```

### Characteristics
| Parameter | Value | Description |
|-----------|-------|-------------|
| bindings_class | `IdealResistiveDeviceParameter` | Internal implementation |
| diffusion | 0.0 | No weight diffusion |
| lifetime | 0.0 | No time-dependent decay |
| reset_std | 0.01 | Reset noise (not used in training) |

### Behavior
- **Perfect floating-point arithmetic**: Weight updates are exact (Δw = lr × gradient)
- **No quantization**: Continuous weight values
- **No noise**: No cycle-to-cycle or device-to-device variation
- **No non-linearity**: Update step size is independent of current weight
- **Use case**: Baseline comparison, theoretical upper bound performance

### Update Equation
```
w_new = w_old - lr × gradient
```

---

## 2. Linear Device (Uniform PiecewiseStepDevice)

### Configuration
```python
from aihwkit.simulator.configs import PiecewiseStepDevice

linear_device = PiecewiseStepDevice(
    w_min=-1.0, w_max=1.0,
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=0.002,
    dw_min_std=0.0,
    dw_min_dtod=0.0,
    up_down=0.0,
    up_down_dtod=0.0,
    write_noise_std=0.0,
    piecewise_up=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    piecewise_down=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)
```

### Characteristics
| Parameter | Value | Description |
|-----------|-------|-------------|
| bindings_class | `PiecewiseStepResistiveDeviceParameter` | Internal implementation |
| w_min / w_max | -1.0 / 1.0 | Weight bounds |
| dw_min | 0.002 | Minimum step size (~500 states per direction) |
| dw_min_std | 0.0 | No cycle-to-cycle noise |
| dw_min_dtod | 0.0 | No device-to-device variation |
| write_noise_std | 0.0 | No write noise |
| up_down | 0.0 | Symmetric up/down updates |
| piecewise_up | [1.0] × 10 | Uniform step size across all segments |
| piecewise_down | [1.0] × 10 | Uniform step size across all segments |

### Behavior
- **Pulse-based updates**: Weight changes occur in discrete steps
- **Quantization**: Weight resolution limited by dw_min (~1000 total states)
- **No noise**: All noise parameters set to 0
- **Linear response**: Step size is constant regardless of weight value
- **Use case**: Study effect of weight quantization without non-linearity

### Update Equation
```
n_pulses = round(|gradient| × lr / dw_min)
Δw = n_pulses × dw_min × sign(gradient)
w_new = clip(w_old + Δw, w_min, w_max)
```

### Piecewise Response (Linear)
```
Step Size
    ^
1.2 |
1.0 |------------------------------------  (uniform)
0.8 |
0.6 |
0.4 |
    +----+----+----+----+----+----+----+----+----+-----> Weight
   -1.0      -0.6      -0.2       0.2       0.6       1.0
```

---

## 3. Fitted Device (Non-uniform PiecewiseStepDevice)

### Configuration
```python
from aihwkit.simulator.configs import PiecewiseStepDevice

fitted_device = PiecewiseStepDevice(
    w_min=-1.0, w_max=1.0,
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=0.002,
    dw_min_std=0.0,
    dw_min_dtod=0.0,
    up_down=-0.000463,
    up_down_dtod=0.0,
    write_noise_std=0.0270,
    apply_write_noise_on_set=True,
    piecewise_up=[0.551, 0.734, 0.988, 1.025, 1.079, 1.109, 1.158, 1.227, 1.264, 1.501],
    piecewise_down=[0.461, 0.625, 0.816, 0.999, 1.196, 1.341, 1.475, 1.576, 1.638, 0.787],
)
```

### Characteristics
| Parameter | Value | Description |
|-----------|-------|-------------|
| bindings_class | `PiecewiseStepResistiveDeviceParameter` | Internal implementation |
| w_min / w_max | -1.0 / 1.0 | Weight bounds |
| dw_min | 0.002 | Base step size (from 1000 pulses) |
| write_noise_std | 0.0270 | ~2.7% write noise |
| up_down | -0.000463 | Slight asymmetry (down slightly larger) |
| piecewise_up | [0.55 → 1.50] | Non-uniform, increasing |
| piecewise_down | [0.46 → 1.64 → 0.79] | Non-uniform, peak then drop |

### Behavior
- **Pulse-based updates**: Same as Linear Device
- **Quantization**: Same dw_min as Linear Device
- **Non-linear response**: Step size varies with weight value
- **Write noise**: ~2.7% Gaussian noise on each update
- **Asymmetry**: Slight up/down asymmetry from real device
- **Use case**: Realistic simulation of actual hardware behavior

### Piecewise Response (Non-linear)
```
UP Direction (SET):
Step Size
    ^
1.6 |                                              *
1.4 |                                         *
1.2 |                               *    *  *
1.0 |                    *    *  *
0.8 |              *
0.6 |    *    *
0.4 |
    +----+----+----+----+----+----+----+----+----+-----> Weight
   -1.0      -0.6      -0.2       0.2       0.6       1.0

DOWN Direction (RESET):
Step Size
    ^
1.8 |                                         *
1.6 |                                    *
1.4 |                               *
1.2 |                          *
1.0 |                    *                              *
0.8 |              *
0.6 |         *
0.4 |    *
    +----+----+----+----+----+----+----+----+----+-----> Weight
   -1.0      -0.6      -0.2       0.2       0.6       1.0
```

### Physical Interpretation
- **UP (SET)**: Step size increases from 0.55× to 1.50× as weight increases
  - Smaller steps at low conductance (hard to increase)
  - Larger steps at high conductance (easier to increase)
- **DOWN (RESET)**: Step size increases then drops at high weight
  - Saturation effect at high conductance region
  - Last segment shows reduced step (0.79×) due to physical limits

---

## Comparison Summary

| Feature | Ideal | Linear | Fitted |
|---------|-------|--------|--------|
| **Update Type** | Floating-point | Pulse-based | Pulse-based |
| **Quantization** | None | dw_min = 0.002 | dw_min = 0.002 |
| **Non-linearity** | None | None | Yes (piecewise) |
| **Write Noise** | None | None | 2.7% |
| **Asymmetry** | None | None | -0.05% |
| **Weight Range** | Unlimited | [-1, 1] | [-1, 1] |
| **States** | Continuous | ~1000 | ~1000 |
| **Realism** | Theoretical | Simplified | Realistic |

---

## Expected Training Behavior

### Convergence Speed
1. **Ideal**: Fastest convergence (no limitations)
2. **Linear**: Slightly slower (quantization limits fine-tuning)
3. **Fitted**: Slowest (noise and non-linearity require more iterations)

### Success Rate (100% accuracy)
1. **Ideal**: Highest success rate
2. **Linear**: High success rate (no noise interference)
3. **Fitted**: Lower success rate (noise can cause instability)

### Final Loss Quality
1. **Ideal**: Lowest loss (can reach exact solution)
2. **Linear**: Limited by quantization (minimum ~dw_min² level)
3. **Fitted**: Higher variance due to noise

---

## Code Reference

File: `generate_xor_plots_all_devices.py`

```python
# 1. Ideal Device
from aihwkit.simulator.configs.devices import IdealDevice
ideal_device = IdealDevice()
ideal_config = SingleRPUConfig(device=ideal_device)

# 2. Linear Device
linear_device = PiecewiseStepDevice(
    w_min=-1, w_max=1,
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=0.002,
    dw_min_std=0.0, dw_min_dtod=0.0,
    up_down=0.0, up_down_dtod=0.0,
    write_noise_std=0.0,
    piecewise_up=[1.0]*10,
    piecewise_down=[1.0]*10,
)
linear_config = SingleRPUConfig(device=linear_device)

# 3. Fitted Device
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
fitted_device = PiecewiseStepDevice(**device_params)
fitted_config = SingleRPUConfig(device=fitted_device)
```

---

## Data Source

- **Fitted Device Parameters**: Extracted from `xor_device_config.json`
- **Experimental Data**: 1000 pulses × 10 cycles, conductance measurements
- **Fitting Quality**: R² > 0.999 for both UP and DOWN directions
