# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit baseline: Wide & Shallow ResNet with CIFAR10 using Fitted PiecewiseStepDevice.

CIFAR10 dataset on a Wide & Shallow ResNet (depth ~10, width 2x) network with configurable
digital (FloatingPoint) and analog (PiecewiseStepDevice) layers.
Uses fitted device from xor_device_config.json.

Architecture Changes from ResNet18:
- Blocks: (2,2,2,2) → (1,1,1,1) - Reduces depth from 18 to ~10 layers
- Channels: [64,128,256,512] → [128,256,512,1024] - Doubles width
- Purpose: Reduce noise accumulation while maintaining model capacity
"""
# pylint: disable=invalid-name

# Imports
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports from PyTorch.
import torch
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

# Progress bar
from tqdm import tqdm

# For t-SNE analysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs import MappingParameter, PiecewiseStepDevice, IdealDevice


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results - CHANGED for Wide & Shallow architecture with Fitted Device
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_WIDE_SHALLOW_FITTED_300EPOCH")
os.makedirs(RESULTS, exist_ok=True)
# Note: WEIGHT_PATH will be set in main() based on N_EPOCHS
WEIGHT_PATH = None  # Will be set dynamically

# Training parameters
SEED = 1
N_EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0005  # L2 regularization
NESTEROV = True  # Nesterov momentum
WARMUP_RATIO = 0.04  # Warmup ratio
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# Device Configuration
# Load Fitted Device Configuration from JSON
DEVICE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'xor_device_config.json')

with open(DEVICE_CONFIG_PATH, 'r') as f:
    device_config = json.load(f)

# Extract device parameters (exclude metadata)
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}

# Override write_noise_std to 0 for noise-free test
device_params['write_noise_std'] = 0.0

ANALOG_DEVICE = PiecewiseStepDevice(**device_params)

print(f"Loaded fitted device config from {DEVICE_CONFIG_PATH}:")
print(f"  dw_min: {device_config.get('dw_min', 'N/A')}")
print(f"  write_noise_std: {device_params['write_noise_std']} (OVERRIDDEN: noise-free test)")

# Option: IdealDevice (commented out for baseline comparison)
# ANALOG_DEVICE = IdealDevice()
# print(f"Using IdealDevice (noise-free baseline)")

# Layer-wise digital/analog configuration
# Set which layers use analog vs digital (FloatingPoint)
# Options: 'analog' (trainable base), 'digital' (FloatingPoint)
#
# Wide & Shallow ResNet structure (1 block per layer):
# - conv1: First 3x3 conv layer
# - layer1: 1 block, NO downsample (128 -> 128 channels)
# - layer2: 1 block, downsample in block0 (128 -> 256 channels)
# - layer3: 1 block, downsample in block0 (256 -> 512 channels)
# - layer4: 1 block, downsample in block0 (512 -> 1024 channels)
# - fc: Final fully connected layer
LAYER_CONFIG = {
    'conv1': 'digital',           # First convolutional layer

    # Layer1 (1 block, no downsample)
    'layer1_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
    },

    # Layer2 (1 block, downsample in block0)
    'layer2_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
        'downsample': 'analog',
    },

    # Layer3 (1 block, downsample in block0)
    'layer3_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
        'downsample': 'analog',
    },

    # Layer4 (1 block, downsample in block0)
    'layer4_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
        'downsample': 'analog',
    },

    'fc': 'digital',              # Final fully connected layer
}


def create_analog_config():
    """Create analog configuration using fitted PiecewiseStepDevice.

    Returns:
        SingleRPUConfig: Configuration with fitted PiecewiseStepDevice from xor_device_config.json
    """
    # Use SingleRPUConfig with the fitted device
    config = SingleRPUConfig(device=ANALOG_DEVICE)

    # Add mapping for larger layers - CHANGED for wider channels (up to 1024)
    config.mapping = MappingParameter(
        weight_scaling_omega=1.0,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=1024,    # CHANGED: 512 → 1024
        max_output_size=1024    # CHANGED: 512 → 1024
    )

    return config


class ResidualBlockBaseline(nn.Module):
    """Residual block with configurable digital/analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1,
                 use_analog_conv1=False, use_analog_conv2=False, use_analog_convskip=False):
        super().__init__()

        # Conv1 configuration
        if use_analog_conv1:
            rpu_config_conv1 = create_analog_config()
            bias_conv1 = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)
        else:
            rpu_config_conv1 = FloatingPointRPUConfig()
            bias_conv1 = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)

        # Conv2 configuration
        if use_analog_conv2:
            rpu_config_conv2 = create_analog_config()
            bias_conv2 = False
        else:
            rpu_config_conv2 = FloatingPointRPUConfig()
            bias_conv2 = False

        # Convskip configuration
        if use_analog_convskip:
            rpu_config_convskip = create_analog_config()
            bias_convskip = False
        else:
            rpu_config_convskip = FloatingPointRPUConfig()
            bias_convskip = False

        # Build layers with individual configurations
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch,
            kernel_size=3, padding=1, stride=stride,
            bias=bias_conv1,
            rpu_config=rpu_config_conv1
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)

        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=bias_conv2,
            rpu_config=rpu_config_conv2
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=bias_convskip,
                rpu_config=rpu_config_convskip
            )
            self.bn_skip = nn.BatchNorm2d(hidden_ch)  # Add BN for shortcut
        else:
            self.convskip = None
            self.bn_skip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.bn_skip(self.convskip(x))  # Apply BN to shortcut
        y += x
        return F.relu(y)


def concatenate_layer_blocks_baseline(in_ch, hidden_ch, num_layer, first_layer=False,
                                  block_configs=None):
    """Concatenate multiple residual blocks to form a layer.

    Args:
        in_ch: Input channels
        hidden_ch: Hidden channels
        num_layer: Number of residual blocks
        first_layer: Whether this is the first layer
        block_configs: List of config dicts for each block, each containing:
                      {'conv1': 'analog'/'digital', 'conv2': 'analog'/'digital',
                       'downsample': 'analog'/'digital' (optional)}

    Returns:
       List: list of layer blocks
    """
    if block_configs is None:
        # Default: all digital
        block_configs = [{'conv1': 'digital', 'conv2': 'digital', 'downsample': 'digital'}] * num_layer

    layers = []
    for i in range(num_layer):
        config = block_configs[i]
        use_analog_conv1 = (config['conv1'] == 'analog')
        use_analog_conv2 = (config['conv2'] == 'analog')
        use_analog_downsample = (config.get('downsample', 'digital') == 'analog')

        if i == 0 and not first_layer:
            # First block with downsampling
            layers.append(ResidualBlockBaseline(
                in_ch, hidden_ch, use_conv=True, stride=2,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_downsample
            ))
        else:
            # Other blocks without downsampling
            layers.append(ResidualBlockBaseline(
                hidden_ch, hidden_ch,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_conv1  # Not used, but kept for consistency
            ))
    return layers


def create_model():
    """Wide & Shallow ResNet model with configurable digital/analog layers.

    Returns:
       nn.Module: created model
    """

    # CHANGED: 1 block per layer instead of 2 (shallow)
    block_per_layers = (1, 1, 1, 1)  # Wide & Shallow structure
    # CHANGED: 2x wider channels
    base_channel = 128  # CHANGED: 64 → 128
    channel = (base_channel, 2 * base_channel, 4 * base_channel, 8 * base_channel)  # (128, 256, 512, 1024)

    print(f"\n=== Architecture: Wide & Shallow ResNet ===")
    print(f"  Blocks per layer: {block_per_layers}")
    print(f"  Channels: {channel}")
    print(f"  Total depth: ~10 layers (vs 18 in standard ResNet18)")
    print(f"  Noise accumulation: √10 ≈ 3.16x (vs √18 ≈ 4.24x)")
    print(f"  Expected improvement: 25% less noise\n")

    # Input layer - use configuration from LAYER_CONFIG
    input_use_analog = (LAYER_CONFIG['conv1'] == 'analog')
    if input_use_analog:
        input_rpu_config = create_analog_config()
        input_bias = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)
    else:
        input_rpu_config = FloatingPointRPUConfig()
        input_bias = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)

    l0 = nn.Sequential(
        AnalogConv2d(
            3, channel[0],
            kernel_size=3, stride=1, padding=1,
            bias=input_bias,
            rpu_config=input_rpu_config
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    # Residual blocks - use per-block configuration from LAYER_CONFIG
    # CHANGED: Only 1 block per layer
    # Layer1 (1 block, no downsample)
    l1 = nn.Sequential(
        *concatenate_layer_blocks_baseline(
            channel[0], channel[0], block_per_layers[0],
            first_layer=True,
            block_configs=[
                LAYER_CONFIG['layer1_block0'],
            ]
        )
    )

    # Layer2 (1 block, downsample in block0)
    l2 = nn.Sequential(
        *concatenate_layer_blocks_baseline(
            channel[0], channel[1], block_per_layers[1],
            block_configs=[
                LAYER_CONFIG['layer2_block0'],
            ]
        )
    )

    # Layer3 (1 block, downsample in block0)
    l3 = nn.Sequential(
        *concatenate_layer_blocks_baseline(
            channel[1], channel[2], block_per_layers[2],
            block_configs=[
                LAYER_CONFIG['layer3_block0'],
            ]
        )
    )

    # Layer4 (1 block, downsample in block0)
    l4_conv = nn.Sequential(
        *concatenate_layer_blocks_baseline(
            channel[2], channel[3], block_per_layers[3],
            block_configs=[
                LAYER_CONFIG['layer4_block0'],
            ]
        )
    )

    # Final classification layer - use configuration from LAYER_CONFIG
    fc_use_analog = (LAYER_CONFIG['fc'] == 'analog')
    if fc_use_analog:
        fc_rpu_config = create_analog_config()
        fc_bias = True
    else:
        fc_rpu_config = FloatingPointRPUConfig()
        fc_bias = True

    l5_fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[3], N_CLASSES,  # 1024 -> 10 for CIFAR-10 (CHANGED: 512 → 1024)
            bias=fc_bias,
            rpu_config=fc_rpu_config
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4_conv, l5_fc)

    # Print configuration summary
    def format_block_config(block_name):
        """Format block configuration for printing"""
        config = LAYER_CONFIG[block_name]
        parts = []
        for conv_type in ['conv1', 'conv2', 'downsample']:
            if conv_type in config:
                parts.append(f"{conv_type}={'A' if config[conv_type] == 'analog' else 'D'}")
        return f"{block_name}: {', '.join(parts)}"

    print(f"\nCreated Wide & Shallow ResNet with per-block analog/digital layer configuration:")
    print(f"  Analog device type: PiecewiseStepDevice (fitted from xor_device_config.json)")
    print(f"  conv1: {'Analog (trainable base)' if input_use_analog else 'Digital (FloatingPoint)'}")
    print(f"  Layer1:")
    print(f"    {format_block_config('layer1_block0')}")
    print(f"  Layer2:")
    print(f"    {format_block_config('layer2_block0')}")
    print(f"  Layer3:")
    print(f"    {format_block_config('layer3_block0')}")
    print(f"  Layer4:")
    print(f"    {format_block_config('layer4_block0')}")
    print(f"  fc: {'Analog (trainable base)' if fc_use_analog else 'Digital (FloatingPoint)'}")
    print(f"  Using random initialization (no pretrained weights)\n")

    # Apply Kaiming initialization to ensure consistent initialization
    initialize_resnet_weights(model)

    return model


def initialize_resnet_weights(model):
    """Apply PyTorch ResNet-style kaiming_normal initialization to all layers.

    This ensures the initialization matches standard PyTorch ResNet18 behavior
    for consistent results across different implementations.

    Args:
        model (nn.Module): Model to initialize
    """
    import math

    print("\nApplying ResNet-style Kaiming initialization...")

    for name, module in model.named_modules():
        if isinstance(module, AnalogConv2d):
            # For AnalogConv2d layers, initialize the analog tile weights
            if hasattr(module, 'analog_module'):
                # Get the weight dimensions
                if hasattr(module, 'out_channels') and hasattr(module, 'in_channels'):
                    out_channels = module.out_channels
                    in_channels = module.in_channels
                    kernel_size = module.kernel_size

                    # Create temporary weight tensor with correct shape for initialization
                    if isinstance(kernel_size, tuple):
                        k_h, k_w = kernel_size
                    else:
                        k_h = k_w = kernel_size

                    temp_weight = torch.empty(out_channels, in_channels, k_h, k_w)

                    # Apply kaiming_normal initialization (ResNet default)
                    # mode='fan_out', nonlinearity='relu' for Conv2d in ResNet
                    nn.init.kaiming_normal_(temp_weight, mode='fan_out', nonlinearity='relu')

                    # Set the initialized weights to the analog tile
                    try:
                        # Check if we need to reshape for analog tile format
                        if hasattr(module.analog_module, 'get_weights'):
                            weights, bias = module.analog_module.get_weights()
                            weight_shape = weights.shape

                            # Reshape temp_weight to match analog tile format
                            if len(weight_shape) == 2:
                                # Flattened format [out_ch, in_ch*k*k]
                                reshaped_weight = temp_weight.view(out_channels, in_channels * k_h * k_w)
                                module.analog_module.set_weights(reshaped_weight, bias)
                            else:
                                # Conv format [out_ch, in_ch, k, k]
                                module.analog_module.set_weights(temp_weight, bias)

                            print(f"  Initialized {name}: Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size})")
                    except Exception as e:
                        print(f"  Warning: Could not initialize {name}: {e}")

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: constant initialization (weight=1, bias=0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, AnalogLinear):
            # For AnalogLinear (FC layer)
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'get_weights'):
                try:
                    weights, bias = module.analog_module.get_weights()
                    in_features = module.in_features if hasattr(module, 'in_features') else weights.shape[1]
                    out_features = module.out_features if hasattr(module, 'out_features') else weights.shape[0]

                    # Create temporary weight for initialization
                    temp_weight = torch.empty(out_features, in_features)

                    # Apply kaiming_uniform initialization for Linear layers (PyTorch default)
                    nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))

                    # Set to analog tile
                    module.analog_module.set_weights(temp_weight, bias)
                    print(f"  Initialized {name}: Linear({in_features}, {out_features})")
                except Exception as e:
                    print(f"  Warning: Could not initialize {name}: {e}")

    print("Weight initialization completed\n")


def load_images():
    """Load images for train from torchvision datasets with data augmentation.

    Returns:
        Dataset, Dataset: train data and validation data"""
    # Training transforms with basic augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=val_transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate, momentum=0.9, weight_decay=5e-4):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate
        momentum (float): momentum factor for SGD
        weight_decay (float): weight decay factor

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=NESTEROV
    )
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer, epoch_num):
    """Train network for one epoch.

    Args:
        train_data (DataLoader): Training data loader
        model (nn.Module): Model to be trained
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        epoch_num (int): Current epoch number

    Returns:
        nn.Module, Optimizer, float, float: model, optimizer, epoch loss, epoch accuracy
    """
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    # Create progress bar
    desc = f"Epoch {epoch_num}"
    pbar = tqdm(train_data, desc=desc, leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()

        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    epoch_loss = total_loss / len(train_data.dataset)
    epoch_acc = 100 * correct / total

    return model, optimizer, epoch_loss, epoch_acc



def test_evaluation(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    # Create progress bar for validation
    pbar = tqdm(validation_data, desc="Validating", leave=False)

    with no_grad():
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

            # Update progress bar
            current_acc = 100 * predicted_ok / total_images
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = total_loss / len(validation_data.dataset)
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    return model, epoch_loss, error, accuracy


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.0, min_lr=1e-5):
    """Apply learning rate warmup + cosine annealing.

    Args:
        optimizer: SGD optimizer
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lr: Base learning rate
        warmup_ratio: Fraction of epochs for warmup (0.0 = no warmup)
        min_lr: Minimum learning rate
    """
    import math

    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch <= warmup_epochs:
        # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
        current_lr = base_lr * (epoch / warmup_epochs)
    else:
        # Cosine annealing after warmup
        # Progress through the cosine schedule (0 to π)
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        current_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def main():
    """Train a PyTorch Wide & Shallow ResNet model with mixed analog/digital to classify CIFAR10."""
    # Seed
    manual_seed(SEED)

    # Load the images.
    train_data, validation_data = load_images()

    # Make the model
    model = create_model()

    # Initialize weights with Kaiming normal (ResNet default)
    initialize_resnet_weights(model)

    if USE_CUDA:
        model = model.to(DEVICE)

    print(f"Model moved to {DEVICE}")

    # Count parameters - analog tiles don't register weights as PyTorch parameters
    pytorch_params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch registered parameters: {pytorch_params:,}")
    print("Note: Analog tile weights are stored internally in C++ and not counted here")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_sgd_optimizer(model, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)

    print(f"\nUsing fitted PiecewiseStepDevice for analog layers (dw_min={device_config.get('dw_min', 'N/A')}, write_noise_std=0.0 [NOISE-FREE TEST])")

    best_accuracy = 0
    best_epoch = 0

    # Set weight path based on N_EPOCHS
    global WEIGHT_PATH
    WEIGHT_PATH = os.path.join(RESULTS, f"wide_shallow_model_weight_{N_EPOCHS}epoch.pth")

    print("\nStarting Wide & Shallow ResNet training on CIFAR10...")
    print("=" * 60)

    # Special case: Save initial model when N_EPOCHS = 0
    if N_EPOCHS == 0:
        save(model.state_dict(), WEIGHT_PATH)
        print(f"\nN_EPOCHS = 0: Initial model saved to {WEIGHT_PATH}")
        print("No training performed.")
        return

    # Training history for saving
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    # Create overall progress bar for epochs
    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    for epoch in epoch_pbar:
        # Apply warmup + cosine annealing learning rate schedule
        apply_warmup_cosine_lr(optimizer, epoch + 1, N_EPOCHS, LEARNING_RATE, WARMUP_RATIO)
        current_lr = optimizer.param_groups[0]['lr']

        # Train one epoch
        model, optimizer, train_loss, train_acc = train_step(
            train_data, model, criterion, optimizer, epoch + 1
        )

        # Run validation after each epoch
        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rate'].append(current_lr)

        # Track best accuracy (for logging only, not for saving)
        latest_val_acc = val_accuracy
        if latest_val_acc > best_accuracy:
            best_accuracy = latest_val_acc
            best_epoch = epoch
            # Save best model
            best_model_path = os.path.join(RESULTS, f"wide_shallow_best_model_{N_EPOCHS}epoch.pth")
            save(model.state_dict(), best_model_path)

        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.2f}%',
            'Val_Acc': f'{latest_val_acc:.2f}%',
            'Best': f'{best_accuracy:.2f}%'
        })

        # Print detailed progress
        if (epoch + 1) % 1 == 0:
            val_info = f", Val Acc {latest_val_acc:.2f}%"
            tqdm.write(f"Epoch {epoch + 1:3d}: "
                      f"Train Loss {train_loss:.4f} (Acc {train_acc:.2f}%)"
                      f"{val_info}")

    # Save the final epoch model (not best model)
    print("=" * 60)
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Final epoch validation accuracy: {latest_val_acc:.2f}%")
    print(f"Saving final epoch model to: {WEIGHT_PATH}")
    save(model.state_dict(), WEIGHT_PATH)
    print(f"✓ Model weights saved (final epoch)")

    # =========================================================================
    # Save Results to Excel and Plot
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving results...")

    # Save to Excel
    df = pd.DataFrame(history)
    excel_path = os.path.join(RESULTS, f"training_history_{N_EPOCHS}epoch.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"✓ Training history saved to: {excel_path}")

    # Save to CSV as backup
    csv_path = os.path.join(RESULTS, f"training_history_{N_EPOCHS}epoch.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Training history saved to: {csv_path}")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['epoch'], history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[0, 1].plot(history['epoch'], history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['epoch'], history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].axhline(y=best_accuracy, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_accuracy:.2f}%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Learning rate schedule
    axes[1, 0].plot(history['epoch'], history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule (Warmup + Cosine)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Train-Val gap (generalization)
    train_val_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(history['epoch'], train_val_gap, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
    axes[1, 1].set_title('Generalization Gap')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Wide & Shallow ResNet CIFAR10 ({N_EPOCHS} epochs)\n'
                 f'Best Val Acc: {best_accuracy:.2f}% @ Epoch {best_epoch + 1}', fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS, f"training_curves_{N_EPOCHS}epoch.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {plot_path}")

    # =========================================================================
    # Detailed Analysis (t-SNE, Confusion Matrix)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running detailed analysis (t-SNE, Confusion Matrix)...")

    # Get feature embeddings and predictions
    model.eval()
    all_features = []
    all_labels = []
    all_preds = []

    # Extract features from the layer before FC (after AdaptiveAvgPool)
    # We'll use a hook to get intermediate features
    features_hook = []

    def hook_fn(module, input, output):
        features_hook.append(output.detach().cpu())

    # Register hook on the flatten layer (before FC)
    # model structure: [l0, l1, l2, l3, l4_conv, l5_fc]
    # l5_fc = [AdaptiveAvgPool, Flatten, AnalogLinear]
    hook = model[5][1].register_forward_hook(hook_fn)  # Flatten layer

    with no_grad():
        for images, labels in tqdm(validation_data, desc="Extracting features"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch_max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    hook.remove()

    # Concatenate all features
    all_features = torch.cat(features_hook, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print(f"Feature shape: {all_features.shape}")

    # t-SNE visualization
    print("Running t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)

    # CIFAR10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Plot t-SNE
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # t-SNE colored by true labels
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                                c=all_labels, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title('t-SNE (True Labels)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    cbar1 = plt.colorbar(scatter1, ax=axes[0], ticks=range(10))
    cbar1.ax.set_yticklabels(class_names)

    # t-SNE colored by predictions
    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1],
                                c=all_preds, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_title('t-SNE (Predictions)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    cbar2 = plt.colorbar(scatter2, ax=axes[1], ticks=range(10))
    cbar2.ax.set_yticklabels(class_names)

    plt.suptitle(f'Feature Embeddings - Wide & Shallow ResNet\nVal Acc: {best_accuracy:.2f}%', fontsize=14)
    plt.tight_layout()

    tsne_path = os.path.join(RESULTS, f"tsne_analysis_{N_EPOCHS}epoch.png")
    plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE visualization saved to: {tsne_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.suptitle(f'Confusion Matrix - Wide & Shallow ResNet\nVal Acc: {best_accuracy:.2f}%', fontsize=14)
    plt.tight_layout()

    cm_path = os.path.join(RESULTS, f"confusion_matrix_{N_EPOCHS}epoch.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_path}")

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy (%)': per_class_acc,
        'Correct': cm.diagonal(),
        'Total': cm.sum(axis=1)
    })
    per_class_path = os.path.join(RESULTS, f"per_class_accuracy_{N_EPOCHS}epoch.xlsx")
    per_class_df.to_excel(per_class_path, index=False)
    print(f"✓ Per-class accuracy saved to: {per_class_path}")

    # Save t-SNE coordinates for further analysis
    tsne_df = pd.DataFrame({
        'tsne_1': features_2d[:, 0],
        'tsne_2': features_2d[:, 1],
        'true_label': all_labels,
        'predicted_label': all_preds,
        'correct': all_labels == all_preds
    })
    tsne_coords_path = os.path.join(RESULTS, f"tsne_coordinates_{N_EPOCHS}epoch.xlsx")
    tsne_df.to_excel(tsne_coords_path, index=False)
    print(f"✓ t-SNE coordinates saved to: {tsne_coords_path}")

    # Summary report
    summary = {
        'experiment': 'Wide & Shallow ResNet CIFAR10 with PiecewiseStepDevice',
        'architecture': {
            'blocks_per_layer': [1, 1, 1, 1],
            'channels': [128, 256, 512, 1024],
            'total_depth': '~10 layers',
            'noise_accumulation': '√10 ≈ 3.16x'
        },
        'epochs': N_EPOCHS,
        'best_val_accuracy': best_accuracy,
        'best_epoch': best_epoch + 1,
        'final_val_accuracy': latest_val_acc,
        'final_train_accuracy': history['train_acc'][-1],
        'device_config': {
            'dw_min': device_config.get('dw_min'),
            'write_noise_std': device_params.get('write_noise_std'),  # 0.0 (overridden)
            'original_write_noise_std': device_config.get('write_noise_std'),  # 0.027 (original)
        },
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY,
            'warmup_ratio': WARMUP_RATIO,
        },
        'per_class_accuracy': {name: acc for name, acc in zip(class_names, per_class_acc)}
    }

    summary_path = os.path.join(RESULTS, f"experiment_summary_{N_EPOCHS}epoch.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Experiment summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("All results saved to:", RESULTS)
    print("=" * 60)


if __name__ == "__main__":
    main()
