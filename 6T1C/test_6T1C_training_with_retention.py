# -*- coding: utf-8 -*-

"""6T1C Training Test with Realistic Retention Characteristics.

This script demonstrates training with 6T1C device characteristics including:
1. Update nonlinearity (gamma_up, gamma_down)
2. Retention/decay (lifetime based on physical τ=775 min)
3. Different retention scenarios (fast vs slow decay)

The goal is to show how retention affects training and model performance.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from preset_6T1C import (
    SixT1CPreset,
    SixT1CPresetNoRetention,
    SixT1CPresetDevice,
    get_lifetime_for_dt_batch,
    print_device_info,
)
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig


# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Training settings
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.05

# Retention scenarios to test
# dt_batch represents assumed physical time per mini-batch
RETENTION_SCENARIOS = {
    "No Retention": None,           # No decay
    "Slow (1s/batch)": 1,           # τ=775min, ~13 hours to significant decay
    "Medium (1min/batch)": 60,      # τ=775min, reasonable for edge device
    "Fast (10min/batch)": 600,      # τ=775min, aggressive decay scenario
}


# =============================================================================
# Dataset: Simple Classification Task
# =============================================================================

def create_classification_dataset(n_samples=5000, n_features=20, n_classes=5):
    """Create a simple classification dataset."""
    torch.manual_seed(SEED)

    # Generate clustered data
    X = []
    y = []

    for class_idx in range(n_classes):
        # Random cluster center
        center = torch.randn(n_features) * 2
        # Points around cluster
        points = center + torch.randn(n_samples // n_classes, n_features) * 0.5
        X.append(points)
        y.extend([class_idx] * (n_samples // n_classes))

    X = torch.cat(X, dim=0)
    y = torch.tensor(y)

    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, n_features, n_classes


# =============================================================================
# Model Creation
# =============================================================================

def create_model(n_features, n_classes, dt_batch=None):
    """Create analog model with specified retention setting.

    Args:
        n_features: Input feature dimension
        n_classes: Number of output classes
        dt_batch: Time per batch in seconds. None = no retention.
    """
    if dt_batch is None:
        # No retention
        rpu_config = SixT1CPresetNoRetention()
    else:
        # With retention based on dt_batch
        device = SixT1CPresetDevice()
        device.lifetime = get_lifetime_for_dt_batch(dt_batch)
        rpu_config = SingleRPUConfig(device=device)

    model = AnalogSequential(
        AnalogLinear(n_features, 64, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(64, 32, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(32, n_classes, bias=True, rpu_config=rpu_config),
    )

    return model.to(DEVICE)


def get_weight_stats(model):
    """Get weight statistics from model."""
    all_weights = []
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            for tile in module.analog_tiles():
                w, _ = tile.get_weights()
                all_weights.append(w.cpu().numpy().flatten())

    weights = np.concatenate(all_weights)
    return {
        'mean': weights.mean(),
        'std': weights.std(),
        'min': weights.min(),
        'max': weights.max(),
        'abs_mean': np.abs(weights).mean(),
    }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), 100. * correct / total


def simulate_idle_decay(model, n_idle_steps):
    """Simulate idle time by applying decay without training."""
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            for tile in module.analog_tiles():
                for _ in range(n_idle_steps):
                    tile.decay_weights(alpha=1.0)


# =============================================================================
# Main Training Experiment
# =============================================================================

def run_training_experiment():
    """Run training with different retention scenarios."""
    print("=" * 70)
    print("6T1C Training with Retention Characteristics")
    print("=" * 70)

    # Print device info
    print_device_info()

    # Create dataset
    print("\n[1] Creating dataset...")
    train_loader, test_loader, n_features, n_classes = create_classification_dataset()
    print(f"    Features: {n_features}, Classes: {n_classes}")
    print(f"    Train samples: {len(train_loader.dataset)}")
    print(f"    Test samples: {len(test_loader.dataset)}")

    # Results storage
    results = {}

    # Train with each retention scenario
    print("\n[2] Training with different retention scenarios...")

    for scenario_name, dt_batch in RETENTION_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        if dt_batch:
            lifetime = get_lifetime_for_dt_batch(dt_batch)
            print(f"  dt_batch={dt_batch}s, lifetime={lifetime:.0f}")
        else:
            print(f"  No retention (lifetime=0)")
        print("="*60)

        # Set seed for reproducibility
        torch.manual_seed(SEED)

        # Create model
        model = create_model(n_features, n_classes, dt_batch)

        # Setup training
        optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
        optimizer.regroup_param_groups(model)
        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'weight_mean': [],
            'weight_std': [],
        }

        # Initial evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        weight_stats = get_weight_stats(model)
        print(f"  Initial - Test Acc: {test_acc:.2f}%, Weight Mean: {weight_stats['mean']:.4f}")

        # Training loop
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            weight_stats = get_weight_stats(model)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['weight_mean'].append(weight_stats['mean'])
            history['weight_std'].append(weight_stats['std'])

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: Train Acc={train_acc:.1f}%, "
                      f"Test Acc={test_acc:.1f}%, "
                      f"W_mean={weight_stats['mean']:.4f}, "
                      f"W_std={weight_stats['std']:.4f}")

        results[scenario_name] = {
            'history': history,
            'final_acc': test_acc,
            'model': model,
            'dt_batch': dt_batch,
        }

    return results, train_loader, test_loader


def test_retention_after_training(results, test_loader):
    """Test model performance after simulated idle time."""
    print("\n" + "=" * 70)
    print("[3] Testing Retention After Training (Simulated Idle Time)")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()
    idle_results = {}

    # Only test models with retention
    for scenario_name, data in results.items():
        if data['dt_batch'] is None:
            continue

        print(f"\n{scenario_name}:")
        model = data['model']

        # Get accuracy right after training
        _, acc_before = evaluate(model, test_loader, criterion)
        print(f"  Accuracy right after training: {acc_before:.2f}%")

        # Simulate different idle periods
        idle_periods = [10, 50, 100, 200]
        accuracies = [acc_before]

        for n_idle in idle_periods:
            # Apply decay
            simulate_idle_decay(model, n_idle)
            _, acc_after = evaluate(model, test_loader, criterion)
            accuracies.append(acc_after)

            weight_stats = get_weight_stats(model)
            print(f"  After {sum(idle_periods[:idle_periods.index(n_idle)+1]):3d} idle steps: "
                  f"Acc={acc_after:.2f}%, W_mean={weight_stats['mean']:.4f}")

        idle_results[scenario_name] = {
            'idle_steps': [0] + list(np.cumsum(idle_periods)),
            'accuracies': accuracies,
        }

    return idle_results


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results, idle_results):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('6T1C Training with Retention Characteristics', fontsize=14, fontweight='bold')

    colors = {'No Retention': 'blue', 'Slow (1s/batch)': 'green',
              'Medium (1min/batch)': 'orange', 'Fast (10min/batch)': 'red'}

    # Plot 1: Test Accuracy
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['history']['test_acc'], color=colors[name],
                linewidth=2, label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy During Training')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data['history']['train_loss'], color=colors[name],
                linewidth=2, label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Weight Mean Evolution
    ax3 = axes[0, 2]
    for name, data in results.items():
        ax3.plot(data['history']['weight_mean'], color=colors[name],
                linewidth=2, label=name)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean Weight')
    ax3.set_title('Weight Mean During Training')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Weight Std Evolution
    ax4 = axes[1, 0]
    for name, data in results.items():
        ax4.plot(data['history']['weight_std'], color=colors[name],
                linewidth=2, label=name)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Weight Std')
    ax4.set_title('Weight Standard Deviation')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Accuracy After Idle Time
    ax5 = axes[1, 1]
    for name, data in idle_results.items():
        ax5.plot(data['idle_steps'], data['accuracies'],
                color=colors[name], linewidth=2, marker='o', label=name)
    ax5.set_xlabel('Idle Steps (decay cycles)')
    ax5.set_ylabel('Test Accuracy (%)')
    ax5.set_title('Accuracy Degradation After Idle Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Final Accuracy Comparison
    ax6 = axes[1, 2]
    names = list(results.keys())
    final_accs = [results[n]['final_acc'] for n in names]
    bars = ax6.bar(range(len(names)), final_accs, color=[colors[n] for n in names])
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=15)
    ax6.set_ylabel('Final Test Accuracy (%)')
    ax6.set_title('Final Accuracy Comparison')
    ax6.set_ylim([0, 100])

    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('6T1C_training_with_retention.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: 6T1C_training_with_retention.png")
    plt.close()


def print_summary(results, idle_results):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print("\n  Final Test Accuracy by Scenario:")
    print("  " + "-" * 50)
    for name, data in results.items():
        dt_info = f"(dt={data['dt_batch']}s)" if data['dt_batch'] else "(no decay)"
        print(f"    {name:25s}: {data['final_acc']:.2f}% {dt_info}")

    print("\n  Accuracy After 360 Idle Steps:")
    print("  " + "-" * 50)
    for name, data in idle_results.items():
        final_idle_acc = data['accuracies'][-1]
        initial_acc = data['accuracies'][0]
        degradation = initial_acc - final_idle_acc
        print(f"    {name:25s}: {final_idle_acc:.2f}% (degraded {degradation:.1f}%)")

    print(f"""
  Key Observations:
  ─────────────────────────────────────────────────────────────────
  1. Training Performance:
     - All scenarios can learn the task
     - Faster decay (shorter lifetime) may require higher learning rate
     - Weight magnitude decreases with stronger retention

  2. Weight Evolution:
     - With retention: weights drift toward 0 during training
     - Without retention: weights stabilize at learned values

  3. Post-Training Degradation:
     - Models with retention lose accuracy during idle time
     - Faster decay = faster accuracy degradation
     - This simulates real capacitor leakage behavior

  4. Practical Implications:
     - 6T1C devices need periodic refresh for long-term deployment
     - Training may need to account for expected inference latency
     - Trade-off between retention and model stability
""")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("6T1C DEVICE TRAINING WITH REALISTIC RETENTION")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    # Run training experiment
    results, train_loader, test_loader = run_training_experiment()

    # Test retention after training
    idle_results = test_retention_after_training(results, test_loader)

    # Create visualization
    print("\n[4] Creating visualization...")
    plot_results(results, idle_results)

    # Print summary
    print_summary(results, idle_results)

    print("=" * 70)
    print("Experiment Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
