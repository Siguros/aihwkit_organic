#!/bin/bash
# Sequential experiment runner
# Waits for IdealizedPreset experiment to finish, then runs Fitted Device experiment

echo "=========================================="
echo "Sequential Experiment Runner"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Wait for the IdealizedPreset experiment (PID 1827073) to finish
IDEALIZED_PID=1827073

echo "Waiting for IdealizedPreset experiment (PID: $IDEALIZED_PID) to complete..."

while kill -0 $IDEALIZED_PID 2>/dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] IdealizedPreset still running..."
    sleep 300  # Check every 5 minutes
done

echo ""
echo "=========================================="
echo "IdealizedPreset experiment completed!"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Small delay before starting next experiment
sleep 10

echo "Starting Fitted Device experiment..."
echo ""

# Run the fitted device experiment
cd /root/aihwkit_organic/xor_task
python3 -u cifar10_resnet_fitted_300epoch.py 2>&1 | tee cifar10_fitted_300epoch.log

echo ""
echo "=========================================="
echo "Fitted Device experiment completed!"
echo "End time: $(date)"
echo "=========================================="
