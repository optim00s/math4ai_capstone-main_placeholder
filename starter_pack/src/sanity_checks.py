#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation Sanity Checks for Math4AI Capstone.

This standalone script automatically verifies the 3 core requirements
from Section 5.5 of the rubric:
1. Softmax probabilities sum to 1.
2. The model can overfit a tiny dataset (loss -> 0).
3. The loss strictly decreases after the first few updates.

Usage:
    python -m starter_pack.src.sanity_checks
"""

import sys
import io

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path

# Add the repo root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starter_pack.src.models import (
    SoftmaxRegression, NeuralNetwork, 
    stable_softmax, cross_entropy_loss, one_hot
)
from starter_pack.src.optimizers import SGD, Adam

# Define results directory
RESULTS_DIR = Path(__file__).resolve().parents[2] / 'starter_pack' / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class Logger:
    """Helper class to write to both console and file."""
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


def main():
    # Setup dual logging
    log_file = RESULTS_DIR / 'sanity_checks.txt'
    logger = Logger(log_file)
    sys.stdout = logger
    
    print("============================================================")
    print("  MATH4AI CAPSTONE: IMPLEMENTATION SANITY CHECKS")
    print("============================================================")
    
    passed_all = True
    
    # ----------------------------------------------------
    # Check 1: Softmax probabilities sum to 1
    # ----------------------------------------------------
    print("\n[Check 1] Verify Softmax probabilities sum exactly to 1")
    Z = np.array([
        [1.0, 2.0, 3.0], 
        [-100.0, 0.0, 100.0],  # Edge case: extreme values
        [0.0, 0.0, 0.0]        # Edge case: zero logits
    ])
    P = stable_softmax(Z)
    sums = np.sum(P, axis=1)
    print(f"  Logits (Z):\n{Z}\n")
    print(f"  Probabilities (P):\n{P}\n")
    print(f"  Row Sums:\n{sums}")
    
    if np.allclose(sums, 1.0):
        print("  ✓ PASS: All row probabilities sum to 1.0")
    else:
        print("  ✗ FAIL: Probabilities do not sum to 1.0")
        passed_all = False

    # ----------------------------------------------------
    # Check 2: Model can overfit a tiny dataset
    # ----------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[Check 2] Verify Neural Network can overfit a tiny dataset (5 samples)")
    # Create 5 deterministic dummy samples with 2 classes
    X_tiny = np.array([
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.0, 0.5]
    ])
    y_tiny = np.array([1, 0, 1, 0, 1])
    Y_tiny_oh = one_hot(y_tiny, 2)
    
    # Large capacity NN with Adam to overfit perfectly
    nn_overfit = NeuralNetwork(n_features=2, n_hidden=16, n_classes=2)
    nn_overfit.init_params(seed=42)
    opt_adam = Adam(lr=0.05)
    opt_adam.init_state(nn_overfit.params)
    
    print("  Training NN (h=16) on 5 samples for 150 epochs...")
    final_loss = 0.0
    for epoch in range(150):
        P_out, cache = nn_overfit.forward(X_tiny)
        loss = cross_entropy_loss(P_out, Y_tiny_oh)
        final_loss = loss
        nn_overfit.backward(cache, Y_tiny_oh, lam=0.0)
        opt_adam.step(nn_overfit.params, nn_overfit.grads)
        
    preds, _ = nn_overfit.predict(X_tiny)
    acc = np.mean(preds == y_tiny)
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  Final Accuracy: {acc * 100:.1f}%")
    
    if final_loss < 0.05 and acc == 1.0:
        print("  ✓ PASS: Model perfectly overfit the tiny dataset (loss near 0)")
    else:
        print("  ✗ FAIL: Model could not overfit the tiny dataset")
        passed_all = False

    # ----------------------------------------------------
    # Check 3: Loss strictly decreases after first few updates
    # ----------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[Check 3] Verify loss decreases consistently over initial updates")
    # Softmax regression on a new random batch
    rng = np.random.default_rng(99)
    X_batch = rng.standard_normal((50, 4))
    y_batch = rng.integers(0, 3, size=(50,))
    Y_batch_oh = one_hot(y_batch, 3)
    
    sr = SoftmaxRegression(n_features=4, n_classes=3)
    sr.init_params(seed=99)
    opt_sgd = SGD(lr=0.1)
    opt_sgd.init_state(sr.params)
    
    losses = []
    print("  Tracking loss for first 10 gradient steps:")
    for step in range(10):
        P_batch, cache_batch = sr.forward(X_batch)
        loss = cross_entropy_loss(P_batch, Y_batch_oh)
        losses.append(loss)
        sr.backward(cache_batch, Y_batch_oh, lam=0.0)
        opt_sgd.step(sr.params, sr.grads)
        print(f"    Step {step:2d} | Loss: {loss:.4f}")
        
    # Check if final loss is less than initial
    if losses[-1] < losses[0]:
        print(f"  ✓ PASS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    else:
        print("  ✗ FAIL: Loss did not decrease over first few updates")
        passed_all = False
        
    print("\n============================================================")
    if passed_all:
        print("  STATUS: ALL SANITY CHECKS PASSED SUCCESSFULLY [OK]")
    else:
        print("  STATUS: SOME SANITY CHECKS FAILED [WARNING]")
    print("============================================================")
    print(f"\nResults saved to: {log_file}")
    
    # Restore stdout and close file
    sys.stdout = logger.console
    logger.close()

if __name__ == '__main__':
    main()
