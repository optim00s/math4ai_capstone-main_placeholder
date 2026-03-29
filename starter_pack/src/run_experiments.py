#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main experiment script for Math4AI Capstone - Track A.

Runs ALL required experiments:
  1. Core comparisons on linear Gaussian, moons, digits
  2. Capacity ablation on moons ({2, 8, 32})
  3. Optimizer study on digits (SGD, Momentum, Adam)
  4. Repeated-seed evaluation (5 seeds, 95% CI)
  5. Track A: PCA/SVD analysis
  6. Failure-case analysis
  7. Implementation sanity checks


Usage:
  python -m starter_pack.src.run_experiments
  (from the repository root)
"""

import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from starter_pack.src.models import (
    SoftmaxRegression, NeuralNetwork,
    cross_entropy_loss, one_hot, stable_softmax
)
from starter_pack.src.optimizers import SGD, Momentum, Adam
from starter_pack.src.train import train_model, compute_accuracy, compute_loss
from starter_pack.src.data_utils import load_synthetic, load_digits
from starter_pack.src.plotting import (
    plot_decision_boundary, plot_decision_boundary_comparison,
    plot_capacity_ablation_boundaries,
    plot_loss_curves, plot_capacity_ablation,
    plot_optimizer_comparison,
    plot_pca_scree, plot_pca_2d, plot_pca_softmax_comparison,
    get_figures_dir
)

# DRY Principle: Import the standalone sanity check logic
from starter_pack.src.sanity_checks import main as run_standalone_sanity_checks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = REPO_ROOT / "starter_pack" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
#  SANITY CHECKS 
# ============================================================

def run_sanity_checks():
    """Run the 5 implementation sanity checks and save results to file.

    This function delegates the actual execution to the standalone 
    sanity_checks.py script to enforce the DRY (Don't Repeat Yourself) principle.
    """
    print("=" * 62)
    print("SANITY CHECKS")
    print("=" * 62)
    print("Delegating to standalone sanity_checks.py...\n")
    
    # Run the main function from sanity_checks.py
    run_standalone_sanity_checks()


# ============================================================
#  EXPERIMENT 1: SYNTHETIC TASKS
# ============================================================

def run_synthetic_experiments():
    """Train and compare both models on linear Gaussian and moons.
    Saves a text summary to results/synthetic_results.txt.
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 1: SYNTHETIC TASKS")
    print("=" * 62)

    summary_lines = [
        "Synthetic Task Results",
        "=" * 50,
    ]

    for dataset_name in ['linear_gaussian', 'moons']:
        print(f"\n{'='*50}")
        print(f"  Dataset: {dataset_name}")
        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic(dataset_name)
        n_classes = len(np.unique(y_train))
        n_features = X_train.shape[1]
        n_train = len(y_train)

        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

        # Softmax Regression (SGD, lr=0.1, full-batch)
        print("\n  [Softmax Regression]")
        sr = SoftmaxRegression(n_features, n_classes)
        sr.init_params(seed=42)
        opt_sr = SGD(lr=0.1)
        hist_sr, best_sr, ep_sr = train_model(
            sr, opt_sr, X_train, y_train, X_val, y_val,
            n_classes=n_classes, n_epochs=300, batch_size=n_train,
            lam=1e-5, seed=42
        )
        sr.set_params(best_sr)
        test_acc_sr = compute_accuracy(sr, X_test, y_test)
        test_loss_sr = compute_loss(sr, X_test, y_test, n_classes)
        print(f"  Test Acc: {test_acc_sr:.4f}, Test CE: {test_loss_sr:.4f}")

        # Neural Network (h=32, Adam, lr=0.01, full-batch)
        print("\n  [Neural Network, h=32, Adam]")
        nn = NeuralNetwork(n_features, 32, n_classes)
        nn.init_params(seed=42)
        opt_nn = Adam(lr=0.01)
        hist_nn, best_nn, ep_nn = train_model(
            nn, opt_nn, X_train, y_train, X_val, y_val,
            n_classes=n_classes, n_epochs=300, batch_size=n_train,
            lam=1e-5, seed=42
        )
        nn.set_params(best_nn)
        test_acc_nn = compute_accuracy(nn, X_test, y_test)
        test_loss_nn = compute_loss(nn, X_test, y_test, n_classes)
        print(f"  Test Acc: {test_acc_nn:.4f}, Test CE: {test_loss_nn:.4f}")

        summary_lines += [
            f"\nDataset: {dataset_name}",
            f"  Softmax | Test Acc: {test_acc_sr:.4f} | Test CE: {test_loss_sr:.4f}",
            f"  NN h=32 | Test Acc: {test_acc_nn:.4f} | Test CE: {test_loss_nn:.4f}",
        ]

        tag = dataset_name.replace('_', '')

        plot_decision_boundary(
            sr, X_all, y_all,
            title=f"Softmax Regression - {dataset_name}",
            filename=f"decision_boundary_{tag}_softmax.png"
        )
        plot_decision_boundary(
            nn, X_all, y_all,
            title=f"Neural Network (h=32) - {dataset_name}",
            filename=f"decision_boundary_{tag}_nn.png"
        )
        plot_decision_boundary_comparison(
            [sr, nn], X_all, y_all,
            titles=['Softmax Regression (Linear)', 'Neural Network (h=32, tanh)'],
            filename=f"comparison_{tag}.png"
        )
        plot_loss_curves(
            [hist_sr, hist_nn],
            ['Softmax', 'NN (h=32)'],
            title=f"Training Dynamics - {dataset_name}",
            filename=f"loss_curves_{tag}.png"
        )

    with open(RESULTS_DIR / "synthetic_results.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\n  Saved: results/synthetic_results.txt")


# ============================================================
#  EXPERIMENT 2: DIGITS BENCHMARK
# ============================================================

def run_digits_experiment():
    """Train and compare both models on the fixed digits benchmark.

    FIX: The NN now uses Adam (lr=0.001) — the final selected configuration
    based on validation evidence (lowest CE in optimizer study).  This ensures
    loss_curves_digits.png is consistent with the five-seed summary table, which
    also uses Adam for the NN.
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 2: DIGITS BENCHMARK")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Softmax Regression (SGD, lr=0.05 — protocol default)
    print("\n  [Softmax Regression — SGD lr=0.05]")
    sr = SoftmaxRegression(n_features, n_classes)
    sr.init_params(seed=42)
    opt_sr = SGD(lr=0.05)
    hist_sr, best_sr, ep_sr = train_model(
        sr, opt_sr, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42, verbose=True
    )
    sr.set_params(best_sr)
    test_acc_sr = compute_accuracy(sr, X_test, y_test)
    test_loss_sr = compute_loss(sr, X_test, y_test, n_classes)
    print(f"  Test Acc: {test_acc_sr:.4f}, Test CE: {test_loss_sr:.4f}, Best Epoch: {ep_sr}")

    # Neural Network — final selected config: Adam lr=0.001
    # Justification: Adam achieves the lowest validation CE in the optimizer study,
    # making it the appropriate final config per Section 7.4 protocol.
    print("\n  [Neural Network h=32 — Adam lr=0.001 (final selected config)]")
    nn = NeuralNetwork(n_features, 32, n_classes)
    nn.init_params(seed=42)
    opt_nn = Adam(lr=0.001)
    hist_nn, best_nn, ep_nn = train_model(
        nn, opt_nn, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42, verbose=True
    )
    nn.set_params(best_nn)
    test_acc_nn = compute_accuracy(nn, X_test, y_test)
    test_loss_nn = compute_loss(nn, X_test, y_test, n_classes)
    print(f"  Test Acc: {test_acc_nn:.4f}, Test CE: {test_loss_nn:.4f}, Best Epoch: {ep_nn}")

    # loss_curves_digits.png now shows the final-config dynamics (Adam for NN)
    plot_loss_curves(
        [hist_sr, hist_nn],
        ['Softmax (SGD)', 'NN h=32 (Adam)'],
        title="Training Dynamics — Digits Benchmark (final selected configs, seed=42)",
        filename="loss_curves_digits.png"
    )

    return hist_sr, hist_nn


# ============================================================
#  EXPERIMENT 3: CAPACITY ABLATION (MOONS)
# ============================================================

def run_capacity_ablation():
    """Capacity ablation on moons with hidden widths {2, 8, 32}."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 3: CAPACITY ABLATION (MOONS)")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic('moons')
    n_classes = 2
    n_features = 2
    n_train = len(y_train)

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    histories = {}
    trained_models = {}

    for h_width in [2, 8, 32]:
        print(f"\n  [NN, h={h_width}, Adam lr=0.01]")
        nn = NeuralNetwork(n_features, h_width, n_classes)
        nn.init_params(seed=42)
        opt = Adam(lr=0.01)
        hist, best_p, best_ep = train_model(
            nn, opt, X_train, y_train, X_val, y_val,
            n_classes=n_classes, n_epochs=500, batch_size=n_train,
            lam=1e-5, seed=42
        )
        nn.set_params(best_p)
        test_acc = compute_accuracy(nn, X_test, y_test)
        print(f"  Test Acc: {test_acc:.4f}, Best Epoch: {best_ep}")
        histories[str(h_width)] = hist
        trained_models[h_width] = nn

        plot_decision_boundary(
            nn, X_all, y_all,
            title=f"NN (h={h_width}) - Moons",
            filename=f"decision_boundary_moons_h{h_width}.png"
        )

    plot_capacity_ablation_boundaries(
        trained_models, X_all, y_all,
        filename="capacity_ablation_boundaries.png"
    )
    plot_capacity_ablation(histories, filename="capacity_ablation_curves.png")


# ============================================================
#  EXPERIMENT 4: OPTIMIZER STUDY (DIGITS)
# ============================================================

def run_optimizer_study():
    """Optimizer study on digits: SGD, Momentum, Adam."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 4: OPTIMIZER STUDY (DIGITS)")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]

    optimizers_config = {
        'SGD':      SGD(lr=0.05),
        'Momentum': Momentum(lr=0.05, mu=0.9),
        'Adam':     Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8),
    }

    histories = {}
    results = {}
    for name, opt in optimizers_config.items():
        print(f"\n  [{name}]")
        nn = NeuralNetwork(n_features, 32, n_classes)
        nn.init_params(seed=42)
        hist, best_p, best_ep = train_model(
            nn, opt, X_train, y_train, X_val, y_val,
            n_classes=n_classes, n_epochs=200, batch_size=64,
            lam=1e-4, seed=42
        )
        nn.set_params(best_p)
        test_acc = compute_accuracy(nn, X_test, y_test)
        test_loss = compute_loss(nn, X_test, y_test, n_classes)
        print(f"  Test Acc: {test_acc:.4f}, Test CE: {test_loss:.4f}, Best Epoch: {best_ep}")
        histories[name] = hist
        results[name] = {'acc': test_acc, 'loss': test_loss, 'best_epoch': best_ep}

    plot_optimizer_comparison(histories, filename="optimizer_study_digits.png")

    with open(RESULTS_DIR / "optimizer_study.txt", "w") as f:
        f.write("Optimizer Study Results (NN h=32, Digits)\n")
        f.write("=" * 50 + "\n")
        for name, r in results.items():
            f.write(f"{name:10s} | Acc: {r['acc']:.4f} | Loss: {r['loss']:.4f} | Best Epoch: {r['best_epoch']}\n")

    return results


# ============================================================
#  EXPERIMENT 5: REPEATED-SEED EVALUATION
# ============================================================

def run_repeated_seed():
    """5-seed evaluation on digits for both final configurations.

    Final selected configurations (chosen by lowest validation CE):
      - Softmax Regression: SGD, lr=0.05 (protocol default; no better option needed)
      - Neural Network h=32: Adam, lr=0.001 (chosen from optimizer study —
        achieves lowest validation CE among SGD, Momentum, Adam; see Section 7.3)
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 5: REPEATED-SEED EVALUATION (DIGITS)")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]
    seeds = [0, 1, 2, 3, 4]

    results = {}
    for model_name in ['Softmax', 'NN']:
        accs, ces = [], []
        for s in seeds:
            if model_name == 'Softmax':
                model = SoftmaxRegression(n_features, n_classes)
                model.init_params(seed=s)
                opt = SGD(lr=0.05)
            else:
                # Final selected config: Adam (lowest val CE in optimizer study)
                model = NeuralNetwork(n_features, 32, n_classes)
                model.init_params(seed=s)
                opt = Adam(lr=0.001)

            _, best_p, _ = train_model(
                model, opt, X_train, y_train, X_val, y_val,
                n_classes=n_classes, n_epochs=200, batch_size=64,
                lam=1e-4, seed=s, verbose=False
            )
            model.set_params(best_p)
            acc = compute_accuracy(model, X_test, y_test)
            loss = compute_loss(model, X_test, y_test, n_classes)
            accs.append(acc)
            ces.append(loss)
            print(f"  {model_name} seed={s}: Acc={acc:.4f}, CE={loss:.4f}")

        accs = np.array(accs)
        ces = np.array(ces)
        t_crit = 2.776  # t_{0.025,4}

        mean_acc = accs.mean()
        ci_acc = t_crit * accs.std(ddof=1) / np.sqrt(5)
        mean_ce = ces.mean()
        ci_ce = t_crit * ces.std(ddof=1) / np.sqrt(5)

        results[model_name] = {
            'mean_acc': mean_acc, 'ci_acc': ci_acc,
            'mean_ce': mean_ce, 'ci_ce': ci_ce,
            'accs': accs, 'ces': ces,
        }
        print(f"\n  {model_name}:")
        print(f"    Acc: {mean_acc:.4f} +/- {ci_acc:.4f}")
        print(f"    CE:  {mean_ce:.4f} +/- {ci_ce:.4f}")

    with open(RESULTS_DIR / "repeated_seed.txt", "w") as f:
        f.write("Repeated-Seed Evaluation Results (5 seeds, Digits)\n")
        f.write("=" * 60 + "\n")
        f.write("95% CI: mean +/- 2.776 * s / sqrt(5)\n\n")
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Test Accuracy: {r['mean_acc']:.4f} +/- {r['ci_acc']:.4f}\n")
            f.write(f"  Test CE Loss:  {r['mean_ce']:.4f} +/- {r['ci_ce']:.4f}\n")
            f.write(f"  Raw accs: {r['accs']}\n")
            f.write(f"  Raw CEs:  {r['ces']}\n\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    names = list(results.keys())
    x = np.arange(len(names))

    means_acc = [results[n]['mean_acc'] for n in names]
    cis_acc   = [results[n]['ci_acc'] for n in names]
    bars1 = axes[0].bar(x, means_acc, yerr=cis_acc, color=['#3498DB', '#E74C3C'],
                edgecolor='white', capsize=12, alpha=0.85, width=0.5,
                error_kw={'linewidth': 2, 'capthick': 2})
    for bar, val, ci in zip(bars1, means_acc, cis_acc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=12)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy (5 seeds, 95% CI)', fontweight='bold')

    means_ce = [results[n]['mean_ce'] for n in names]
    cis_ce   = [results[n]['ci_ce'] for n in names]
    bars2 = axes[1].bar(x, means_ce, yerr=cis_ce, color=['#3498DB', '#E74C3C'],
                edgecolor='white', capsize=12, alpha=0.85, width=0.5,
                error_kw={'linewidth': 2, 'capthick': 2})
    for bar, val, ci in zip(bars2, means_ce, cis_ce):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=12)
    axes[1].set_ylabel('Test Cross-Entropy')
    axes[1].set_title('Test CE Loss (5 seeds, 95% CI)', fontweight='bold')

    fig.suptitle('Repeated-Seed Evaluation — Digits',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(get_figures_dir() / "repeated_seed_digits.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: repeated_seed_digits.png")

    return results


# ============================================================
#  EXPERIMENT 6: TRACK A — PCA/SVD ANALYSIS
# ============================================================

def run_track_a_pca():
    """Track A: PCA/SVD and input geometry analysis."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 6: TRACK A — PCA/SVD ANALYSIS")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10

    X_full = np.vstack([X_train, X_val, X_test])
    y_full = np.concatenate([y_train, y_val, y_test])

    mean = X_train.mean(axis=0)
    X_train_c = X_train - mean
    X_val_c   = X_val   - mean
    X_test_c  = X_test  - mean
    X_full_c  = X_full  - mean

    U, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
    total_var = np.sum(S ** 2)
    explained_variance_ratio = (S ** 2) / total_var

    print("\n  [Scree Plot]")
    plot_pca_scree(explained_variance_ratio, filename="pca_scree_digits.png")
    cumvar = np.cumsum(explained_variance_ratio)
    for k in [5, 10, 20, 40]:
        print(f"    Top {k:2d} PCs: {cumvar[k-1]*100:.1f}%")

    print("\n  [2D PCA Visualization]")
    X_2d = X_full_c @ Vt[:2].T
    plot_pca_2d(X_2d, y_full, filename="pca_2d_digits.png")

    print("\n  [Softmax at PCA dimensions]")
    pca_dims = [10, 20, 40, 64]
    val_accs, val_losses = [], []

    for m in pca_dims:
        if m < X_train_c.shape[1]:
            Vm = Vt[:m]
            X_tr_pca = X_train_c @ Vm.T
            X_va_pca = X_val_c   @ Vm.T
        else:
            X_tr_pca = X_train_c
            X_va_pca = X_val_c

        sr_pca = SoftmaxRegression(X_tr_pca.shape[1], n_classes)
        sr_pca.init_params(seed=42)
        opt_pca = SGD(lr=0.05)
        _, best_p, _ = train_model(
            sr_pca, opt_pca, X_tr_pca, y_train, X_va_pca, y_val,
            n_classes=n_classes, n_epochs=200, batch_size=64,
            lam=1e-4, seed=42, verbose=False
        )
        sr_pca.set_params(best_p)
        v_acc  = compute_accuracy(sr_pca, X_va_pca, y_val)
        v_loss = compute_loss(sr_pca, X_va_pca, y_val, n_classes)
        val_accs.append(v_acc)
        val_losses.append(v_loss)
        print(f"    m={m:3d} | Val Acc: {v_acc:.4f} | Val CE: {v_loss:.4f}")

    plot_pca_softmax_comparison(pca_dims, val_accs, val_losses,
                                filename="pca_softmax_comparison.png")

    with open(RESULTS_DIR / "track_a_pca.txt", "w") as f:
        f.write("Track A: PCA/SVD Analysis Results\n")
        f.write("=" * 50 + "\n")
        f.write("\nCumulative Explained Variance:\n")
        for k in [5, 10, 20, 40, 64]:
            idx = min(k, len(cumvar)) - 1
            f.write(f"  Top {k:2d} PCs: {cumvar[idx]*100:.1f}%\n")
        f.write("\nSoftmax at PCA dimensions:\n")
        for m, va, vl in zip(pca_dims, val_accs, val_losses):
            f.write(f"  m={m:3d}: Val Acc={va:.4f}, Val CE={vl:.4f}\n")


# ============================================================
#  EXPERIMENT 7: FAILURE-CASE ANALYSIS
# ============================================================

def run_failure_analysis():
    """Failure-case analysis: under-capacity NN (h=2) on digits."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 7: FAILURE-CASE ANALYSIS")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]

    print("\n  [Failure: NN h=2 on 10-class digits]")
    nn_fail = NeuralNetwork(n_features, 2, n_classes)
    nn_fail.init_params(seed=42)
    opt_fail = SGD(lr=0.05)
    hist_fail, best_p_fail, _ = train_model(
        nn_fail, opt_fail, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42
    )
    nn_fail.set_params(best_p_fail)
    test_acc_fail  = compute_accuracy(nn_fail, X_test, y_test)
    test_loss_fail = compute_loss(nn_fail, X_test, y_test, n_classes)
    print(f"  Test Acc: {test_acc_fail:.4f}, Test CE: {test_loss_fail:.4f}")

    # Good model for comparison (Adam, h=32)
    nn_good = NeuralNetwork(n_features, 32, n_classes)
    nn_good.init_params(seed=42)
    opt_good = Adam(lr=0.001)
    hist_good, best_p_good, _ = train_model(
        nn_good, opt_good, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42, verbose=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = range(1, len(hist_fail['train_loss']) + 1)

    axes[0].plot(epochs, hist_fail['train_loss'], '-', color='#E74C3C',
                 label='h=2 SGD (train)', alpha=0.5, linewidth=1)
    axes[0].plot(epochs, hist_fail['val_loss'], '-', color='#E74C3C',
                 label='h=2 SGD (val)', linewidth=2.5)
    axes[0].plot(epochs, hist_good['train_loss'], '-', color='#27AE60',
                 label='h=32 Adam (train)', alpha=0.5, linewidth=1)
    axes[0].plot(epochs, hist_good['val_loss'], '-', color='#27AE60',
                 label='h=32 Adam (val)', linewidth=2.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss: Under-capacity (h=2) vs Adequate (h=32)', fontweight='bold')
    axes[0].legend(fontsize=9, framealpha=0.9)

    axes[1].plot(epochs, hist_fail['val_acc'], '-', color='#E74C3C',
                 label='h=2 SGD', linewidth=2.5)
    axes[1].plot(epochs, hist_good['val_acc'], '-', color='#27AE60',
                 label='h=32 Adam', linewidth=2.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Accuracy: Under-capacity vs Adequate', fontweight='bold')
    axes[1].legend(fontsize=10, framealpha=0.9)

    fig.suptitle('Failure Case: Under-Capacity NN (h=2) on 10-Class Digits',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(get_figures_dir() / "failure_analysis.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: failure_analysis.png")

    with open(RESULTS_DIR / "failure_analysis.txt", "w") as f:
        f.write("Failure-Case Analysis: Under-Capacity NN (h=2) on 10-Class Digits\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"NN h=2  | Test Acc: {test_acc_fail:.4f} | Test CE: {test_loss_fail:.4f}\n")
        f.write("\nAnalysis:\n")
        f.write("With only 2 hidden units, the network has a 2-dimensional bottleneck.\n")
        f.write("The hidden representation h = tanh(W1 x + b1) maps the 64-dim input\n")
        f.write("to only 2 dimensions, which is insufficient to separate 10 classes.\n")
        f.write("The capacity is far below what is needed for a 10-class problem.\n")
        f.write("The model underfits: both training and validation loss remain high,\n")
        f.write("and accuracy plateaus far below what a wider network achieves.\n")
        f.write("This demonstrates that model complexity must match task complexity.\n")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 62)
    print("  Math4AI Capstone — Track A: Full Experiment Suite")
    print("=" * 62)

    run_sanity_checks()
    run_synthetic_experiments()
    run_digits_experiment()
    run_capacity_ablation()
    run_optimizer_study()
    run_repeated_seed()
    run_track_a_pca()
    run_failure_analysis()

    print("\n" + "=" * 62)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Figures: {get_figures_dir()}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()