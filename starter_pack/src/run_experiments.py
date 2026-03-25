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
  6. Implementation sanity checks

Usage:
  python -m starter_pack.src.run_experiments
  (from the repository root)
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

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

RESULTS_DIR = REPO_ROOT / "starter_pack" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
#  SANITY CHECKS
# ============================================================

def run_sanity_checks():
    """Implementation sanity checks as required by the rubric."""
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # 1. Softmax probabilities sum to 1
    print("\n[Check 1] Softmax probabilities sum to 1")
    Z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [100.0, 0.0, -100.0]])
    P = stable_softmax(Z)
    sums = P.sum(axis=1)
    print(f"  Row sums: {sums}")
    assert np.allclose(sums, 1.0), "FAIL: softmax rows don't sum to 1"
    print("  [PASS]")

    # 2. Loss decreases on a tiny subset
    print("\n[Check 2] Loss decreases on tiny subset")
    rng = np.random.default_rng(0)
    X_tiny = rng.standard_normal((10, 2))
    y_tiny = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model = SoftmaxRegression(2, 2)
    model.init_params(seed=0)
    opt = SGD(lr=0.5)
    opt.init_state(model.params)
    losses = []
    for _ in range(50):
        P, cache = model.forward(X_tiny)
        Y_oh = one_hot(y_tiny, 2)
        loss = cross_entropy_loss(P, Y_oh)
        losses.append(loss)
        model.backward(cache, Y_oh)
        opt.step(model.params, model.grads)
    print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "FAIL: loss did not decrease"
    print("  [PASS]")

    # 3. Gradient sanity check (numerical vs analytical)
    print("\n[Check 3] Gradient check (numerical vs analytical)")
    model_check = NeuralNetwork(2, 4, 2)
    model_check.init_params(seed=42)
    rng2 = np.random.default_rng(1)
    X_g = rng2.standard_normal((5, 2))
    y_g = np.array([0, 1, 0, 1, 0])
    Y_oh_g = one_hot(y_g, 2)

    P_g, cache_g = model_check.forward(X_g)
    model_check.backward(cache_g, Y_oh_g)

    eps = 1e-5
    for param_key in ['W1', 'W2', 'b1', 'b2']:
        grad_analytical = model_check.grads[param_key].copy()
        rows = min(3, grad_analytical.shape[0])
        cols = min(3, grad_analytical.shape[1]) if grad_analytical.ndim > 1 else 1

        max_diff = 0.0
        for i in range(rows):
            jrange = range(cols) if grad_analytical.ndim > 1 else [0]
            for j in jrange:
                idx = (i, j) if grad_analytical.ndim > 1 else (0, i)
                orig = model_check.params[param_key][idx]

                model_check.params[param_key][idx] = orig + eps
                P_plus, _ = model_check.forward(X_g)
                loss_plus = cross_entropy_loss(P_plus, Y_oh_g)

                model_check.params[param_key][idx] = orig - eps
                P_minus, _ = model_check.forward(X_g)
                loss_minus = cross_entropy_loss(P_minus, Y_oh_g)

                model_check.params[param_key][idx] = orig
                grad_num = (loss_plus - loss_minus) / (2 * eps)
                grad_ana = grad_analytical[idx]
                max_diff = max(max_diff, abs(grad_ana - grad_num))

        print(f"  {param_key}: max |analytical - numerical| = {max_diff:.2e}")
        assert max_diff < 1e-4, f"FAIL: gradient check for {param_key}"

    print("  [PASS]")

    # 4. No NaN/Inf check
    print("\n[Check 4] No NaN/Inf in parameters after training")
    model_nan = NeuralNetwork(2, 8, 2)
    model_nan.init_params(seed=7)
    opt_nan = SGD(lr=0.05)
    _, _, _ = train_model(model_nan, opt_nan, X_tiny, y_tiny, X_tiny, y_tiny,
                          n_classes=2, n_epochs=20, batch_size=5, lam=0,
                          seed=7, verbose=False)
    has_nan = any(np.any(np.isnan(v)) or np.any(np.isinf(v))
                  for v in model_nan.params.values())
    print(f"  NaN/Inf detected: {has_nan}")
    assert not has_nan, "FAIL: NaN/Inf in parameters"
    print("  [PASS]")

    # 5. Overfit tiny subset perfectly
    print("\n[Check 5] Can overfit tiny subset to near-zero loss")
    model_of = NeuralNetwork(2, 16, 2)
    model_of.init_params(seed=0)
    opt_of = Adam(lr=0.01)
    hist_of, _, _ = train_model(model_of, opt_of, X_tiny, y_tiny, X_tiny, y_tiny,
                                n_classes=2, n_epochs=200, batch_size=10, lam=0,
                                seed=0, verbose=False)
    final_acc = hist_of['train_acc'][-1]
    print(f"  Final train accuracy on 10-sample subset: {final_acc:.4f}")
    assert final_acc >= 0.99, "FAIL: could not overfit tiny subset"
    print("  [PASS]")

    print("\n[OK] All 5 sanity checks passed!\n")


# ============================================================
#  EXPERIMENT 1: SYNTHETIC TASKS
# ============================================================

def run_synthetic_experiments():
    """Train and compare both models on linear Gaussian and moons.

    Key design decisions for synthetic tasks:
    - Use Adam for NN to allow it to learn nonlinear features
    - Use lower regularization since datasets are small
    - Use full-batch training (batch_size = len(train)) for stability
    - Train longer to ensure convergence
    """
    print("=" * 60)
    print("EXPERIMENT 1: SYNTHETIC TASKS")
    print("=" * 60)

    for dataset_name in ['linear_gaussian', 'moons']:
        print(f"\n{'='*50}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*50}")
        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic(dataset_name)
        n_classes = len(np.unique(y_train))
        n_features = X_train.shape[1]
        n_train = len(y_train)

        # Full dataset for plotting
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

        # --- Softmax Regression (SGD, lr=0.1) ---
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
        print(f"  Test Acc: {test_acc_sr:.4f}, Test Loss: {test_loss_sr:.4f}")

        # --- Neural Network (h=32, Adam, lr=0.01) ---
        # Adam allows the NN to fully explore nonlinear feature space
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
        print(f"  Test Acc: {test_acc_nn:.4f}, Test Loss: {test_loss_nn:.4f}")

        # --- Plots ---
        tag = dataset_name.replace('_', '')

        # Individual decision boundary plots
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

        # Side-by-side comparison (most impactful)
        plot_decision_boundary_comparison(
            [sr, nn], X_all, y_all,
            titles=['Softmax Regression (Linear)', 'Neural Network (h=32, tanh)'],
            filename=f"comparison_{tag}.png"
        )

        # Training dynamics
        plot_loss_curves(
            [hist_sr, hist_nn],
            ['Softmax', 'NN (h=32)'],
            title=f"Training Dynamics - {dataset_name}",
            filename=f"loss_curves_{tag}.png"
        )


# ============================================================
#  EXPERIMENT 2: DIGITS BENCHMARK
# ============================================================

def run_digits_experiment():
    """Train and compare both models on the fixed digits benchmark."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: DIGITS BENCHMARK")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Softmax Regression (SGD, lr=0.05, as per protocol)
    print("\n  [Softmax Regression]")
    sr = SoftmaxRegression(n_features, n_classes)
    sr.init_params(seed=42)
    opt_sr = SGD(lr=0.05)
    hist_sr, best_sr, ep_sr = train_model(
        sr, opt_sr, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42
    )
    sr.set_params(best_sr)
    test_acc_sr = compute_accuracy(sr, X_test, y_test)
    test_loss_sr = compute_loss(sr, X_test, y_test, n_classes)
    print(f"  Test Acc: {test_acc_sr:.4f}, Test Loss: {test_loss_sr:.4f}")

    # Neural Network (h=32, SGD, as per protocol)
    print("\n  [Neural Network, h=32, SGD]")
    nn = NeuralNetwork(n_features, 32, n_classes)
    nn.init_params(seed=42)
    opt_nn = SGD(lr=0.05)
    hist_nn, best_nn, ep_nn = train_model(
        nn, opt_nn, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42
    )
    nn.set_params(best_nn)
    test_acc_nn = compute_accuracy(nn, X_test, y_test)
    test_loss_nn = compute_loss(nn, X_test, y_test, n_classes)
    print(f"  Test Acc: {test_acc_nn:.4f}, Test Loss: {test_loss_nn:.4f}")

    plot_loss_curves(
        [hist_sr, hist_nn],
        ['Softmax', 'NN (h=32)'],
        title="Training Dynamics - Digits Benchmark",
        filename="loss_curves_digits.png"
    )

    return hist_sr, hist_nn


# ============================================================
#  EXPERIMENT 3: CAPACITY ABLATION (MOONS)
# ============================================================

def run_capacity_ablation():
    """Capacity ablation on moons with hidden widths {2, 8, 32}.

    Uses Adam to allow each capacity model to reach its best representation,
    so the comparison truly reflects capacity differences not optimizer limitations.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: CAPACITY ABLATION (MOONS)")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic('moons')
    n_classes = 2
    n_features = 2
    n_train = len(y_train)

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    histories = {}
    trained_models = {}

    for h_width in [2, 8, 32]:
        print(f"\n  [NN, h={h_width}, Adam]")
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

        # Individual plot for each width
        plot_decision_boundary(
            nn, X_all, y_all,
            title=f"NN (h={h_width}) - Moons",
            filename=f"decision_boundary_moons_h{h_width}.png"
        )

    # Side-by-side comparison of all three (the key figure!)
    plot_capacity_ablation_boundaries(
        trained_models, X_all, y_all,
        filename="capacity_ablation_boundaries.png"
    )

    # Loss/accuracy curves
    plot_capacity_ablation(histories, filename="capacity_ablation_curves.png")


# ============================================================
#  EXPERIMENT 4: OPTIMIZER STUDY (DIGITS)
# ============================================================

def run_optimizer_study():
    """Optimizer study on digits: SGD, Momentum, Adam."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: OPTIMIZER STUDY (DIGITS)")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]

    optimizers_config = {
        'SGD': SGD(lr=0.05),
        'Momentum': Momentum(lr=0.05, mu=0.9),
        'Adam': Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8),
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
        print(f"  Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, Best Epoch: {best_ep}")
        histories[name] = hist
        results[name] = {'acc': test_acc, 'loss': test_loss, 'best_epoch': best_ep}

    plot_optimizer_comparison(histories, filename="optimizer_study_digits.png")

    # Save results
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
    """5-seed evaluation on digits for both final configs."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: REPEATED-SEED EVALUATION (DIGITS)")
    print("=" * 60)

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
                model = NeuralNetwork(n_features, 32, n_classes)
                model.init_params(seed=s)
                opt = Adam(lr=0.001)  # best optimizer from study

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
        t_crit = 2.776  # t-distribution, df=4, 95% CI

        mean_acc = accs.mean()
        std_acc = accs.std(ddof=1)
        ci_acc = t_crit * std_acc / np.sqrt(5)

        mean_ce = ces.mean()
        std_ce = ces.std(ddof=1)
        ci_ce = t_crit * std_ce / np.sqrt(5)

        results[model_name] = {
            'mean_acc': mean_acc, 'ci_acc': ci_acc,
            'mean_ce': mean_ce, 'ci_ce': ci_ce,
            'accs': accs, 'ces': ces,
        }
        print(f"\n  {model_name}:")
        print(f"    Acc:  {mean_acc:.4f} +/- {ci_acc:.4f}")
        print(f"    CE:   {mean_ce:.4f} +/- {ci_ce:.4f}")

    # Save results
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

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    names = list(results.keys())
    x = np.arange(len(names))

    means_acc = [results[n]['mean_acc'] for n in names]
    cis_acc = [results[n]['ci_acc'] for n in names]
    bars1 = axes[0].bar(x, means_acc, yerr=cis_acc, color=['#3498DB', '#E74C3C'],
                edgecolor='white', capsize=12, alpha=0.85, width=0.5,
                error_kw={'linewidth': 2, 'capthick': 2})
    for i, (bar, val) in enumerate(zip(bars1, means_acc)):
        # Place text above the error bar cap
        y_pos = bar.get_height() + cis_acc[i] + 0.005
        axes[0].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=12)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy (5 seeds, 95% CI)', fontweight='bold')

    means_ce = [results[n]['mean_ce'] for n in names]
    cis_ce = [results[n]['ci_ce'] for n in names]
    bars2 = axes[1].bar(x, means_ce, yerr=cis_ce, color=['#3498DB', '#E74C3C'],
                edgecolor='white', capsize=12, alpha=0.85, width=0.5,
                error_kw={'linewidth': 2, 'capthick': 2})
    for i, (bar, val) in enumerate(zip(bars2, means_ce)):
        # Place text above the error bar cap
        y_pos = bar.get_height() + cis_ce[i] + 0.005
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=12)
    axes[1].set_ylabel('Test Cross-Entropy')
    axes[1].set_title('Test CE Loss (5 seeds, 95% CI)', fontweight='bold')

    fig.suptitle('Repeated-Seed Evaluation - Digits',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(get_figures_dir() / "repeated_seed_digits.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: repeated_seed_digits.png")

    return results


# ============================================================
#  EXPERIMENT 6: TRACK A - PCA/SVD ANALYSIS
# ============================================================

def run_track_a_pca():
    """Track A: PCA/SVD and input geometry analysis."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: TRACK A - PCA/SVD ANALYSIS")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10

    X_full = np.vstack([X_train, X_val, X_test])
    y_full = np.concatenate([y_train, y_val, y_test])

    # Center the data (using training mean)
    mean = X_train.mean(axis=0)
    X_train_c = X_train - mean
    X_val_c = X_val - mean
    X_test_c = X_test - mean
    X_full_c = X_full - mean

    # SVD
    U, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
    total_var = np.sum(S ** 2)
    explained_variance_ratio = (S ** 2) / total_var

    # 1. Scree plot
    print("\n  [Scree Plot]")
    plot_pca_scree(explained_variance_ratio, filename="pca_scree_digits.png")
    cumvar = np.cumsum(explained_variance_ratio)
    for k in [5, 10, 20, 40]:
        print(f"    Top {k:2d} PCs explain {cumvar[k-1]*100:.1f}% of variance")

    # 2. 2D PCA visualization
    print("\n  [2D PCA Visualization]")
    X_2d = X_full_c @ Vt[:2].T
    plot_pca_2d(X_2d, y_full, filename="pca_2d_digits.png")

    # 3. Softmax comparison at PCA dims {10, 20, 40} + full (64)
    print("\n  [Softmax at different PCA dimensions]")
    pca_dims = [10, 20, 40, 64]
    val_accs = []
    val_losses = []

    for m in pca_dims:
        if m < X_train_c.shape[1]:
            Vm = Vt[:m]
            X_tr_pca = X_train_c @ Vm.T
            X_va_pca = X_val_c @ Vm.T
            X_te_pca = X_test_c @ Vm.T
        else:
            X_tr_pca = X_train_c
            X_va_pca = X_val_c
            X_te_pca = X_test_c

        sr_pca = SoftmaxRegression(X_tr_pca.shape[1], n_classes)
        sr_pca.init_params(seed=42)
        opt_pca = SGD(lr=0.05)
        _, best_p, _ = train_model(
            sr_pca, opt_pca, X_tr_pca, y_train, X_va_pca, y_val,
            n_classes=n_classes, n_epochs=200, batch_size=64,
            lam=1e-4, seed=42, verbose=False
        )
        sr_pca.set_params(best_p)
        v_acc = compute_accuracy(sr_pca, X_va_pca, y_val)
        v_loss = compute_loss(sr_pca, X_va_pca, y_val, n_classes)
        t_acc = compute_accuracy(sr_pca, X_te_pca, y_test)
        t_loss = compute_loss(sr_pca, X_te_pca, y_test, n_classes)
        val_accs.append(v_acc)
        val_losses.append(v_loss)
        print(f"    PCA m={m:3d} | Val Acc: {v_acc:.4f} | Val CE: {v_loss:.4f} "
              f"| Test Acc: {t_acc:.4f} | Test CE: {t_loss:.4f}")

    plot_pca_softmax_comparison(pca_dims, val_accs, val_losses,
                                filename="pca_softmax_comparison.png")

    # Save results
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
    """Failure-case analysis: under-capacity NN on digits."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: FAILURE-CASE ANALYSIS")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10
    n_features = X_train.shape[1]

    # Failure case: very small hidden width (h=2) on 10-class digits
    print("\n  [Failure: NN with h=2 on 10-class digits]")
    nn_fail = NeuralNetwork(n_features, 2, n_classes)
    nn_fail.init_params(seed=42)
    opt_fail = SGD(lr=0.05)
    hist_fail, best_p_fail, ep_fail = train_model(
        nn_fail, opt_fail, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42
    )
    nn_fail.set_params(best_p_fail)
    test_acc_fail = compute_accuracy(nn_fail, X_test, y_test)
    test_loss_fail = compute_loss(nn_fail, X_test, y_test, n_classes)
    print(f"  Test Acc: {test_acc_fail:.4f}, Test Loss: {test_loss_fail:.4f}")

    # Compare with good NN (h=32, Adam)
    nn_good = NeuralNetwork(n_features, 32, n_classes)
    nn_good.init_params(seed=42)
    opt_good = Adam(lr=0.001)
    hist_good, best_p_good, _ = train_model(
        nn_good, opt_good, X_train, y_train, X_val, y_val,
        n_classes=n_classes, n_epochs=200, batch_size=64,
        lam=1e-4, seed=42, verbose=False
    )

    # Plot failure vs success
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

    # Save analysis
    with open(RESULTS_DIR / "failure_analysis.txt", "w") as f:
        f.write("Failure-Case Analysis: Under-Capacity NN (h=2) on 10-Class Digits\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"NN h=2  | Test Acc: {test_acc_fail:.4f} | Test CE: {test_loss_fail:.4f}\n")
        f.write(f"\nAnalysis:\n")
        f.write(f"With only 2 hidden units, the network has a 2-dimensional bottleneck.\n")
        f.write(f"The hidden representation h = tanh(W1 x + b1) maps the 64-dim input\n")
        f.write(f"to only 2 dimensions, which is insufficient to separate 10 classes.\n")
        f.write(f"The capacity is far below what is needed for a 10-class problem.\n")
        f.write(f"The model underfits: both training and validation loss remain high,\n")
        f.write(f"and accuracy plateaus far below what a wider network achieves.\n")
        f.write(f"This demonstrates that model complexity must match task complexity.\n")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Math4AI Capstone - Track A: Full Experiment Suite")
    print("=" * 60)
    print()

    run_sanity_checks()
    run_synthetic_experiments()
    run_digits_experiment()
    run_capacity_ablation()
    run_optimizer_study()
    run_repeated_seed()
    run_track_a_pca()
    run_failure_analysis()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Figures saved to: {get_figures_dir()}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
