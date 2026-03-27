# 📋 Math4AI Capstone — Detallı Git Commit Planı (Kod Blokları ilə)

> **Task Bölgüsü:** Sharaf → 1, 2, 5 | Samir → 3, 6 | Nicat → 4, 7

---

# 🔵 SHARAF — Fərdi Fayllar

## Commit S1: [__init__.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/__init__.py) — Package initialization

```bash
git add starter_pack/src/__init__.py
git commit -m "feat(src): initialize src as Python package"
```

```python
"""Math4AI Capstone source package."""
```

---

## Commit S2: [models.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/models.py) — Tasks 1 & 2: Softmax Regression + Neural Network

```bash
git add starter_pack/src/models.py
git commit -m "feat(models): implement SoftmaxRegression and NeuralNetwork with softmax, cross-entropy, one-hot utils"
```

### Utility functions (L1–L71):
```python
"""
Core model implementations for Math4AI Capstone.
Softmax Regression and One-Hidden-Layer Neural Network (tanh + softmax).
Only NumPy is used — no ML frameworks.
"""

import numpy as np

def stable_softmax(Z):
    """Row-wise numerically stable softmax."""
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

def cross_entropy_loss(P, Y_onehot):
    """Mean cross-entropy loss (negative log-likelihood)."""
    n = P.shape[0]
    eps = 1e-12
    log_probs = -np.log(np.clip(P, eps, 1.0))
    return np.sum(Y_onehot * log_probs) / n

def one_hot(y, k):
    """Convert integer labels to one-hot encoding."""
    n = len(y)
    Y = np.zeros((n, k))
    Y[np.arange(n), y.astype(int)] = 1.0
    return Y
```

### Task 1 — SoftmaxRegression class (L78–L158):
```python
class SoftmaxRegression:
    """Multiclass softmax (logistic) regression. s(x) = Wx + b"""
    def __init__(self, n_features, n_classes): ...
    def init_params(self, seed=None): ...      # Xavier init
    def forward(self, X): ...                   # S = XW^T + b → softmax
    def backward(self, cache, Y_onehot, lam=0.0): ...  # dW = (P-Y)^T X / n
    def predict(self, X): ...
    def get_params(self): ...
    def set_params(self, params): ...
```

### Task 2 — NeuralNetwork class (L165–L278):
```python
class NeuralNetwork:
    """One-hidden-layer NN: h=tanh(XW1^T+b1), s=hW2^T+b2, P=softmax(s)"""
    def __init__(self, n_features, n_hidden, n_classes): ...
    def init_params(self, seed=None): ...      # Xavier both layers
    def forward(self, X): ...                   # Z1→tanh→S→softmax
    def backward(self, cache, Y_onehot, lam=0.0): ...  # chain rule through tanh
    def predict(self, X): ...
    def get_params(self): ...
    def set_params(self, params): ...
```

---

## Commit S3: [data_utils.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/data_utils.py) — Data loading utilities

```bash
git add starter_pack/src/data_utils.py
git commit -m "feat(data): add synthetic and digits data loaders with mini-batch generator"
```

```python
"""Data loading and mini-batch utilities for Math4AI Capstone."""
import numpy as np
from pathlib import Path

def get_data_dir(): ...           # Path to starter_pack/data/
def load_synthetic(name): ...     # Loads linear_gaussian.npz or moons.npz
def load_digits(): ...            # Loads digits_data.npz + split indices
def mini_batches(X, y, batch_size, rng=None): ...  # Random mini-batch generator
```

---

# 🟢 SAMIR — Fərdi Fayllar

## Commit M1: [optimizers.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/optimizers.py) — Task 6: SGD, Momentum, Adam

> ⚠️ **Dependency:** [models.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/models.py) (Sharaf S2) repoda olmalıdır — optimizerlər `model.params` üzərində işləyir

```bash
git add starter_pack/src/optimizers.py
git commit -m "feat(optimizers): implement SGD, Momentum, and Adam optimizers from scratch"
```

### SGD (L9–L30):
```python
class SGD:
    """Vanilla mini-batch Stochastic Gradient Descent."""
    def __init__(self, lr=0.05): ...
    def init_state(self, params): ...   # No extra state
    def step(self, params, grads): ...  # θ -= lr * g
```

### Momentum (L33–L52):
```python
class Momentum:
    """SGD with momentum: v_t = μ*v_{t-1} + grad, θ -= lr*v_t"""
    def __init__(self, lr=0.05, mu=0.9): ...
    def init_state(self, params): ...   # Zero velocity buffers
    def step(self, params, grads): ...  # v update + param update
```

### Adam (L55–L89):
```python
class Adam:
    """Adam optimizer (Kingma & Ba, 2015) with bias correction."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8): ...
    def init_state(self, params): ...   # Zero m, v buffers
    def step(self, params, grads): ...  # m, v update → bias correct → param update
```

---

## Commit M2: [train.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/train.py) — Training loop (Shared, Samir owns)

> ⚠️ **Dependency:** [models.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/models.py) (S2) + [optimizers.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/optimizers.py) (M1) + [data_utils.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/data_utils.py) (S3)

```bash
git add starter_pack/src/train.py
git commit -m "feat(train): add mini-batch training loop with validation checkpointing"
```

```python
"""Training loop for Math4AI Capstone. Mini-batch training with validation checkpointing."""
from .models import cross_entropy_loss, one_hot
from .data_utils import mini_batches

def compute_accuracy(model, X, y): ...    # preds == y
def compute_loss(model, X, y, n_classes, lam=0.0): ...  # CE + L2 reg

def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                n_classes, n_epochs=200, batch_size=64, lam=1e-4,
                seed=42, verbose=True):
    """Mini-batch training with best-val-loss checkpointing.
    Returns: history dict, best_params, best_epoch"""
    # Epoch loop → mini-batch → forward → backward → optimizer.step
    # End-of-epoch: compute train/val loss+acc → checkpoint if val_loss improves
```

---

## Commit M3: [sanity_checks.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/sanity_checks.py) — Task 3: Rubric Sanity Checks

> ⚠️ **Dependency:** [models.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/models.py) (S2) + [optimizers.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/optimizers.py) (M1)

```bash
git add starter_pack/src/sanity_checks.py
git commit -m "feat(sanity): implement 5 automated sanity checks per rubric Section 5.5"
```

### Check 1 — Softmax sum = 1 (L72–L90):
```python
# [Check 1] Verify Softmax probabilities sum exactly to 1
Z = np.array([[1.0, 2.0, 3.0], [-100.0, 0.0, 100.0], [0.0, 0.0, 0.0]])
P = stable_softmax(Z)
sums = np.sum(P, axis=1)
assert np.allclose(sums, 1.0)  # ✓ PASS
```

### Check 2 — Overfit tiny dataset (L92–L132):
```python
# [Check 2] NN (h=16) overfit 5 samples → loss < 0.05, acc = 100%
nn_overfit = NeuralNetwork(n_features=2, n_hidden=16, n_classes=2)
opt_adam = Adam(lr=0.05)
# 150 epochs → final_loss < 0.05 and acc == 1.0
```

### Check 3 — Loss decreases (L134–L165):
```python
# [Check 3] SoftmaxRegression loss decreases over 10 SGD steps
sr = SoftmaxRegression(n_features=4, n_classes=3)
opt_sgd = SGD(lr=0.1)
# 10 steps → losses[-1] < losses[0]
```

---

# 🔴 NICAT — Fərdi Fayllar Yoxdur (hamısı shared [run_experiments.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py)-dadır)

Nicat-ın bütün işi [run_experiments.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py)-nin Track A + linear_gaussian hissəsindədir (aşağıda).

---

# 📊 SHARED: [plotting.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py) — Kod Bloklarının Bölgüsü

> ⚠️ **Dependency:** [models.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/models.py) (S2) — `model.predict()` istifadə edir

## Commit P1 (Sharaf): Base plotting + decision boundary funksiyaları

```bash
git add starter_pack/src/plotting.py
git commit -m "feat(plotting): add decision boundary, comparison, and loss curve visualizations"
```

### Global style + helpers (L1–L43):
```python
"""Professional plotting utilities for Math4AI Capstone."""
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({...})  # DPI, fonts, grid, spines
PROB_CMAP = plt.cm.RdYlBu
CLASS_COLORS = ['#E74C3C', '#3498DB']
MULTI_COLORS = [...]

def get_figures_dir(): ...  # starter_pack/figures/
```

### [plot_decision_boundary()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#46-118) — L46–L117 (Sharaf — Task 5):
```python
def plot_decision_boundary(model, X, y, title, filename, resolution=300):
    """2D decision boundary with probability heatmap.
    Binary: P(class=1) heatmap + contour at p=0.5
    Multiclass: class regions + colored points"""
```

### [plot_decision_boundary_comparison()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#120-176) — L120–L175 (Sharaf — Task 5):
```python
def plot_decision_boundary_comparison(models, X, y, titles, filename, resolution=300):
    """Side-by-side comparison of 2+ models with shared axes/colorbar."""
```

### [plot_loss_curves()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#237-278) — L237–L277 (Sharaf — shared by all):
```python
def plot_loss_curves(histories, labels, title, filename):
    """Train/val loss and accuracy curves for multiple models."""
```

---

## Commit P2 (Sharaf): Capacity ablation plots — Task 5

```bash
git add starter_pack/src/plotting.py
git commit -m "feat(plotting): add capacity ablation boundary and curve plots for moons"
```

### [plot_capacity_ablation_boundaries()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#178-235) — L178–L234:
```python
def plot_capacity_ablation_boundaries(models_dict, X, y, filename, resolution=300):
    """Side-by-side h=2, h=8, h=32 decision boundaries on moons."""
```

### [plot_capacity_ablation()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#280-314) — L280–L313:
```python
def plot_capacity_ablation(histories_dict, title, filename):
    """Loss/acc curves colored by hidden width {2:'red', 8:'orange', 32:'green'}."""
```

---

## Commit P3 (Samir): Optimizer comparison plot — Task 6

```bash
git add starter_pack/src/plotting.py
git commit -m "feat(plotting): add optimizer comparison plot for digits study"
```

### [plot_optimizer_comparison()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#316-354) — L316–L353:
```python
def plot_optimizer_comparison(histories_dict, title, filename):
    """SGD vs Momentum vs Adam: loss/acc curves with color coding."""
    colors = {'SGD': '#E74C3C', 'Momentum': '#3498DB', 'Adam': '#27AE60'}
```

---

## Commit P4 (Nicat): PCA/SVD plots — Task 7

```bash
git add starter_pack/src/plotting.py
git commit -m "feat(plotting): add PCA scree, 2D projection, and softmax comparison plots for Track A"
```

### [plot_pca_scree()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#356-386) — L356–L385:
```python
def plot_pca_scree(explained_variance_ratio, filename):
    """Scree plot: individual bars + cumulative line, 90% threshold."""
```

### [plot_pca_2d()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#388-413) — L388–L412:
```python
def plot_pca_2d(X_2d, y, filename):
    """2D PCA scatter of digits data — each class colored differently."""
```

### [plot_pca_softmax_comparison()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py#415-455) — L415–L455:
```python
def plot_pca_softmax_comparison(dims, val_accs, val_losses, filename):
    """Bar chart: softmax accuracy/loss at PCA dims {10, 20, 40, 64}."""
```

---

# 🔬 SHARED: [run_experiments.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py) — Kod Blokları Bölgüsü

## Commit R0 (Sharaf): Imports + setup (L1–L56)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add base imports and experiment infrastructure"
```

```python
#!/usr/bin/env python3
"""Main experiment script for Math4AI Capstone - Track A."""
import sys, io, numpy as np
from pathlib import Path

# Fix Windows encoding
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from starter_pack.src.models import SoftmaxRegression, NeuralNetwork, ...
from starter_pack.src.optimizers import SGD, Momentum, Adam
from starter_pack.src.train import train_model, compute_accuracy, compute_loss
from starter_pack.src.data_utils import load_synthetic, load_digits
from starter_pack.src.plotting import (
    plot_decision_boundary, plot_decision_boundary_comparison,
    plot_capacity_ablation_boundaries, plot_loss_curves,
    plot_capacity_ablation, plot_optimizer_comparison,
    plot_pca_scree, plot_pca_2d, plot_pca_softmax_comparison,
    get_figures_dir
)
import matplotlib; matplotlib.use('Agg')

RESULTS_DIR = REPO_ROOT / "starter_pack" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
```

---

## Commit R1 (Samir): [run_sanity_checks()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#62-169) — Task 3 (L58–L168)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add automated sanity checks (softmax-sum, loss-decrease, gradient, NaN, overfit)"
```

```python
def run_sanity_checks():
    """Implementation sanity checks as required by the rubric."""
    # Check 1: Softmax probabilities sum to 1
    Z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [100.0, 0.0, -100.0]])
    P = stable_softmax(Z)
    assert np.allclose(P.sum(axis=1), 1.0)

    # Check 2: Loss decreases on tiny subset (50 SGD steps)
    # SoftmaxRegression(2, 2), SGD(lr=0.5), 10 samples
    assert losses[-1] < losses[0]

    # Check 3: Numerical vs analytical gradient check
    # NeuralNetwork(2, 4, 2), eps=1e-5 finite difference
    # For W1, W2, b1, b2: max |analytical - numerical| < 1e-4

    # Check 4: No NaN/Inf after training
    # NeuralNetwork(2, 8, 2), 20 epochs, check all params

    # Check 5: Overfit tiny subset → train_acc >= 0.99
    # NeuralNetwork(2, 16, 2), Adam(lr=0.01), 200 epochs
```

---

## Commit R2 (Sharaf + Nicat): [run_synthetic_experiments()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#175-261) — Tasks 4 & 5 (L171–L261)

> ⚠️ Bu funksiya həm `linear_gaussian` (Nicat), həm `moons` (Sharaf) işlədir — **birlikdə** commitlənməlidir

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add synthetic benchmarks (linear_gaussian + moons) with Softmax vs NN comparison"
```

```python
def run_synthetic_experiments():
    """Train and compare both models on linear Gaussian and moons."""
    for dataset_name in ['linear_gaussian', 'moons']:  # ← hər iki dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic(dataset_name)

        # Softmax Regression (SGD, lr=0.1, 300 epochs, full-batch)
        sr = SoftmaxRegression(n_features, n_classes)
        hist_sr, best_sr, ep_sr = train_model(sr, SGD(lr=0.1), ...)

        # Neural Network (h=32, Adam, lr=0.01, 300 epochs, full-batch)
        nn = NeuralNetwork(n_features, 32, n_classes)
        hist_nn, best_nn, ep_nn = train_model(nn, Adam(lr=0.01), ...)

        # Plots: decision boundaries (individual + comparison) + loss curves
        plot_decision_boundary(sr, ..., filename=f"decision_boundary_{tag}_softmax.png")
        plot_decision_boundary(nn, ..., filename=f"decision_boundary_{tag}_nn.png")
        plot_decision_boundary_comparison([sr, nn], ..., filename=f"comparison_{tag}.png")
        plot_loss_curves([hist_sr, hist_nn], ..., filename=f"loss_curves_{tag}.png")
```

---

## Commit R3 (Samir): [run_digits_experiment()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#267-317) — Task 6-nın base hissəsi (L263–L316)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add digits benchmark baseline (Softmax vs NN h=32)"
```

```python
def run_digits_experiment():
    """Train and compare both models on the fixed digits benchmark."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_classes = 10

    # Softmax Regression (SGD, lr=0.05, 200 epochs, batch=64)
    sr = SoftmaxRegression(n_features, n_classes)
    hist_sr, best_sr, ep_sr = train_model(sr, SGD(lr=0.05), ...)

    # Neural Network (h=32, SGD, lr=0.05, 200 epochs, batch=64)
    nn = NeuralNetwork(n_features, 32, n_classes)
    hist_nn, best_nn, ep_nn = train_model(nn, SGD(lr=0.05), ...)

    plot_loss_curves([hist_sr, hist_nn], ..., filename="loss_curves_digits.png")
```

---

## Commit R4 (Sharaf): [run_capacity_ablation()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#323-375) — Task 5 (L319–L375)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add capacity ablation on moons with hidden widths {2, 8, 32}"
```

```python
def run_capacity_ablation():
    """Capacity ablation on moons with hidden widths {2, 8, 32}."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic('moons')

    for h_width in [2, 8, 32]:
        nn = NeuralNetwork(n_features, h_width, n_classes)
        nn.init_params(seed=42)
        opt = Adam(lr=0.01)
        hist, best_p, best_ep = train_model(nn, opt, ..., n_epochs=500)
        # Individual decision boundary per width
        plot_decision_boundary(nn, ..., filename=f"decision_boundary_moons_h{h_width}.png")

    # Side-by-side h=2 vs h=8 vs h=32
    plot_capacity_ablation_boundaries(trained_models, ..., filename="capacity_ablation_boundaries.png")
    plot_capacity_ablation(histories, filename="capacity_ablation_curves.png")
```

---

## Commit R5 (Samir): [run_optimizer_study()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#381-425) — Task 6 (L377–L424)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add optimizer study (SGD/Momentum/Adam) on digits with h=32 NN"
```

```python
def run_optimizer_study():
    """Optimizer study on digits: SGD, Momentum, Adam."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

    optimizers_config = {
        'SGD':      SGD(lr=0.05),
        'Momentum': Momentum(lr=0.05, mu=0.9),
        'Adam':     Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8),
    }

    for name, opt in optimizers_config.items():
        nn = NeuralNetwork(n_features, 32, n_classes)
        hist, best_p, best_ep = train_model(nn, opt, ..., n_epochs=200, batch_size=64)

    plot_optimizer_comparison(histories, filename="optimizer_study_digits.png")
    # Save results → results/optimizer_study.txt
```

---

## Commit R6 (Nicat): [run_repeated_seed()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#431-544) — Task 7 (L427–L543)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(trackA): add 5-seed repeated evaluation with 95% CI on digits"
```

```python
def run_repeated_seed():
    """5-seed evaluation on digits for both final configs."""
    seeds = [0, 1, 2, 3, 4]

    for model_name in ['Softmax', 'NN']:
        for s in seeds:
            if model_name == 'Softmax':
                model = SoftmaxRegression(...); opt = SGD(lr=0.05)
            else:
                model = NeuralNetwork(..., 32, ...); opt = Adam(lr=0.001)
            _, best_p, _ = train_model(model, opt, ..., seed=s, verbose=False)
            acc = compute_accuracy(model, X_test, y_test)
            loss = compute_loss(model, X_test, y_test, n_classes)

        # 95% CI: t_crit=2.776 (df=4), mean ± t*s/√5
        ci_acc = 2.776 * std_acc / np.sqrt(5)

    # Bar chart with error bars → repeated_seed_digits.png
    # Save → results/repeated_seed.txt
```

---

## Commit R7 (Nicat): [run_track_a_pca()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#550-635) — Task 7: PCA/SVD (L546–L635)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(trackA): add PCA/SVD analysis with scree plot and softmax at dims {10, 20, 40, 64}"
```

```python
def run_track_a_pca():
    """Track A: PCA/SVD and input geometry analysis."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

    # Center data, SVD
    mean = X_train.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    explained_variance_ratio = (S**2) / np.sum(S**2)

    # 1. Scree plot
    plot_pca_scree(explained_variance_ratio, filename="pca_scree_digits.png")

    # 2. 2D PCA visualization
    X_2d = X_full_c @ Vt[:2].T
    plot_pca_2d(X_2d, y_full, filename="pca_2d_digits.png")

    # 3. Softmax at PCA dims {10, 20, 40, 64}
    for m in [10, 20, 40, 64]:
        Vm = Vt[:m]; X_tr_pca = X_train_c @ Vm.T
        sr_pca = SoftmaxRegression(m, n_classes)
        train_model(sr_pca, SGD(lr=0.05), ...)

    plot_pca_softmax_comparison(pca_dims, val_accs, val_losses, ...)
    # Save → results/track_a_pca.txt
```

---

## Commit R8 (Nicat): [run_failure_analysis()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#641-723) — Task 4+7 (L637–L722)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(analysis): add failure-case analysis — under-capacity NN (h=2) vs adequate (h=32) on digits"
```

```python
def run_failure_analysis():
    """Failure-case: under-capacity NN on digits."""
    # Failure: NN h=2 on 10-class digits (bottleneck)
    nn_fail = NeuralNetwork(n_features, 2, n_classes)
    hist_fail = train_model(nn_fail, SGD(lr=0.05), ..., n_epochs=200)

    # Success: NN h=32, Adam
    nn_good = NeuralNetwork(n_features, 32, n_classes)
    hist_good = train_model(nn_good, Adam(lr=0.001), ..., n_epochs=200)

    # Plot: loss/acc comparison h=2 vs h=32 → failure_analysis.png
    # Save analysis → results/failure_analysis.txt
```

---

## Commit R9 (Shared): [main()](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py#729-749) — Final orchestration (L725–L753)

```bash
git add starter_pack/src/run_experiments.py
git commit -m "feat(experiments): add main() orchestration — runs all experiments sequentially"
```

```python
def main():
    run_sanity_checks()          # Samir — Task 3
    run_synthetic_experiments()  # Sharaf+Nicat — Task 4+5
    run_digits_experiment()      # Samir — Task 6
    run_capacity_ablation()      # Sharaf — Task 5
    run_optimizer_study()        # Samir — Task 6
    run_repeated_seed()          # Nicat — Task 7
    run_track_a_pca()            # Nicat — Task 7
    run_failure_analysis()       # Nicat — Task 4+7

if __name__ == "__main__":
    main()
```

---

# 📦 Final Commits (Hamı)

## Commit F1: Results & Figures

```bash
git add starter_pack/figures/ starter_pack/results/
git commit -m "results: add all experiment figures and result logs"
```

## Commit F2: README update

```bash
git add README.md starter_pack/README.md
git commit -m "docs: update README with team roles, setup instructions, and reproduction steps"
```

## Commit F3: Report & Slides

```bash
git add starter_pack/report/main.tex starter_pack/slides/
git commit -m "docs: add final LaTeX report and presentation slides"
```

---

# ⏱️ ÜMUMI TIMELINE — Bütün Git Commandları Ardıcıllıqla

```
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — Foundation (Sharaf birinci)                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Sharaf:                                                             ║
║    S1. git commit -m "feat(src): initialize src as Python package"   ║
║    S2. git commit -m "feat(models): implement SoftmaxRegression      ║
║        and NeuralNetwork with softmax, cross-entropy, one-hot utils" ║
║    S3. git commit -m "feat(data): add synthetic and digits data      ║
║        loaders with mini-batch generator"                            ║
║                                                                      ║
║        >>> Sharaf pushes branch + PR → merge to main <<<             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PHASE 2 — Parallel (Samir + Sharaf)                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Samir (git pull main first):                                        ║
║    M1. git commit -m "feat(optimizers): implement SGD, Momentum,     ║
║        and Adam optimizers from scratch"                             ║
║    M2. git commit -m "feat(train): add mini-batch training loop      ║
║        with validation checkpointing"                                ║
║    M3. git commit -m "feat(sanity): implement 5 automated sanity     ║
║        checks per rubric Section 5.5"                                ║
║                                                                      ║
║  Sharaf (eyni zamanda):                                              ║
║    P1. git commit -m "feat(plotting): add decision boundary,         ║
║        comparison, and loss curve visualizations"                    ║
║    P2. git commit -m "feat(plotting): add capacity ablation          ║
║        boundary and curve plots for moons"                           ║
║                                                                      ║
║        >>> Both push branches + PRs → merge to main <<<             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PHASE 3 — Experiments (ardıcıl merge — conflict qaçınmaq üçün)   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Sharaf (git pull main first):                                       ║
║    R0. git commit -m "feat(experiments): add base imports and        ║
║        experiment infrastructure"                                    ║
║    R4. git commit -m "feat(experiments): add capacity ablation       ║
║        on moons with hidden widths {2, 8, 32}"                      ║
║                                                                      ║
║        >>> Sharaf PR → merge to main <<<                             ║
║                                                                      ║
║  Samir (git pull main first):                                        ║
║    R1. git commit -m "feat(experiments): add automated sanity        ║
║        checks"                                                       ║
║    R3. git commit -m "feat(experiments): add digits benchmark        ║
║        baseline (Softmax vs NN h=32)"                                ║
║    R5. git commit -m "feat(experiments): add optimizer study         ║
║        (SGD/Momentum/Adam) on digits with h=32 NN"                  ║
║    P3. git commit -m "feat(plotting): add optimizer comparison       ║
║        plot for digits study"                                        ║
║                                                                      ║
║        >>> Samir PR → merge to main <<<                              ║
║                                                                      ║
║  Nicat (git pull main first):                                        ║
║    R2. git commit -m "feat(experiments): add synthetic benchmarks    ║
║        (linear_gaussian + moons)"                                    ║
║    R6. git commit -m "feat(trackA): add 5-seed repeated             ║
║        evaluation with 95% CI on digits"                             ║
║    R7. git commit -m "feat(trackA): add PCA/SVD analysis with        ║
║        scree plot and softmax at dims {10,20,40,64}"                 ║
║    R8. git commit -m "feat(analysis): add failure-case analysis"     ║
║    P4. git commit -m "feat(plotting): add PCA scree, 2D projection,  ║
║        and softmax comparison plots for Track A"                     ║
║                                                                      ║
║        >>> Nicat PR → merge to main <<<                              ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PHASE 4 — Final (Hamı birlikdə)                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  R9. git commit -m "feat(experiments): add main() orchestration"     ║
║  F1. git commit -m "results: add all experiment figures and logs"    ║
║  F2. git commit -m "docs: update README with team roles and setup"   ║
║  F3. git commit -m "docs: add final LaTeX report and slides"         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

> [!IMPORTANT]
> **Merge conflict qaçınmaq üçün:** Phase 3-də [run_experiments.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/run_experiments.py) və [plotting.py](file:///c:/Users/Sharaf/Desktop/math4ai_capstone-main/math4ai_capstone-main/starter_pack/src/plotting.py) üçün PRlar **ardıcıl** merge edilməlidir: Sharaf → Samir → Nicat. Paralel merge etsəniz conflict olacaq!
