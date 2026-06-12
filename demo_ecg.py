# %% [markdown]
# # DiffTW Demo 2 — Real Data (ECG data and UCR Archive)
#
# This notebook applies the DiffTW algorithm to a **real-world** time series from
# the UCR Time Series Archive.  The default dataset is **ECG** (electrocardiogram), 
# but the commented-out paths at the top of the
# script show how to swap in other UCR datasets or your own CSV files.
#
# Unlike Demo 1, the true warping function is *unknown*.  The goal is to find a
# smooth, differentiable velocity field β that aligns one class of waveforms
# (φ₀, the test signal) to another (φ₁, the training signal).
#
# ## What this demo covers
# 1. Loading pre-formatted CSV data via `DataLoader`.
# 2. Verifying the input signals and running the DiffTW optimiser.
# 3. Visualising the learned warping path and aligned signals.
# 4. Comparing with a classic DTW baseline.
#
# ## Data format
# Each CSV should be a matrix where **rows are time series instances** and
# **columns are time steps**.  `indexphi0` and `indexphi1` select which row
# (instance) to use as φ₀ and φ₁ respectively.  Adjust these indices to
# experiment with different pairs of waveforms.
#
# ## Prerequisites
# Install dependencies before running:
#   pip install numpy matplotlib tensorflow dtw-python scipy
#
# Make sure `difftw.py`, `plotting.py`, and `characteristics.py` are on your
# Python path (or in the same directory as this file).

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import dtw
from dtw import *

# ---------------------------------------------------------------------------
# Path setup — adjust if your project layout differs
# ---------------------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.append(project_root)

from difftw import get_beta, Optimization, DataLoader, proj_kernel
from plotting import optimize_and_plot, optimize_and_plot_dashboard
from characteristics import plot_warping_characteristics


# %% [markdown]
# ---
# ## Step 1 — Choose your dataset
#
# Set `datapath_train` and `datapath_test` to point at your CSV files.
# A selection of commented-out alternatives is provided below for convenience.
# The active paths use the ECG dataset adn can be swtiched to datq from the UCR archive.
#
# To use your own data, replace the paths and ensure the CSV matrices match the
# format described in the header above.

# %%
# ---------------------------------------------------------------------------
# Dataset paths — edit these to switch datasets
# ---------------------------------------------------------------------------

# UCR: BME (default)
#datapath_train = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/UCRArchive_data/BME/BME_train_matrix.csv"
#datapath_test  = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/UCRArchive_data/BME/BME1_test_matrix.csv"

# ---------------------------------------------------------------------------
# Uncomment one of the blocks below to try a different dataset:
# ---------------------------------------------------------------------------

# UCR: ACSFone
# datapath_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/UCRArchive_data/ACSFone/ACSFone_train_matrix.csv'
# datapath_test  = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/UCRArchive_data/ACSFone/ACSFone0_test_matrix.csv'

# UCR: Adiac
# datapath_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/UCRArchive_data/Adiac/Adiac_train_matrix.csv'
# datapath_test  = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/UCRArchive_data/Adiac/Adiac1_test_matrix.csv'

# ECG data 
    
datapath_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/ECG/all1_train_matrix.csv'
datapath_test = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/DiffTW/dataframes/ECG/N_test_matrix.csv' 


# %% [markdown]
# ---
# ## Step 2 — Load data and initialise DiffTW
#
# `get_beta` accepts the CSV paths and a `DataLoader` to handle file I/O.
# `indexphi0` and `indexphi1` select which rows of the train/test matrices to
# use as the source signal (φ₀) and target signal (φ₁).  Change these integers
# to align a different pair of signals.

# %%
if __name__ == "__main__":
    overall = get_beta(
        datapath_train,
        datapath_test,
        DataLoader,
        proj_kernel,
        indexphi0=0,   # row index for φ₀ in the test matrix
        indexphi1=0,   # row index for φ₁ in the train matrix
    )

    print(f'φ₀ shape: {overall.phi0_test.shape}')
    print(f'φ₁ shape: {overall.phi1_train.shape}')

    x0 = overall.i_traj(len(overall.phi0_test))  # uniform time axis

    plt.figure(figsize=(16, 8))
    plt.plot(x0, overall.phi1_train, label=r'$\phi_1$ (target, train)', color='blue')
    plt.plot(x0, overall.phi0_test,  label=r'$\phi_0$ (source, test)',  color='orange')
    plt.title(r'Input signals: $\phi_0$ and $\phi_1$')
    plt.xlabel('Normalised time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## Step 3 — Set hyperparameters and initialise β
#
# Key hyperparameters:
# - `n_steps_beta`  : Euler integration steps per forward pass.
#                     More steps → more accurate integration, but slower.
# - `n_iters_beta`  : gradient-descent iterations.
#                     Increase if the loss has not yet plateaued.
# - `LAM`           : L2 regularisation weight on β.
#                     Larger values encourage a smoother (closer to identity) warp.
# - `lr`            : learning rate passed to `optimize_and_plot`.
#                     Tune if loss oscillates (decrease) or converges too slowly (increase).
#
# β is initialised with small random values near zero, which corresponds to a
# near-identity warp — a sensible starting point.

# %%
if __name__ == "__main__":
    n_steps_beta = 20     # Euler steps per forward pass
    n_iters_beta = 11     # gradient-descent iterations
    LAM = 1e-4            # regularisation weight

    np.random.seed(3)
    beta_initial = np.random.normal(
        0.001, 0.001,
        ((overall.proj_kernel.num_features * 2) - 2, 1),
    )
    print(f'β shape: {beta_initial.shape}')

    beta_initial_tf = tf.Variable(beta_initial, dtype=tf.float64)

    # Quick sanity check — warp φ₀ with the initialised (near-zero) β
    x1_est, _ = overall.forward_euler(
        beta_initial_tf, overall.phi0_test, num_steps=n_steps_beta
    )


# %% [markdown]
# ---
# ## Step 4 — Run the optimiser
#
# `Optimization.optimize` performs gradient descent on the DiffTW objective:
#
#   L(β) = ‖φ₁ − φ₀ ∘ warp_β‖² + λ ‖β‖²
#
# The loss is printed after each iteration.  If it is not decreasing, try
# reducing the learning rate or increasing `LAM`.

# %%
if __name__ == "__main__":
    model = Optimization(overall)
    optimized_beta, loss_list = model.optimize(
        beta_init=beta_initial,
        num_steps=n_steps_beta,
        num_iterations=n_iters_beta,
        LAM=LAM,
    )

    print(f'Minimum loss during optimisation: {min(loss_list):.4f}')

    plt.figure(figsize=(16, 8))
    plt.plot(range(len(loss_list)), loss_list, label='Loss')
    plt.scatter(range(len(loss_list)), loss_list, color='red', zorder=5)
    plt.title('Optimisation loss vs. iteration')
    plt.ylim(0, max(loss_list) * 1.1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/loss_vs_iterations.pdf')
    plt.show()

    # Show the aligned signals
    x1_optimized, _ = overall.forward_euler(
        optimized_beta, overall.phi0_test, num_steps=n_steps_beta
    )

    plt.figure(figsize=(16, 8))
    plt.plot(x0, overall.phi0_test,   label=r'$\phi_0$ (source)',  color='blue')
    plt.plot(x0, overall.phi1_train,  label=r'$\phi_1$ (target)',  color='orange')
    plt.plot(x1_optimized, overall.phi0_test,
             label=r'$\phi_0$ warped with $\hat{\beta}$', color='red', linestyle='--')
    plt.title(r'Alignment result: $\phi_0$, $\phi_1$, and warped $\phi_0$')
    plt.xlabel('Normalised time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/phi_estimated_with_optimized_beta.pdf')
    plt.show()


# %% [markdown]
# ---
# ## Step 5 — Live optimization visualisation
#
# `optimize_and_plot` re-runs the optimiser and renders a live plot that updates after each step, letting you watch
# the warping path evolve.
#
# For real data there is no `beta_true` to pass, so the ground-truth overlay is
# omitted automatically.

# %%I
if __name__ == "__main__":
    optimized_beta, loss_history = optimize_and_plot(
        overall=overall,
        n_iters_beta=11,
        n_steps_beta=20,
        LAM=1e-4,
        lr=1 / (2 ** 9),
        save_dir="plots",
        phi0_label="N", #"BME1", #"N"
        phi1_label="R" #"BME train" #"R"
        # No beta_true here — ground truth is unknown for real data
    )

    # Recover the full Euler trajectory for warping characteristic plots
    x1_optimized, X_est_opt = overall.forward_euler(
        optimized_beta, overall.phi0_test, num_steps=20
    )

    # Warping characteristic curves — show how individual time points move
    # through the learned flow field.  `line_stride` controls how many
    # characteristic lines are drawn; reduce it for a denser plot.
    plot_warping_characteristics(
        overall=overall,
        X_est_opt=X_est_opt,
        n_steps_beta=20,
        offset=4,
        line_stride=4,
        save_dir="plots",
    )

# %%
# `optimize_and_plot_dashboard` runs the optimiser and renders a live 3x2 
# dashboard that updates after each step, letting you watch the warping path 
# evolve, and calculating the characteristic curves at the end.

# %%
if __name__ == "__main__":
    optimized_beta, loss_history = optimize_and_plot_dashboard(
        overall=overall,
        n_iters_beta=11,
        n_steps_beta=20,
        LAM=1e-4,
        lr=1 / (2 ** 9),
        save_dir="plots",
        phi0_label="N", 
        phi1_label="R",
        offset=4,           # Controls characteristic plot spacing
        line_stride=4       # Controls how many grey lines are plotted
    )

# %% [markdown]
# ---
# ## Step 6 — DTW alignment path (baseline comparison)
#
# Classic Dynamic Time Warping provides a useful point of comparison.  The two
# plots below show:
# - **alignment** — the DTW matching path in index space.
# - **twoway**    — both signals overlaid after DTW alignment.
#
# Because DTW produces a piecewise-constant, non-differentiable alignment, it
# cannot be used directly in a gradient-based pipeline.  DiffTW addresses this
# limitation by parameterising the warp as a smooth flow.

# %%
if __name__ == "__main__":
    dtw(overall.phi1_train, overall.phi0_test,
        keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="alignment")

    dtw(overall.phi1_train, overall.phi0_test,
        keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway", offset=-4)
# %%
