# %% [markdown]
# # DiffTW Demo 1 — Synthetic Data
#
# This notebook demonstrates the DiffTW algorithm on **synthetically generated**
# time series data.  This is a good starting point if you are new to DiffTW,
# because the true warping function is known in advance and can be compared directly
# against the learned result.
#
# ## What this demo covers
# 1. Generating a reference signal φ₀ from a random-feature kernel expansion
#    using difftw.py.
# 2. Constructing a ground-truth velocity field α_true and integrating it via
#    forward-Euler to produce the warped signal φ₁.
# 3. Running the DiffTW optimizer to learn β̂ (the parameterisation of the
#    velocity field) from (φ₀, φ₁) alone.
# 4. Visualising the optimisation loss, the recovered warping path, and comparing
#    it against the known β_true.
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
from numpy import linalg as LA
import tensorflow as tf
import os
import sys
import time
import dtw
from dtw import *
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Path setup — adjust if your project layout differs
# ---------------------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.append(project_root)

from difftw import get_beta, Optimization, DataLoader, proj_kernel
from plotting import optimize_and_plot, optimize_and_plot_dashboard, plot_warping_characteristics


# %% [markdown]
# ---
# ## Step 1 — Generate the reference signal φ₀
#
# φ₀ is constructed as a linear combination of random Fourier features (RFF):
#
#   φ₀(x) = β₀ᵀ γ₀(x)
#
# where γ₀ is the feature map defined by `proj_kernel` and β₀ is drawn from a
# scaled normal distribution.  The domain is x ∈ [0, 1] discretised at n = 1000
# evenly-spaced points.

# %%
np.random.seed(3)
D = 50                                           # number of random Fourier features
feat_coeff = 1 / np.sqrt(D)
beta_0 = np.random.normal(scale=100 / np.sqrt(D), size=(2 * D, 1)) / 10
beta_0 = tf.constant(beta_0, dtype=tf.float64)

n = 1000
a, b = 0, 1
x_ini = np.linspace(a, b, n, endpoint=True).reshape(n, 1)

rff_class = proj_kernel()

def phi_0(x, beta_0=beta_0):
    """Evaluate the reference signal φ₀ at query points x."""
    return tf.linalg.matmul(tf.transpose(beta_0), rff_class._feature_map(x))

phi0 = phi_0(x_ini)

plt.figure(figsize=(16, 8))
plt.plot(x_ini, phi0.numpy().flatten(), label=r'$\phi_0$', color='blue')
plt.title(r'Reference signal $\phi_0$')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## Step 2 — Define the true velocity field α_true
#
# The velocity field α_true(t, x) = β_trueᵀ γ₀(x) governs how the time axis
# is deformed.  β_true is another RFF coefficient vector drawn independently
# of β₀.  A *smaller* scale parameter is used so the warping is smooth but
# non-trivial.

# %%
np.random.seed(7)
beta_t = np.random.normal(scale=2 / np.sqrt(D), size=((2 * D) - 2, 1)) / 5
beta_true = tf.constant(beta_t, dtype=tf.float64)

def alpha(t, x, beta_true=beta_true):
    """True velocity field α evaluated at time t and spatial points x."""
    rff_class.make_proj_info()
    return tf.linalg.matmul(tf.transpose(beta_true), rff_class.feature_map_0(x))

plt.figure(figsize=(16, 8))
plt.plot(x_ini, alpha(1, x_ini).numpy().flatten(),
         label=r'$\alpha_\mathrm{true}$', color='orange')
plt.title(r'True velocity field $\alpha_\mathrm{true}$')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## Step 3 — Integrate α_true to obtain the warped time axis
#
# We use a simple forward-Euler integrator with `num` steps to propagate each
# point on the original time axis x under the flow induced by α_true.  The
# resulting trajectory `sol_alpha` has shape (n, num); the final column
# `sol_alpha[:, -1]` gives the warped positions.

# %%
def forward_euler_alpha(x, b=beta_true, num=20, a=alpha):
    """
    Integrate the ODE  dx/dt = α(t, x)  from t=0 to t=1
    using forward Euler with `num` steps.

    Returns
    -------
    sa : ndarray of shape (n, num)
        Position of each point at each Euler step.
    """
    sa = np.zeros((len(x), num))
    x_t = tf.cast(x, tf.float64)
    sa[:, 0] = np.reshape(x, len(x))

    for i in range(1, num):
        x_t += (1 / (num - 1)) * tf.transpose(a(1, x_t, b))
        x_t = tf.clip_by_value(x_t, 0.0, 1.0)
        sa[:, i] = np.reshape(x_t.numpy(), len(x))
    return sa

sol_alpha = forward_euler_alpha(x_ini)

plt.figure(figsize=(16, 8))
plt.plot(x_ini, sol_alpha[:, -1],
         label=r'Warping path $\beta_\mathrm{true}$', color='orange')
plt.plot(x_ini, x_ini, linestyle='--', color='gray', label='Identity (no warp)')
plt.title(r'Ground-truth warping path produced by $\alpha_\mathrm{true}$')
plt.xlabel('Original time axis')
plt.ylabel('Warped time axis')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## Step 4 — Construct the observed signal φ₁
#
# φ₁ is obtained by evaluating φ₀ on the *warped* time axis and then
# re-interpolating it back onto the uniform grid.  In practice, this means
# φ₁(x) ≈ φ₀(warp(x)).

# %%
warped_time_axis = sol_alpha[:, -1].reshape(-1, 1)

plt.figure(figsize=(16, 8))
plt.plot(x_ini, phi_0(x_ini).numpy().flatten(), label=r'$\phi_0$ (original)', color='blue')
plt.plot(sol_alpha[:, -1], phi_0(x_ini).numpy().flatten(),
         label=r'$\phi_1$ (warped)', color='orange')
plt.title(r'$\phi_0$ and $\phi_1$ before interpolation')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()


def phi_1(x_query):
    """
    Evaluate φ₁ at arbitrary query points by linearly interpolating the
    warped signal back onto the uniform grid.
    """
    x_warped = sol_alpha[:, -1]
    y_values = phi_0(x_ini).numpy().flatten()
    interp_func = interp1d(x_warped, y_values,
                           kind='linear', fill_value='extrapolate')
    return interp_func(np.array(x_query).flatten())

phi1 = phi_1(x_ini)

plt.figure(figsize=(16, 8))
plt.plot(x_ini, phi_0(x_ini).numpy().flatten(), label=r'$\phi_0$', color='blue')
plt.plot(x_ini, phi1, label=r'$\phi_1$ (interpolated onto uniform grid)', color='orange')
plt.title(r'$\phi_0$ and $\phi_1$ on the uniform grid')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## Step 5 — Initialise DiffTW and run the optimiser
#
# `get_beta` is the top-level DiffTW object.  It stores the training and test
# signals and exposes the forward-Euler integration and loss computation methods
# used by `Optimization`.
#
# Key hyperparameters:
# - `n_steps_beta`  : number of Euler integration steps per forward pass.
# - `n_iters_beta`  : number of gradient-descent iterations.
# - `LAM`           : L2 regularisation weight on β.
# - `lr`            : learning rate (default inside `optimize_and_plot` is 0.00675).

# %%
if __name__ == "__main__":
    phi0_synthetic = phi_0(x_ini).numpy().flatten()
    phi1_synthetic = phi1.flatten()

    overall = get_beta(
        proj_kernel_class=proj_kernel,
        phi0_array=phi0_synthetic,
        phi1_array=phi1_synthetic,
    )

    # ---------------------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------------------
    n_steps_beta = 20    # Euler steps per forward pass
    n_iters_beta = 21    # gradient-descent iterations
    LAM = 1e-4           # regularisation weight

    x0 = overall.i_traj(len(overall.phi0_test))  # uniform time axis for φ₀

    # Initialise β close to zero so the initial warp is near the identity
    np.random.seed(3)
    beta_initial = np.random.normal(0.001, 0.001,
                                    ((overall.proj_kernel.num_features * 2) - 2, 1))
    beta_initial_tf = tf.Variable(beta_initial, dtype=tf.float64)

    # Inspect the initial (un-learned) estimate of the warped axis
    x1_est, _ = overall.forward_euler(beta_initial_tf, overall.phi0_test,
                                       num_steps=n_steps_beta)

    plt.figure(figsize=(16, 8))
    plt.plot(x0, overall.phi1_train, label=r'$\phi_1$ (target)', color='blue')
    plt.plot(x0, overall.phi0_test,  label=r'$\phi_0$ (source)', color='orange')
    plt.title(r'Input signals: $\phi_0$ and $\phi_1$')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------------
    # Run the optimiser
    # ---------------------------------------------------------------------------
    model = Optimization(overall)
    optimized_beta, loss_list = model.optimize(
        beta_init=beta_initial,
        num_steps=n_steps_beta,
        num_iterations=n_iters_beta,
        LAM=LAM,
    )

    print(f'Minimum loss during optimization: {min(loss_list):.4f}')

    plt.figure(figsize=(16, 8))
    plt.plot(range(len(loss_list)), loss_list, label='Loss')
    plt.scatter(range(len(loss_list)), loss_list, color='red', zorder=5)
    plt.title('Optimization loss vs. iteration')
    plt.ylim(0, max(loss_list) * 1.1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## Step 6 — Live optimisation visualisation
#
# `optimize_and_plot` reruns the optimiser with a live plot that updates after
# every iteration, letting you watch the warping path converge.  Because the
# true β is available here (synthetic data), we pass `beta_true` so that the
# plot can overlay the ground-truth path for comparison.
#
# **Note:** omit the `beta_true` argument entirely when working with real data.

# %%
if __name__ == "__main__":
    optimized_beta, loss_history = optimize_and_plot(
        overall=overall,
        n_iters_beta=21,
        n_steps_beta=20,
        LAM=1e-4,
        lr=0.00675,
        beta_true=beta_true,   # remove this line when using real data
    )

    # Recover the full Euler trajectory for the optimised β
    x1_optimized, X_est_opt = overall.forward_euler(
        optimized_beta, overall.phi0_test, num_steps=n_steps_beta
    )

    # Warping characteristic curves — illustrate how individual time points move
    plot_warping_characteristics(
        overall=overall,
        X_est_opt=X_est_opt,
        n_steps_beta=20,
        offset=4,
        line_stride=16,
        save_dir="plots",
    )

# %%
if __name__ == "__main__":
    optimized_beta, loss_history = optimize_and_plot_dashboard(
        overall=overall,
        n_iters_beta=21,
        n_steps_beta=20,
        LAM=1e-4,
        lr=0.00675,
        beta_true=beta_true,   # remove this line when using real data
        offset=4,              # Controls characteristic plot spacing
        line_stride=16,        # Controls how many grey lines are plotted
        save_dir="plots"
    )

# %% [markdown]
# ---
# ## Step 7 — DTW alignment path (baseline comparison)
#
# Classic Dynamic Time Warping (DTW) provides a useful baseline.  The two plots
# below show the DTW alignment path and a two-way overlay of the signals after
# DTW alignment.  Comparing the DTW path with the DiffTW warping path highlights
# the key difference: DiffTW learns a *smooth, differentiable* flow rather than
# the piecewise-constant alignment of DTW.

# %%
if __name__ == "__main__":
    dtw(overall.phi1_train, overall.phi0_test,
        keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="alignment")

    dtw(overall.phi1_train, overall.phi0_test,
        keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway", offset=-4)
# %%
