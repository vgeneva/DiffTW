# %% [markdown]
# # DiffTW Demo
# This notebook demonstrates the DiffTW algorithm for learning time warping functions to align time series data. It includes:
# 1. Importing necessary libraries and defining the DiffTW class.
# 2. Loading and preprocessing time series data.
# 3. Initializing the DiffTW model and learning the time warping function (beta).
# 4. Visualizing the results, including the original and aligned time series, and the optimization process.

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


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

sys.path.append(project_root)

# bring in difftw class from difftw.py
from difftw import get_beta, Optimization, DataLoader, proj_kernel
from plotting import optimize_and_plot
from characteristics import plot_warping_characteristics
# %% [markdown]
# # Synthetic Data Generation
# For demonstration purposes, we will generate synthetic time series data. 
# We will create two time series, $\phi_0$ and $\phi_1$, where $\phi_1$ is a warped 
# version of $\phi_0$. The goal of DiffTW is to learn the warping function ($\beta_{\text{true}}$) 
# that aligns $\phi_0$ to $\phi_1$.

# %% [markdown]
# Generate $$\phi_0 = \beta_0^T \gamma_0(x).$$

# %%
np.random.seed(3)
D = 50 # No. of features for feature map
feat_coeff = 1/np.sqrt(D)
beta_0 = np.random.normal(scale = 100 / np.sqrt(D), size = (2 * D, 1)) / 10
beta_0 = tf.constant(beta_0, dtype=tf.float64)


# Generate $\phi_0$ function values using the feature map and beta_0
n=1000
a=0
b=1
x_ini = np.linspace(a, b, n, endpoint=True)
x_ini = np.reshape(x_ini,(n,1))

rff_class = proj_kernel()
def phi_0(x, beta_0=beta_0):
    return tf.linalg.matmul(tf.transpose(beta_0), rff_class._feature_map(x))

phi0 = phi_0(x_ini)


plt.figure(figsize=(16,8))
plt.plot(x_ini, phi0.numpy().flatten(), label=r'$\phi_0$', color='blue')
plt.title(r'$\phi_0$ function values')
plt.legend()
plt.show()

# %% [markdown]
# Generate $$\alpha_{\text{true}} = \beta_{\text{true}}^T \gamma_0(x).$$

np.random.seed(7)
beta_t = np.random.normal(scale = 2/ np.sqrt(D), size = ((2 * D)-2, 1)) / 5
beta_true = tf.constant(beta_t, dtype=tf.float64)

def alpha(t, x, beta_true=beta_true):
    make_LU = rff_class.make_proj_info()
    return tf.linalg.matmul(tf.transpose(beta_true), rff_class.feature_map_0(x))

plt.plot(x_ini, alpha(1, x_ini).numpy().flatten(), label=r'$\alpha_{\text{true}}$', color='orange')
plt.title(r'$\alpha_{\text{true}}$ function values')
plt.legend()
plt.show()

# %% [markdown]
# Generate $\phi_0$ by integrating $\alpha_{\text{true}}$ to get the warped time axis, then applying $\phi_0$ to the warped time axis:
def forward_euler_alpha(x, b=beta_true, num=20, a = alpha):
    sa = np.zeros((len(x), num))
    x_t = tf.cast(x, tf.float64)
    sa[:, 0] = np.reshape(x, len(x))
  
    for i in range(1, num):
        x_t += (1/(num-1)) * tf.transpose(a(1, x_t, b))
        x_t = tf.clip_by_value(x_t, 0.0, 1.0)
        sa[:, i] = np.reshape(x_t.numpy(), len(x))
    return sa #(n,num) matrix array of x @ time for each euler step

sol_alpha = forward_euler_alpha(x_ini)


# %% [markdown]
# Look at actual warping alignment path
plt.plot(x_ini, sol_alpha[:, -1], label=r'$\alpha_{\text{true}}$ warping path', color='orange')
plt.title(r'Actual warping alignment path')
plt.legend()
plt.show()

# %% [markdown]
print(sol_alpha[:, -1].shape)
warped_time_axis = sol_alpha[:, -1].reshape(-1, 1)
print(warped_time_axis.shape)
print(x_ini.shape)
# Now we can apply $\phi_0$ to the warped time axis to get $\phi_1$:
plt.plot(x_ini, phi_0(x_ini).numpy().flatten(), label=r'$\phi_0$', color='blue')
plt.plot(sol_alpha[:, -1], phi_0(x_ini).numpy().flatten(), label=r'$\phi_1$', color='orange')
plt.title(r'$\phi_0$ and $\phi_1$ function values')
plt.legend()
plt.show()

# %% [markdown]
def phi_1(x_query):
    """
    Interpolate the warped signal so it can be evaluated at any x.
    """
    x_warped = sol_alpha[:, -1]
    y_values = phi_0(x_ini).numpy().flatten()
    interp_func = interp1d(x_warped, y_values, kind='linear', fill_value='extrapolate')
    x_query_flat=np.array(x_query).flatten()
    return interp_func(x_query_flat)

phi1 = phi_1(x_ini)
print(phi1.shape)

plt.figure(figsize=(16,8))
plt.plot(sol_alpha[:, -1], phi_0(x_ini).numpy().flatten(), label=r'$\phi_0$ (warped)', color='blue')
plt.plot(x_ini, phi1.reshape(-1, 1), label=r'$\phi_1$ (warped)', color='orange')
plt.title(r'$\phi_1$ function values (warped $\phi_0$)')
plt.legend()
plt.show()

# %% [markdown]
# # DiffTW Optimization
# Now we will use the DiffTW algorithm to learn the warping function $\beta$ that
# aligns $\phi_0$ to $\phi_1$. We will initialize $\beta$ randomly and optimize 
# it using gradient descent to minimize the objective function defined in the DiffTW class.

# %%

# %% [markdown]
# Use synthetic data to test the DiffTW optimization process.
# We will initialize the get_beta class with the synthetic data and 
# then run the optimization to learn the warping function beta.
if __name__ == "__main__":
    phi0_synthetic = phi_0(x_ini).numpy().flatten()
    phi1_synthetic = phi1.flatten()
    overall=get_beta(proj_kernel_class=proj_kernel,
                     phi0_array=phi0_synthetic,
                     phi1_array=phi1_synthetic)
    
    n_steps_beta = 20 # global number of euler steps, integration using updated beta
    n_iters_beta = 20  # number of iterations for optimizing beta
    LAM = 1e-4
    x0=overall.i_traj(len(overall.phi0_test))  # original time axis for phi0

    np.random.seed(3)
    mean = 0.001
    sd = 0.001
    beta_initial = np.random.normal(mean, sd, ((overall.proj_kernel.num_features*2)-2,1)) 
    print(beta_initial.shape)
    beta_initial_tf = tf.Variable(beta_initial, dtype=tf.float64)
    # x1_est: estimated aligned time axis for phi0_test w/o learning, using initalized beta
    x1_est, _ = overall.forward_euler(beta_initial_tf, overall.phi0_test, num_steps = n_steps_beta)

    plt.figure(figsize=(16,8))
    plt.plot(x0, overall.phi1_train, label=r'$\phi_1$ (train)', color='blue')
    plt.plot(x0, overall.phi0_test, label=r'$\phi_0$ (test)', color='orange')
    plt.title(r'$\phi_0$, $\phi_1$ Functions')
    plt.legend()
    plt.show()

    model = Optimization(overall)
    optimized_beta, loss_list = model.optimize(beta_init = beta_initial, num_steps = n_steps_beta, num_iterations=n_iters_beta, LAM=LAM)
    print('Minimum loss during optimization:', round(min(loss_list), 4))
    plt.figure(figsize=(16,8))
    plt.plot(range(len(loss_list)), loss_list, label='Loss over iterations')
    plt.scatter(range(len(loss_list)), loss_list, color='red')
    plt.title('Loss vs Iterations during Beta Optimization')
    plt.ylim(0, max(loss_list)*1.1)  # set y-axis limits for better visualization
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # %% [markdown]
    # We can use the optimze and plot function
    # to provide a live visualization of the optimization process, 
    # including the warping paths and objective values across iterations.
    optimized_beta, loss_history = optimize_and_plot(
    overall=overall,
    n_iters_beta=20,
    n_steps_beta=20,
    LAM=1e-4,
    lr=0.00675,
    beta_true=beta_true  # Omit this argument entirely if using real data!
    )

    x1_optimized, X_est_opt = overall.forward_euler(optimized_beta, overall.phi0_test, num_steps=20)

    plot_warping_characteristics(
        overall=overall, 
        X_est_opt=X_est_opt, 
        n_steps_beta=20, 
        offset=4, 
        line_stride=16, 
        save_dir="plots"
    )

    # %% [markdown]
    # DTW Alignment Path Visualization
    # To further understand the alignment process, we can visualize the DTW alignment 
    # path between $\phi_0$ and $\phi_1$ before and after optimization. This will show how 
    # the warping function is aligning the two signals over iterations.
    dtw(overall.phi1_train, overall.phi0_test, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="alignment");

    dtw(overall.phi1_train, overall.phi0_test, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway", offset=-4);


# %%
# Using example from UCR dataset, BME.
if __name__ == "__main__":


    
    #datapath_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/STW/current_filtered_data/jj_train_matrix.csv'
    #datapath_test = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/STW/current_filtered_data/N_test_matrix.csv'  #g
    #datapath_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/STW/UCRArchive_data/ACSFone/ACSFone_train_matrix.csv'
    #datapath_test = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/STW/UCRArchive_data/ACSFone/ACSFone0_test_matrix.csv'
    #datapath_train = '/Users/hane492/safe/PSU/git_repos/UCRArchive_data/Adiac/Adiac_train_matrix.csv'
    #datapath_test = '/Users/hane492/safe/PSU/git_repos/UCRArchive_data/Adiac/Adiac1_test_matrix.csv'
    #datapath_train = '/Users/hane492/safe/PSU/git_repos/current_filtered_data/all1_train_matrix.csv'
    #datapath_test = '/Users/hane492/safe/PSU/git_repos/current_filtered_data/V_test_matrix.csv'  #g
    datapath_train =  "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/STW/UCRArchive_data/BME/BME_train_matrix.csv"
    datapath_test = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/STW/UCRArchive_data/BME/BME1_test_matrix.csv"

    overall = get_beta(datapath_train, datapath_test, DataLoader, proj_kernel, indexphi0 = 0, indexphi1 = 0)
    print(overall.phi0_test.shape)
    print(overall.phi1_train.shape)

    #1. Initial beta and x@time estimate
    n_steps_beta = 20 # global number of euler steps, integration using updated beta
    n_iters_beta = 50  # number of iterations for optimizing beta
    LAM = 1e-4
    x0 = overall.i_traj(len(overall.phi0_test))  # original time axis for phi0

    np.random.seed(3)
    mean = 0.001
    sd = 0.001
    beta_initial = np.random.normal(mean, sd, ((overall.proj_kernel.num_features*2)-2,1)) 
    print(beta_initial.shape)
    beta_initial_tf = tf.Variable(beta_initial, dtype=tf.float64)
    # x1_est: estimated aligned time axis for phi0_test w/o learning, using initalized beta
    x1_est, _ = overall.forward_euler(beta_initial_tf, overall.phi0_test, num_steps = n_steps_beta)


    # plot phi_0, phi_1, 
    plt.figure(figsize=(16,8))
    plt.plot(x0, overall.phi1_train, label=r'$\phi_1$ (train)', color='blue')
    plt.plot(x0, overall.phi0_test, label=r'$\phi_0$ (test)', color='orange')
    plt.title(r'$\phi_0$, $\phi_1$, $\phi_0$ estimated with initialized $\beta$')
    plt.legend()
    plt.show()

    # 2. Learning beta
    model = Optimization(overall)
    optimized_beta, loss_list = model.optimize(beta_init = beta_initial, num_steps = n_steps_beta, num_iterations=n_iters_beta, LAM=LAM)
    # print the minimum loss, round to 4 decimal places
    print('Minimum loss during optimization:', round(min(loss_list), 4))
    # plot loss_list
    plt.figure(figsize=(16,8))
    plt.plot(range(len(loss_list)), loss_list, label='Loss over iterations')
    # make scatter ontop of line plot
    plt.scatter(range(len(loss_list)), loss_list, color='red')
    plt.title('Loss vs Iterations during Beta Optimization')
    plt.ylim(0, max(loss_list)*1.1)  # set y-axis limits for better visualization
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(f'plots/loss_vs_iterations.pdf')

    # plot phi_0, phi_1, and phi_0 using optimized beta
    x1_optimized, _ = overall.forward_euler(optimized_beta, overall.phi0_test, num_steps = n_steps_beta)
    plt.figure(figsize=(16,8))
    plt.plot(x0, overall.phi0_test, label=r'$\phi_0$ (test)', color='blue')
    plt.plot(x0, overall.phi1_train, label=r'$\phi_1$ (train)', color='orange')
    plt.plot(x1_optimized, overall.phi0_test, label=r'$\phi_0$ with optimized $\beta$', color='red')
    plt.title(r'$\phi_0$, $\phi_1$, $\phi_0$ estimated with optimized $\beta$')
    plt.legend()
    plt.savefig(f'plots/phi_estimated_with_optimized_beta.pdf')
    plt.show()
    
    # %% [markdown]
    optimized_beta, loss_history = optimize_and_plot(
        overall=overall,
        n_iters_beta=11,
        n_steps_beta=20,
        LAM=1e-4,
        lr=1/(2**8),
        save_dir="plots"
    )

    x1_optimized, X_est_opt = overall.forward_euler(optimized_beta, overall.phi0_test, num_steps=20)

    plot_warping_characteristics(
        overall=overall, 
        X_est_opt=X_est_opt, 
        n_steps_beta=20, 
        offset=4, 
        line_stride=4, 
        save_dir="plots"
    )

    # %% [markdown]
        # %% [markdown]
    # DTW Alignment Path Visualization
    # To further understand the alignment process, we can visualize the DTW alignment 
    # path between $\phi_0$ and $\phi_1$ before and after optimization. This will show how 
    # the warping function is aligning the two signals over iterations.
    dtw(overall.phi1_train, overall.phi0_test, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="alignment");

    dtw(overall.phi1_train, overall.phi0_test, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway", offset=-4);

 