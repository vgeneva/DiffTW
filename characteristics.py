import os
import numpy as np
import matplotlib.pyplot as plt

def plot_warping_characteristics(overall, X_est_opt, n_steps_beta, offset=4, line_stride=4, save_dir="plots"):
    """
    Plots the characteristic curves showing the warping evolution.
    Uses 'line_stride' to skip points and reduce visual clutter.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_phi0 = len(overall.phi0_test)
    num_iters = n_steps_beta 
    
    x0 = overall.i_traj(num_phi0).numpy().flatten()

    plt.figure(figsize=(10, 8))
    
    # Plot base signals
    plt.plot(x0, overall.phi0_test, label=r'$\phi_0$ (test)', color='black', alpha=1, linewidth=2)
    plt.plot(x0, overall.phi1_train + offset, label=r'$\phi_1$ (train) offset', color='blue', alpha=1, linewidth=2)
    
    # Extract final warped estimation safely
    final_x = X_est_opt[-1].numpy().flatten() if hasattr(X_est_opt[-1], 'numpy') else np.array(X_est_opt[-1]).flatten()
    plt.plot(final_x, overall.phi0_test + offset, label=r'Offset Estimated $\phi_1$', color='red', alpha=1, linewidth=2)

    # Plot characteristic lines using line_stride to skip every Nth line
    for j in range(0, num_phi0, line_stride):
        PHI0_array = np.full(num_iters, overall.phi0_test[j])
        
        X_array = []
        for i in range(num_iters):
            val = X_est_opt[i][j, 0].numpy() if hasattr(X_est_opt[i], 'numpy') else X_est_opt[i][j, 0]
            X_array.append(val)
            
        X_array = np.array(X_array)
        PHI0_array_offset = PHI0_array + np.linspace(0, offset, num_iters)
        
        # Thinned out lines
        plt.plot(X_array, PHI0_array_offset, '-', color='gray', alpha=0.5, linewidth=1)

    plt.xlabel("Warped Time (x) Across Euler Iterations")
    plt.ylabel("Amplitude (with offset)")
    plt.title(f"Warping Evolution (Plotting every {line_stride}th line)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "warping_characteristics.pdf"), format='pdf', bbox_inches='tight')
    plt.show()

