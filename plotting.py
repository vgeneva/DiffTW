import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def optimize_and_plot(overall, n_iters_beta=50, n_steps_beta=20, LAM=1e-4, lr=0.00675, beta_true=None, save_dir="plots", phi0_label="Start", phi1_label="Target"):
    """
    Optimizes beta to align phi0 to phi1, showing live updates of alpha, warping paths, 
    objective values, and gradient norms, followed by standalone alignment plots.
    
    Args:
        overall: the instantiated get_beta object containing your data and kernels.
        n_iters_beta: number of optimization steps.
        n_steps_beta: number of forward Euler simulation steps.
        LAM: regularization parameter.
        lr: learning rate for GD.
        beta_true: (Optional) the true beta tensor for synthetic toy problems.
        save_dir: folder name to save the PDF plots.
        phi0_label: label for the phi0 signal.
        phi1_label: label for the phi1 signal.
    """
    os.makedirs(save_dir, exist_ok=True)
    x0 = overall.i_traj(len(overall.phi0_test))


    # Construct the legend string dynamically
    phi1_legend = r'$\phi_1$ ({})'.format(phi1_label)
    phi0_legend = r'$\phi_0$ ({})'.format(phi0_label)
    # >>> END OF ADDITION <<<


    # Initialize Beta
    np.random.seed(3)
    mean = 0.001
    sd = 0.001
    beta_initial = np.random.normal(mean, sd, ((overall.proj_kernel.num_features*2)-2, 1)) 
    beta_var = tf.Variable(beta_initial, dtype=tf.float64)
    
    print(f'Learning rate: {lr}')
    optimizer = tf.optimizers.SGD(learning_rate=lr)

    # --- Initialize 4-panel subplot ---
    fig, ((fig1, fig2), (fig3, fig4)) = plt.subplots(2, 2, figsize=(16, 10))

    fig2.set_xlim(0, n_iters_beta)
    fig2.set_ylim(-0.1, 0.5)
    fig3.set_xlim(0, n_iters_beta)
    fig3.set_ylim(0.0, 5.0)  

    scatter_obj = fig2.scatter([], [])
    line_obj, = fig2.plot([], [], 'r-')

    scatter_grad = fig3.scatter([], [])
    line_grad, = fig3.plot([], [], 'b-')

    obj_values = []
    iterations = []
    gradient_norms = []
    phi1_est_history = [] 

    # --- Pre-plot True Alpha if provided ---
    if beta_true is not None:
        true_alpha_vals = tf.squeeze(overall.make_func(x0, beta_true))
        fig4.plot(x0.numpy(), true_alpha_vals.numpy(), label=r'$\alpha_{\text{true}}$', color='green', linewidth=3)

    print("\n--- Starting Optimization Loop ---")
    start_time = time.time()

    for batch in range(n_iters_beta):
        with tf.GradientTape() as tape:
            obj = overall.J_obj(beta_var, num_steps=n_steps_beta, LAM=LAM)

        obj_val_sq = np.squeeze(obj.numpy())
        obj_values.append(obj_val_sq)  
        iterations.append(batch)
        grad = tape.gradient(obj, [beta_var])
        optimizer.apply_gradients(zip(grad, [beta_var]))

        print(f'Iteration: {batch} | Loss: {obj_val_sq:.4f}')

        # Step progression tracking
        if batch % 5 == 0:
            iter_color = cm.plasma(batch / n_iters_beta)

            # Fig 1: Warping Path
            x_traj, _ = overall.forward_euler(beta_var, overall.phi0_test, num_steps=n_steps_beta)
            fig1.plot(x0.numpy(), x_traj.numpy().flatten(), color=iter_color, label=f'{batch}', alpha=0.7)
            
            # Fig 4: Learned Alpha morphing
            func_vals = tf.squeeze(overall.make_func(x0, beta_var))
            fig4.plot(x0.numpy(), func_vals.numpy(), color=iter_color, label=f' {batch}', alpha=0.8)

            # Store phi_1 estimation for final standalone figure
            phi1_est = overall.phi_1_est(beta_var, num_steps=n_steps_beta)
            phi1_est_history.append((batch, x_traj.numpy(), iter_color))

        # Track gradient norms
        grad_norm = tf.norm(grad[0]).numpy()
        gradient_norms.append(grad_norm)

        # Live Update tracking plots
        scatter_obj.set_offsets(np.c_[iterations, obj_values])
        line_obj.set_data(iterations, obj_values)
        scatter_grad.set_offsets(np.c_[iterations, gradient_norms])
        line_grad.set_data(iterations, gradient_norms)
        
        # Dynamically scale tracking plots if values jump out of bounds
        if obj_val_sq > fig2.get_ylim()[1]:
            fig2.set_ylim(-0.1, obj_val_sq * 1.2)
        if grad_norm > fig3.get_ylim()[1]:
            fig3.set_ylim(0.0, grad_norm * 1.2)

    print(f"\nElapsed time: {time.time() - start_time:.2f} sec")

    # --- Plot final Learned Alpha ---
    final_alpha_vals = tf.squeeze(overall.make_func(x0, beta_var))
    fig4.plot(x0.numpy(), final_alpha_vals.numpy(), label=r'$\alpha$_est', color='blue', linewidth=3)

    # --- Panel Annotations ---
    fig1.set_title('Warping Path Progression', fontsize=20)
    fig1.plot(x0.numpy(), x0.numpy(), label='No Warping', color='lightgray', linestyle='--', linewidth=2)
    fig1.set_xlabel('x(t=0)', fontsize=18)
    fig1.set_ylabel('x(t) Across Euler Iterations', fontsize=18)
    fig1.legend(loc='upper left', fontsize=18)

    fig2.set_title('Objective Value Progress', fontsize=20)
    fig2.set_xlabel('Iteration', fontsize=18)
    fig2.set_ylabel('Objective Value', fontsize=18)

    fig3.set_title(r'Gradient Norm Progress', fontsize=20)
    fig3.set_xlabel('Iteration', fontsize=18)
    fig3.set_ylabel('Norm Value', fontsize=18)

    fig4.set_title(r'True $\alpha$ vs Learned $\alpha$' if beta_true is not None else r'Learned $\alpha$ Progression', fontsize=20)
    fig4.axhline(0, color='gray', linestyle='--', linewidth=1)
    fig4.set_xlabel('x0', fontsize=18)
    fig4.set_ylabel(r'$\alpha$', fontsize=18)
    fig4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "optimization_progress.pdf"), format='pdf', bbox_inches='tight')
    plt.show()

    # =====================================================================
    # --- Standalone Plot 1: Just original Phi_0 and Phi_1 ---
    # =====================================================================
    plt.figure(figsize=(14, 7.5))
    # CHANGED: Using dynamic labels
    plt.plot(x0.numpy(), overall.phi1_train, label=phi1_legend, color='blue', linewidth=2)
    plt.plot(x0.numpy(), overall.phi0_test, label=phi0_legend, color='orange', linewidth=2)
    plt.title(r'Original $\phi_0$ and $\phi_1$', fontsize=20)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 20})
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "original_phi_signals.pdf"), format='pdf', bbox_inches='tight')
    plt.show()

    # =====================================================================
    # --- Standalone Plot 2: Iterations and Final Warped phi_0 ---
    # =====================================================================
    plt.figure(figsize=(14, 7.5))
    
    # CHANGED: Using dynamic labels
    plt.plot(x0.numpy(), overall.phi1_train, label=phi1_legend, color='blue', linewidth=2, linestyle='-', alpha=1)
    plt.plot(x0.numpy(), overall.phi0_test, label=phi0_legend, color='orange', linewidth=2, linestyle='-', alpha=1)
    
    # Plot the iterations
    for batch_num, x_warped, col in phi1_est_history:
        plt.plot(x_warped, overall.phi0_test.flatten(), color=col, alpha=0.35, linewidth=1.5, label=f'Iter {batch_num}')
        
    # Plot the final iteration
    x_traj_final, _ = overall.forward_euler(beta_var, overall.phi0_test, num_steps=n_steps_beta)
    plt.plot(x_traj_final.numpy().flatten(), overall.phi0_test.flatten(), label=r'$\phi_{1,\text{est}}$', color='red', linewidth=2.5)

    plt.title(r'Progression of $\phi_0$ Warping Across Learning Iterations', fontsize=20)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 20})
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_alignment_iterations.pdf"), format='pdf', bbox_inches='tight')
    plt.show()

    return beta_var, obj_values



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



import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def optimize_and_plot_dashboard(overall, n_iters_beta=50, n_steps_beta=20, LAM=1e-4, lr=0.00675, 
                                beta_true=None, offset=4, line_stride=16, save_dir="plots", 
                                phi0_label="Start", phi1_label="Target"):
    """
    Optimizes beta to align phi0 to phi1, and displays a 3x2 dashboard:
    [0,0] Original phi0 & phi1           [0,1] Warping Characteristics
    [1,0] Warping Path Progression       [1,1] Alpha Progression
    [2,0] Objective Value Progress       [2,1] Gradient Norm Progress
    """
    os.makedirs(save_dir, exist_ok=True)
    x0 = overall.i_traj(len(overall.phi0_test))

    # Font size configurations
    TITLE_FONT = 26
    LABEL_FONT = 24
    TICK_FONT = 20
    LEGEND_FONT = 20

    # Construct legend strings
    phi1_legend = r'$\phi_1$ ({})'.format(phi1_label)
    phi0_legend = r'$\phi_0$ ({})'.format(phi0_label)

    # Initialize Beta
    np.random.seed(3)
    mean, sd = 0.001, 0.001
    beta_initial = np.random.normal(mean, sd, ((overall.proj_kernel.num_features*2)-2, 1)) 
    beta_var = tf.Variable(beta_initial, dtype=tf.float64)
    
    print(f'Learning rate: {lr}')
    optimizer = tf.optimizers.SGD(learning_rate=lr)

    # --- Initialize 3x2 subplot grid ---
    fig, ax = plt.subplots(3, 2, figsize=(20, 24))
    ax_orig      = ax[0, 0] # Top Left
    ax_warp_char = ax[0, 1] # Top Right
    ax_warp_path = ax[1, 0] # Middle Left
    ax_alpha     = ax[1, 1] # Middle Right
    ax_obj       = ax[2, 0] # Bottom Left
    ax_grad      = ax[2, 1] # Bottom Right

    # Set up dynamic plot bounds for objective and gradient
    ax_obj.set_xlim(0, n_iters_beta)
    ax_obj.set_ylim(-0.1, 0.5)
    ax_grad.set_xlim(0, n_iters_beta)
    ax_grad.set_ylim(0.0, 5.0)  

    scatter_obj = ax_obj.scatter([], [])
    line_obj,   = ax_obj.plot([], [], 'r-', linewidth=2)
    scatter_grad = ax_grad.scatter([], [])
    line_grad,   = ax_grad.plot([], [], 'b-', linewidth=2)

    obj_values, iterations, gradient_norms = [], [], []

    # =====================================================================
    # 1. Plot Top Left: Original Phi0 and Phi1
    # =====================================================================
    ax_orig.plot(x0.numpy(), overall.phi1_train, label=phi1_legend, color='blue', linewidth=3)
    ax_orig.plot(x0.numpy(), overall.phi0_test, label=phi0_legend, color='orange', linewidth=3)
    ax_orig.set_title(r'Original $\phi_0$ and $\phi_1$', fontsize=TITLE_FONT)
    ax_orig.set_xlabel('x', fontsize=LABEL_FONT)
    ax_orig.set_ylabel('Amplitude', fontsize=LABEL_FONT)

    # --- Pre-plot True Alpha if provided ---
    if beta_true is not None:
        true_alpha_vals = tf.squeeze(overall.make_func(x0, beta_true))
        ax_alpha.plot(x0.numpy(), true_alpha_vals.numpy(), label=r'$\alpha$_true', color='green', linewidth=3)

    print("\n--- Starting Optimization Loop ---")
    start_time = time.time()

    # =====================================================================
    # 2. Optimization Loop (Populating Panels [1,0], [1,1], [2,0], [2,1])
    # =====================================================================
    for batch in range(n_iters_beta):
        with tf.GradientTape() as tape:
            obj = overall.J_obj(beta_var, num_steps=n_steps_beta, LAM=LAM)

        obj_val_sq = np.squeeze(obj.numpy())
        obj_values.append(obj_val_sq)  
        iterations.append(batch)
        grad = tape.gradient(obj, [beta_var])
        optimizer.apply_gradients(zip(grad, [beta_var]))

        print(f'Iteration: {batch} | Loss: {obj_val_sq:.4f}')

        if batch % 5 == 0 or batch == n_iters_beta - 1:
            iter_color = cm.plasma(batch / max(1, n_iters_beta - 1))

            # Warping Path Progression
            x_traj, _ = overall.forward_euler(beta_var, overall.phi0_test, num_steps=n_steps_beta)
            ax_warp_path.plot(x0.numpy(), x_traj.numpy().flatten(), color=iter_color, label=f'Iter {batch}', alpha=0.7, linewidth=2)
            
            # Learned Alpha morphing
            func_vals = tf.squeeze(overall.make_func(x0, beta_var))
            ax_alpha.plot(x0.numpy(), func_vals.numpy(), color=iter_color, label=f'Iter {batch}', alpha=0.8, linewidth=2)

        # Track gradient norms
        grad_norm = tf.norm(grad[0]).numpy()
        gradient_norms.append(grad_norm)

        # Update Tracking Plots
        scatter_obj.set_offsets(np.c_[iterations, obj_values])
        line_obj.set_data(iterations, obj_values)
        scatter_grad.set_offsets(np.c_[iterations, gradient_norms])
        line_grad.set_data(iterations, gradient_norms)
        
        # Dynamically scale tracking axes
        if obj_val_sq > ax_obj.get_ylim()[1]:
            ax_obj.set_ylim(-0.1, obj_val_sq * 1.2)
        if grad_norm > ax_grad.get_ylim()[1]:
            ax_grad.set_ylim(0.0, grad_norm * 1.2)

    print(f"\nElapsed time: {time.time() - start_time:.2f} sec")

    # =====================================================================
    # 3. Post-Optimization Updates (Panel [0,1] and final Alpha)
    # =====================================================================
    # Final learned Alpha
    final_alpha_vals = tf.squeeze(overall.make_func(x0, beta_var))
    ax_alpha.plot(x0.numpy(), final_alpha_vals.numpy(), label=r'$\alpha_{\text{est}}$ (Final)', color='blue', linewidth=3)

    # Get final trajectories for characteristic plot
    x1_optimized, X_est_opt = overall.forward_euler(beta_var, overall.phi0_test, num_steps=n_steps_beta)
    num_phi0 = len(overall.phi0_test)
    x0_flat = x0.numpy().flatten()
    
    # Plot Base signals for characteristics
    ax_warp_char.plot(x0_flat, overall.phi1_train + offset, label=r'$\phi_1$ (Target) offset', color='blue', alpha=1, linewidth=3)
    ax_warp_char.plot(x0_flat, overall.phi0_test, label=r'$\phi_0$ (Start)', color='black', alpha=1, linewidth=3)

    
    # Plot final warped estimation
    final_x = X_est_opt[-1].numpy().flatten() if hasattr(X_est_opt[-1], 'numpy') else np.array(X_est_opt[-1]).flatten()
    ax_warp_char.plot(final_x, overall.phi0_test + offset, label=r'$\phi_1$,est offset', color='red', alpha=1, linewidth=3)

    # Plot thinned characteristic lines
    for j in range(0, num_phi0, line_stride):
        PHI0_array = np.full(n_steps_beta, overall.phi0_test[j])
        X_array = [X_est_opt[i][j, 0].numpy() if hasattr(X_est_opt[i], 'numpy') else X_est_opt[i][j, 0] for i in range(n_steps_beta)]
        PHI0_array_offset = PHI0_array + np.linspace(0, offset, n_steps_beta)
        ax_warp_char.plot(X_array, PHI0_array_offset, '-', color='gray', alpha=0.5, linewidth=1.5)

    # =====================================================================
    # 4. Global Titles, Labels, and Formatting
    # =====================================================================
    # Setup Titles
    ax_warp_char.set_title(f"The Characteristics: Warping Evolution", fontsize=TITLE_FONT)
    ax_warp_path.set_title('Warping Path Progression', fontsize=TITLE_FONT)
    ax_alpha.set_title(r'True $\alpha$ vs Learned $\alpha$' if beta_true is not None else r'Learned $\alpha$ Progression', fontsize=TITLE_FONT)
    ax_obj.set_title('Objective Value Progress', fontsize=TITLE_FONT)
    ax_grad.set_title(r'Gradient Norm Progress', fontsize=TITLE_FONT)

    # Setup Labels
    ax_warp_char.set_xlabel("x", fontsize=LABEL_FONT)
    ax_warp_char.set_ylabel("Amplitude (with offset)", fontsize=LABEL_FONT)
    
    ax_warp_path.set_xlabel('x(t=0)', fontsize=LABEL_FONT)
    ax_warp_path.set_ylabel('x(t) Across Iterations', fontsize=LABEL_FONT)
    ax_warp_path.plot(x0_flat, x0_flat, label='No Warping', color='lightgray', linestyle='--', linewidth=2)

    ax_alpha.set_xlabel('x', fontsize=LABEL_FONT)
    ax_alpha.set_ylabel(r'$\alpha(x)$', fontsize=LABEL_FONT)
    ax_alpha.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax_obj.set_xlabel('Iteration', fontsize=LABEL_FONT)
    ax_obj.set_ylabel('Objective Value', fontsize=LABEL_FONT)

    ax_grad.set_xlabel('Iteration', fontsize=LABEL_FONT)
    ax_grad.set_ylabel('Norm Value', fontsize=LABEL_FONT)

    # Apply global formatting to ALL axes
    for axis in ax.flatten():
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT)
        axis.grid(True, linestyle=':', alpha=0.7)
        # Place legends INSIDE the plot using loc='best' or specific safe spots
        if axis.get_legend_handles_labels()[0]: 
            axis.legend(loc='best', fontsize=LEGEND_FONT)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "full_dashboard.pdf"), format='pdf', bbox_inches='tight')
    plt.show()

    return beta_var, obj_values