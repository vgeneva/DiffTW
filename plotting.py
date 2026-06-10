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
        fig4.plot(x0.numpy(), true_alpha_vals.numpy(), label=r'$\alpha$_true', color='green', linewidth=3)

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