import torch
from matplotlib import pyplot as plt


def show_1dseqimg(
    x, dt=1.0, y_axis=None, x_label=None, ax=None, cmap=None, title=None, vmin=None, vmax=None, center_on_zero=False
):
    """Plots the evaluation of a multivariate time series through regularly-sampled time steps.

    Args:
        x (torch.Tensor): the time series to represent
        dt: the time step in the time series
        y_axis: the y axis in the visualization
        x_label: the label to use for the x axis
        ax: the ax on which to display the visualization
        cmap (str): the colormap
        title (str): the title for the visualization
        vmin: the minimum value for the color range
        vmax: the maximum value for the color range
        center_y (bool): if true, automatically sets vmin and vmax in order to center the color range on zero
    """
    y_axis = torch.arange(0, x.shape[0]) if y_axis is None else y_axis
    if center_on_zero:
        vmax_abs = torch.max(torch.abs(x))
        vmin, vmax = -vmax_abs, vmax_abs
    cmap = "viridis" if cmap is None else cmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 3))
    time_axis = torch.arange(x.shape[1]) * dt
    img = ax.pcolor(time_axis, y_axis, x, cmap=cmap, vmin=vmin, vmax=vmax)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title)
    plt.colorbar(img, ax=ax)


def show_1dseqimg_irregular_time(
    x, time_axis, y_axis=None, x_label=None, ax=None, cmap=None, title=None, vmin=None, vmax=None, center_on_zero=False
):
    """Plots the evaluation of a multivariate time series through regularly-sampled time steps.

    Args:
        x (torch.Tensor): the time series to represent
        time_axis: the set of times associated to the values of x
        y_axis: the y axis in the visualization
        x_label: the label to use for the x axis
        ax: the ax on which to display the visualization
        cmap (str): the colormap
        title (str): the title for the visualization
        vmin: the minimum value for the color range
        vmax: the maximum value for the color range
        center_y (bool): if true, automatically sets vmin and vmax in order to center the color range on zero
    """
    y_axis = torch.arange(0, x.shape[0]) if y_axis is None else y_axis
    if center_on_zero:
        vmax_abs = torch.max(torch.abs(x))
        vmin, vmax = -vmax_abs, vmax_abs
    cmap = "viridis" if cmap is None else cmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 3))
    img = ax.pcolor(time_axis, y_axis, x, cmap=cmap, vmin=vmin, vmax=vmax)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title)
    plt.colorbar(img, ax=ax)


def plot_ensemble_analysis(
    ensemble_trajectories, true_sim, sim_time_axis, time_step, n_members=None, max_members_plot=10
):
    """Comprehensive ensemble analysis plotting function.

    Args:
        ensemble_trajectories: List of ensemble member trajectories
        true_sim: True trajectory tensor
        sim_time_axis: Time axis for simulation
        time_step: Time step size
        n_members: Number of ensemble members (inferred if None)
        max_members_plot: Maximum number of members to plot in RMSE evolution
    """

    if n_members is None:
        n_members = len(ensemble_trajectories)

    # Stack ensemble trajectories for easier processing
    ensemble_trajectories_tensor = torch.stack(ensemble_trajectories)  # (n_members, time_steps, n_variables)

    # Calculate ensemble statistics
    ensemble_mean = ensemble_trajectories_tensor.mean(dim=0)  # Mean across ensemble members
    ensemble_std = ensemble_trajectories_tensor.std(dim=0)  # Std across ensemble members

    # Calculate errors for first 2 ensemble members
    ensemble_errors = []
    sub_n_members = min(n_members, 2)  # Limit to first 2 members for error plotting
    for i in range(sub_n_members):
        error = ensemble_trajectories[i] - true_sim
        ensemble_errors.append(error)

    # Calculate mean ensemble error and error spread
    ensemble_error_mean = torch.stack(ensemble_errors).mean(dim=0)
    ensemble_error_std = torch.stack(ensemble_errors).std(dim=0)

    # ========== PLOT 1: True trajectory and individual member errors ==========
    fig, axes = plt.subplots(sub_n_members + 1, 1, figsize=(8, 6))

    # Plot true trajectory for reference
    show_1dseqimg(true_sim.T, dt=time_step, ax=axes[0], title="True Trajectory (Reference)")

    # Plot errors for each ensemble member
    for i in range(sub_n_members):
        show_1dseqimg(
            ensemble_errors[i].T,
            dt=time_step,
            ax=axes[i + 1],
            title=f"Ensemble Member {i+1} Error (Member - True)",
            cmap="bwr",
            center_on_zero=True,
        )

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    # ========== PLOT 2: Mean ensemble error and error spread ==========
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))

    # Plot mean error
    show_1dseqimg(
        ensemble_error_mean.T, dt=time_step, ax=axes[0], title="Mean Ensemble Error", cmap="bwr", center_on_zero=True
    )

    # Plot error spread (standard deviation)
    show_1dseqimg(ensemble_error_std.T, dt=time_step, ax=axes[1], title="Ensemble Error Spread (Standard Deviation)")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()

    # ========== PLOT 3: RMSE evolution for all ensemble members ==========
    # Calculate RMSE for each ensemble member vs true trajectory
    rmse_ensemble = []
    for i in range(n_members):
        rmse = torch.sqrt(((ensemble_trajectories[i] - true_sim) ** 2).mean(dim=1))  # RMSE over space at each time
        rmse_ensemble.append(rmse)

    # Calculate ensemble mean RMSE
    ensemble_mean_rmse = torch.sqrt(((ensemble_mean - true_sim) ** 2).mean(dim=1))

    plt.figure(figsize=(8, 4))

    # Plot individual member RMSEs (limit number for clarity)
    n_plot = min(max_members_plot, n_members)
    for i in range(n_plot):
        alpha = 0.6 if n_plot > 5 else 0.8
        plt.plot(sim_time_axis, rmse_ensemble[i], alpha=alpha, label=f"Member {i+1}" if i < 5 else "", linewidth=1)

    # Plot ensemble mean RMSE (highlighted)
    plt.plot(sim_time_axis, ensemble_mean_rmse, label="Ensemble Mean", linewidth=3, color="black", linestyle="--")
    # Calculate ensemble spread over time
    ensemble_spread_time = ensemble_std.mean(dim=1)  # Average spread across all variables at each time
    plt.plot(sim_time_axis, ensemble_spread_time, label="Ensemble Spread", linewidth=3, color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title("RMSE Evolution: Ensemble Members vs True State")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale to better see exponential growth
    plt.show()

    # ========== STATISTICS SUMMARY ==========
    print(f"\n{'='*60}")
    print("ENSEMBLE ANALYSIS SUMMARY")
    print(f"{'='*60}")

    print(f"\nFinal RMSE values (at t={sim_time_axis[-1]:.2f}s):")
    print(f"Ensemble mean RMSE: {ensemble_mean_rmse[-1]:.4f}")
    print("Individual member RMSEs:")
    for i in range(min(5, n_members)):  # Print first 5 members
        print(f"  Member {i+1}: {rmse_ensemble[i][-1]:.4f}")
    if n_members > 5:
        print(f"  ... and {n_members-5} more members")

    # Print spread-to-error ratio
    final_spread = ensemble_spread_time[-1]
    final_rmse = ensemble_mean_rmse[-1]
    ratio = final_spread / final_rmse
    print(f"\nFinal ensemble spread: {final_spread:.4f}")
    print(f"Final ensemble mean RMSE: {final_rmse:.4f}")
    print(f"Spread/RMSE ratio: {ratio:.2f}")

    if ratio < 0.8:
        print("  → Ensemble is under-dispersed (too confident)")
    elif ratio > 1.2:
        print("  → Ensemble is over-dispersed (too uncertain)")
    else:
        print("  → Ensemble dispersion is reasonable")

    # Print error statistics at different time points
    time_points = [50, 100, 150, min(199, len(rmse_ensemble[0]) - 1)]  # Different time indices
    print("\nRMSE at different time points:")
    print(f"{'Time':<10} {'Member 1':<12} {'Member 2':<12} {'Ens. Mean':<12} {'Spread':<12}")
    print("-" * 65)

    for t_idx in time_points:
        if t_idx < len(sim_time_axis):
            t_seconds = sim_time_axis[t_idx].item()
            member1_rmse = rmse_ensemble[0][t_idx].item() if len(rmse_ensemble) > 0 else 0
            member2_rmse = rmse_ensemble[1][t_idx].item() if len(rmse_ensemble) > 1 else 0
            mean_rmse = ensemble_mean_rmse[t_idx].item()
            spread = ensemble_spread_time[t_idx].item()
            print(f"{t_seconds:<10.2f} {member1_rmse:<12.4f} {member2_rmse:<12.4f} {mean_rmse:<12.4f} {spread:<12.4f}")

    return {
        "rmse_ensemble": rmse_ensemble,
        "ensemble_mean_rmse": ensemble_mean_rmse,
        "ensemble_spread": ensemble_spread_time,
        "ensemble_errors": ensemble_errors,
        "ensemble_error_mean": ensemble_error_mean,
        "ensemble_error_std": ensemble_error_std,
    }
