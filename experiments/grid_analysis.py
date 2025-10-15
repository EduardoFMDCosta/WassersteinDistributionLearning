import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import run_combinations
import plotting.plot as plot


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="GaussianMixture",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_clusters=1000,
        beta=1e-4,
        method='stochastic_vertice_ascent',
        plot=False,
        compute_moment_bound=True,
        compute_discrete_bound=False,
    )

    N_options = [100, 1000] # [100, 1000, 5000, 10000, 50000, 100000]
    M_options = [10, 50] # [10, 50, 100]
    max_memory_mb = 2000  # 2GB memory limit for fast testing

    (quantizations, data_driven_radii, fournier_radii), (quantization_times, radius_computation_times, computation_times) = run_combinations(args, M_options=M_options, N_options=N_options, max_memory_mb=max_memory_mb)

    # Plot Bounds
    xlabel, ylabel = "Number of samples (N)", "Number of clusters (M)"

    fig, ax = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
    ax[0] = plot.plot_data_driven_radii(ax[0], data_driven_radii, field='moment_bound', ylabel=ylabel, title="Bound on Moment-Term (e1)")
    ax[1] = plot.plot_data_driven_radii(ax[1], data_driven_radii, field='discrete_bound', ylabel=ylabel, title="Bound on Discrete-Term Bound (e2)")
    ax[2] = plot.plot_data_driven_radii(ax[2], data_driven_radii, field='radius', xlabel=xlabel, ylabel=ylabel, title="Data-Driven Bound")
    plt.show()

    # Plot Computation Times
    fig, ax = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
    ax[0] = plot.plot_time_logger(ax[0], quantization_times, ylabel=ylabel, title="Quantization time")
    ax[1] = plot.plot_time_logger(ax[1], radius_computation_times, ylabel=ylabel, title="Radius computation time")
    ax[2] = plot.plot_time_logger(ax[2], computation_times, xlabel=xlabel, ylabel=ylabel, title="Total computation time")
    plt.show()