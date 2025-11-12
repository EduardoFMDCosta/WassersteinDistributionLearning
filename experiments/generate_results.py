import torch

from configs.handlers import parse_arguments, pickle_dump
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        beta=1e-6,
        method='joint_optimization_milp', 
        wasserstein_order=2,
        save=True,
    )

    # We assume num_samples_training = num_samples
    # N_options = [1000]
    # M_options = [5, 20]

    # N_options = [1000, 2500, 5000, 7500, 10000]
    # M_options = [150, 200, 300, 500, 1000]

    N_options = [1000, 2500, 5000, 7500, 10000, 25000]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 2000]

    combinations = [(N, M) for N in N_options for M in M_options]

    # (quantizations, data_driven_radii), _ = data_driven_radii_for_combinations(
    #     args, 
    #     combinations=combinations, 
    #     time_limit=60*5, 
    #     generate_partition_if_missing=False
    # )

    fournier_radii = fournier_radii_for_combinations(args, combinations)

    if args.save:
        # pickle_dump(data_driven_radii, args.data_driven_radii_file)
        pickle_dump(fournier_radii, args.fournier_radii_file)
