import torch

from configs.handlers import parse_arguments, pickle_dump
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=1,
        distribution="GaussianMixture",
        num_dims=3,
        setting=0,
        beta=1e-6,
        method='triangle_inequality_vertex', 
        num_samples_training=5000,
        wasserstein_order=2,
        save=True,
    )

    # We assume num_samples_training = num_samples
    # N_options = [10000]
    # M_options = [1000]

    if args.num_samples_training == 5000:
        N_options = [5000, 7500, 10000, 100000, 1000000]
        M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    elif args.num_samples_training == 1000:
        N_options = [1000, 2500]
        M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    else:
        raise ValueError("Unsupported num_samples_training")

    data_driven_radii, time_logger = data_driven_radii_for_combinations(
        args, 
        combinations=[(args.num_samples_training, N, M) for N in N_options for M in M_options], 
        time_limit=60*10,
        generate_partition_if_missing=False,
        return_all_available_combinations=True
    )

    fournier_radii = fournier_radii_for_combinations(
        args, 
        combinations=[(N, M) for N in N_options for M in M_options]
    )

    if args.save:
        pickle_dump(data_driven_radii, args.data_driven_radii_file)
        pickle_dump(fournier_radii, args.fournier_radii_file)
        pickle_dump(time_logger, args.data_driven_radii_timing_file)
