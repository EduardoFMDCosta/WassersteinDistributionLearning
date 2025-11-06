import torch

from configs.handlers import parse_arguments, pickle_dump
from experiments.utils import run_combinations


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        beta=1e-6,
        method='full_search', 
        save=True,
    )

    # We assume num_samples_training = num_samples
    M_options = [10, 15]  # [10, 25, 75, 100, 200, 500, 1000]
    N_options = [1000, 2500] #  [1000, 2500, 5000, 7500, 10000]

    # TODO add TimeLogger
    (quantizations, data_driven_radii, fournier_radii, empirical_radii), _ = run_combinations(args, M_options=M_options, N_options=N_options, compute_empirical_radii=True)

    if args.save:
        pickle_dump(quantizations, args.quantizations_file)
        pickle_dump(data_driven_radii, args.data_driven_radii_file)
        pickle_dump(fournier_radii, args.fournier_radii_file)
        pickle_dump(empirical_radii, args.empirical_radii_file)
