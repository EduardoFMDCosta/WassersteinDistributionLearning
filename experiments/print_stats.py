import os

from configs.handlers import parse_arguments
from experiments.utils import load_data
from experiments.datastructures import DataDrivenRadii


if __name__ == '__main__':


    args = parse_arguments(
        distribution="Uniform",
        num_dims=2,
        setting=0,
        beta=1e-6,
        wasserstein_order=1,
        method='diagonal_constrained_tp',
        save=False,
    )

    if os.path.exists(args.data_driven_radii_file):
        data_driven_radii = load_data(args.data_driven_radii_file, DataDrivenRadii)
        print(f"Available combinations in {args.data_driven_radii_file}:")
        for (N_train, N, M) in data_driven_radii.keys():
            print(f"\t N_train={N_train}, N={N}, M={M}")
    else:
        print(f"No data found at {args.data_driven_radii_file}.")