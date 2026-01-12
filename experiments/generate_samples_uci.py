from configs.handlers import parse_arguments, pickle_dump, process_args
from configs.construct import get_distribution

SIZE = {
    "UCI-Turbine": 36_733,
    "UCI-MiniBooNE": 130_064,
}


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="UCI-MiniBooNE", # "UCI-MiniBooNE" or "UCI-Turbine"
        num_dims=50,
        setting=0,
        num_samples=1000, # PLACEHOLDER
        num_samples_training=5000,
        num_clusters=10,
        wasserstein_order=2,
        beta=1e-6,
        method='triangle_inequality_vertex',
        save=False,
        plot=False,
    )

    for seed in range(10):
        args.random_seed = seed
        args = process_args(args)
        distribution = get_distribution(**vars(args))

        partition_samples = distribution.sample(args.num_samples_training)
        pickle_dump(partition_samples, args.partition_samples_file)

        quantization_samples = distribution.sample(SIZE[args.distribution] - args.num_samples_training )
        pickle_dump(quantization_samples, args.quantization_samples_file)

        pass

