from typing import Optional
import json
import argparse
import os

dir = os.path.dirname(os.path.abspath(__file__))

def load_json(filename: str):
    _, extension = os.path.splitext(filename)
    if extension != ".json":
        filename = f"{filename}.json"
    filepath = os.path.join(dir, filename)
    with open(filepath, "r") as read_file:
        data = json.load(read_file)
    return data


def param_handler(
    param_name: str,
    dataset_name: str,
    dimension: int,
    setting_tag: Optional[int] = None
):
    params = load_json(param_name)[dataset_name]
    return argparse.Namespace(**params["dimension"][str(dimension)]["settings"][str(setting_tag)])


def parse_arguments(
    distribution: str,
    dimension: int,
    setting: int,
    num_clusters: int,
    num_samples_training: int = 1000,
    num_samples: int = 1000,
    beta: float = 1e-4,
    plot: bool = False
):
    parser = argparse.ArgumentParser(description='Setup experiments.')
    parser.add_argument('--distribution',
                        type=str,
                        default=distribution,
                        help='Distribution to generate samples.')
    parser.add_argument('--dimension',
                        type=int,
                        default=dimension,
                        help='Dimension of the problem.')
    parser.add_argument('--setting',
                        type=int,
                        default=setting,
                        help='Experiment setting.')
    parser.add_argument('--num_samples_training',
                        type=int,
                        default=num_samples_training,
                        help='Number of samples for training (i.e. defining clustering).')
    parser.add_argument('--num_samples',
                        type=int,
                        default=num_samples,
                        help='Number of samples.')
    parser.add_argument('--num_clusters',
                        type=int,
                        default=num_clusters,
                        help='Number of clusters (M).')
    parser.add_argument('--beta',
                        type=float,
                        default=beta,
                        help='Confidence level.')
    parser.add_argument('--plot',
                        type=bool,
                        default=plot,
                        help='Plot charts.')

    args = parser.parse_args()

    dynamics_params = param_handler(
        param_name="parameters",
        dataset_name=args.distribution,
        dimension=args.dimension,
        setting_tag=args.setting
    )

    args.__dict__.update(vars(dynamics_params))
    return args