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
    num_dims: int,
    setting_tag: Optional[int] = None
):
    params = load_json(param_name)[dataset_name]
    return argparse.Namespace(**params["num_dims"][str(num_dims)]["settings"][str(setting_tag)])


def parse_arguments(
    distribution: str,
    num_dims: int,
    setting: int,
    num_clusters: int,
    method: str = 'stochastic_vertice_ascent',
    num_samples_training: int = 1000,
    num_samples: int = 1000,
    beta: float = 1e-4,
    plot: bool = False,
    save: bool = False,
    compute_moment_bound: bool = True, 
    compute_discrete_bound: bool = True
):
    parser = argparse.ArgumentParser(description='Setup experiments.')
    parser.add_argument('--distribution',
                        type=str,
                        default=distribution,
                        help='Distribution to generate samples.')
    parser.add_argument('--num_dims',
                        type=int,
                        default=num_dims,
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
    parser.add_argument('--method',
                        type=str,
                        default=method,
                        help='Method to compute discrete-term of data-driven radius.')
    parser.add_argument('--beta',
                        type=float,
                        default=beta,
                        help='Confidence level.')
    parser.add_argument('--plot',
                        type=bool,
                        default=plot,
                        help='Plot charts.')
    parser.add_argument('--save',
                        type=bool,
                        default=save,
                        help='Save results.')
    parser.add_argument('--compute_moment_bound',
                        type=bool,
                        default=compute_moment_bound,
                        help='Compute moment-term of data-driven radius.')
    parser.add_argument('--compute_discrete_bound',
                        type=bool,
                        default=compute_discrete_bound,
                        help='Compute discrete-term of data-driven radius.')

    args = parser.parse_args()

    dynamics_params = param_handler(
        param_name="parameters",
        dataset_name=args.distribution,
        num_dims=args.num_dims,
        setting_tag=args.setting
    )

    args.__dict__.update(vars(dynamics_params))
    args.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    return args