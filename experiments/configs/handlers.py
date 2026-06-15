from typing import Optional, Union
import json
import argparse
import os
import torch

from wasserstein_distribution_learning.solvers import get_solver, get_discrete_solver
from wasserstein_distribution_learning.quantization import FullLearningQuantization, ConditionalLearningQuantization

_LEARNING_CLASS_MAP = {
    'full_learning': FullLearningQuantization,
    'conditional_learning': ConditionalLearningQuantization,
}
_LEARNING_CLASS_TO_STR = {v: k for k, v in _LEARNING_CLASS_MAP.items()}

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
    random_seed: int = 0,
    num_clusters: int = 10,
    method: str = 'stochastic_vertice_ascent',
    num_samples_training: Optional[int] = None,
    num_samples: int = 1000,
    wasserstein_order: int = 2,
    beta: float = 1e-6,
    plot: bool = False,
    save: bool = False,
    compute_moment_bound: bool = True, 
    compute_discrete_bound: bool = True,
    partition_type: str = 'voronoi',
    learning_type: str = 'full_learning',
):
    parser = argparse.ArgumentParser(description='Setup experiments.')
    parser.add_argument('--random_seed', type=int, default=random_seed, help='Random seed for reproducibility.')
    parser.add_argument('--distribution', type=str, default=distribution, help='Distribution to generate samples.')
    parser.add_argument('--num_dims', type=int, default=num_dims, help='Dimension of the problem.')
    parser.add_argument('--setting', type=int, default=setting, help='Experiment setting.')
    parser.add_argument('--num_samples_training', type=int, default=num_samples_training, help='Number of samples for training (i.e. defining clustering).')
    parser.add_argument('--num_samples', type=int, default=num_samples, help='Number of samples.')
    parser.add_argument('--num_clusters', type=int, default=num_clusters, help='Number of clusters (M).')
    parser.add_argument('--method', type=str, default=method, choices=get_solver.supported_methods + get_discrete_solver.supported_methods, help='Method to compute discrete-term of data-driven radius.')
    parser.add_argument('--wasserstein_order', type=int, default=wasserstein_order, help='Wasserstein order, assuming L2 norm.')
    parser.add_argument('--beta', type=float, default=beta, help='Confidence level.')
    parser.add_argument('--plot', type=bool, default=plot, help='Plot charts.')
    parser.add_argument('--save', type=bool, default=save, help='Save results.')
    parser.add_argument('--compute_moment_bound', type=bool, default=compute_moment_bound, help='Compute moment-term of data-driven radius.')
    parser.add_argument('--compute_discrete_bound', type=bool, default=compute_discrete_bound, help='Compute discrete-term of data-driven radius.')
    parser.add_argument('--partition_type', type=str, default=partition_type, choices=['voronoi', 'hyperrectangle'], help='Type of partition to use for quantization.')
    parser.add_argument('--learning_type', type=str, default=learning_type, choices=list(_LEARNING_CLASS_MAP.keys()), help='Learning mode: full distribution or conditional on bounded sets.')
    args = parser.parse_args()

    return process_args(args)

def num_samples_training_from_num_samples(N: int) -> int:
    if N < 5000:
        return 1000
    else:
        return 5000
    
def process_args(args):
    if not args.method in get_solver.supported_methods + get_discrete_solver.supported_methods:
        raise ValueError(f"Method {args.method} not supported. Supported methods: {get_solver.supported_methods}")

    args.LearningClass = _LEARNING_CLASS_MAP.get(args.learning_type, FullLearningQuantization)

    if args.num_samples_training is None:
        args.num_samples_training = num_samples_training_from_num_samples(args.num_samples)

    dynamics_params = param_handler(
        param_name="parameters",
        dataset_name=args.distribution,
        num_dims=args.num_dims,
        setting_tag=args.setting
    )

    args.__dict__.update(vars(dynamics_params))

    torch.manual_seed(args.random_seed)

    return args

