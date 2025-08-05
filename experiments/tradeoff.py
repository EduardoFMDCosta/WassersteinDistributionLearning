import torch
import time
import os

from sets import KMeansPartition
from bound import data_driven_radius, fournier_radius
from plotting.plot import colored_scatter

from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="GaussianMixture",
        dimension=2,
        setting=0,
        num_samples=None,
        num_clusters=None,
        beta=1e-4,
        plot=False
    )

    beta = args.beta
    method = 'dual_sinkhorn'
    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    N_options = [1000, 5000, 10000]
    M_options = [10, 100]

    Ns, Ms, kmean_times, data_driven_times, data_driven_bounds, fournier_bounds = [], [], [], [], [], []
    for N in N_options:
        samples = distribution.sample((N,))
        for M in M_options + [N]:
            print(f"### Kmeans for: clusters (M) / num_samples (N): {M} / {N}--- ###")    
            start = time.time()
            partition = KMeansPartition(support=support_assumption, samples=samples, k=int(M))
            kmean_times.append(time.time() - start)

            print(f"### Bounding for: clusters (M) / num_samples (N): {M} / {N}--- ###")    
            start = time.time()
            data_driven_bounds.append(data_driven_radius(partition=partition, beta=beta, method=method))
            data_driven_times.append(time.time() - start)

            fournier_bounds.append(fournier_radius(partition=partition, beta=beta))

            Ns.append(N)
            Ms.append(M)

    for title, data in zip(
        ['Data-Driven Bound', 'Computation Time Kmeans', 'Computation Time Bounds', 'Fournier Bound'], 
        [data_driven_bounds, kmean_times, data_driven_times, fournier_bounds]
    ):
        colored_scatter(
            x=torch.as_tensor(Ns), 
            y=torch.as_tensor(Ms), 
            c=torch.as_tensor(data).real, 
            title=title, 
            s=200, 
            file_name=f'figures{os.sep}{title.lower().replace(" ", "_")}.png'
        )
