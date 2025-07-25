import torch
from sets import HyperRectangle
from sklearn.cluster import KMeans

def k_means_cluster(samples: torch.Tensor, k: int):
    # Convert to numpy for sklearn
    samples_np = samples.cpu().numpy()

    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(samples_np)
    labels = kmeans.labels_

    return labels

def cluster_regions(samples: torch.Tensor, labels: torch.Tensor):

    k = labels.max() + 1
    for cluster_id in range(k):
        # Get samples in this cluster
        cluster_samples = samples[labels == cluster_id]

        if cluster_id == 0:
            lower = cluster_samples.min(dim=0).values
            upper = cluster_samples.max(dim=0).values
        else:
            lower = torch.stack([lower, cluster_samples.min(dim=0).values], dim=0)
            upper = torch.stack([upper, cluster_samples.max(dim=0).values], dim=0)

    if k == 1:
        regions = HyperRectangle(lower=lower.unsqueeze(dim=0), upper=upper.unsqueeze(dim=0))
    else:
        regions = HyperRectangle(lower=lower, upper=upper)
    return regions