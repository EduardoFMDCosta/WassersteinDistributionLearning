import medmnist
import torch
from configs.construct import EmpiricalDistribution, MinMaxNormalizer


if __name__ == '__main__':
    ds = medmnist.OCTMNIST(split='train', download=True)

    print(f"{ds.imgs.nbytes / 1024**2:.2f} MB")

    features = torch.from_numpy(ds.imgs).half()
    transform = MinMaxNormalizer().fit(features)
    dist = EmpiricalDistribution(features, transform=transform)

    bytes_used = dist.X.numel() * dist.X.element_size()
    print(f"{bytes_used / 1024**2:.2f} MB")

    pass
