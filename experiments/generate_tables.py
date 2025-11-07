import os
import torch
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, TypeVar, Generic, Type

from configs.handlers import parse_arguments, pickle_load
from experiments.datastructures import Quantizations, DataDrivenRadii, FournierRadii, EmpiricalRadii

from experiments.utils import load_data


@dataclass
class DataDrivenRadiiPerMethod:
    data: Dict[str, DataDrivenRadii] = field(default_factory=dict)

    def append(self, key: str, rec: DataDrivenRadii) -> None:
        self.data[key] = rec

    def at(self, key: str) -> DataDrivenRadii:
        return self.data[key]

    def keys(self, method: Optional[str] = None) -> List[str]:
        return [key for key in self.data.keys() if (method is None or key[0] == method)]


def generate_csv(
    methods: List[str],
    data_driven_radii: DataDrivenRadiiPerMethod,
    fournier_radii: FournierRadii,
    args,
):
    rows = []
    for (N, M) in data_driven_radii.at(methods[0]).keys():
        row = dict(
            N=N,
            M=M,
        )
        if (N, M) in fournier_radii.keys():
            row["fournier"] = f"{fournier_radii.radius_at((N, M))}"

        for method in methods:
            if (N, M) in data_driven_radii.at(method).keys():
                row[method] = f"{data_driven_radii.at(method).radius_at((N, M))}"

        rows.append(row)

    # Write to CSV
    file_name = f"W{args.wasserstein_order}_{args.distribution.lower()}_num_dims={args.num_dims}_setting={args.setting}_radii.csv"
    csv_path = os.path.join(args.figures_dir, file_name)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    torch.manual_seed(0)

    methods = [
        'stochastic_vertice_ascent',
        'joint_optimization_milp',
        'diagonal_constrained_tp',
        'triangle_inequality_vertex',
        'no_triangle_inequality',
        'scalar_strategy',
    ]

    data_driven_radii = DataDrivenRadiiPerMethod()
    for method in methods:    
        args = parse_arguments(
            distribution="Gaussian",
            num_dims=2,
            setting=0,
            beta=1e-6,
            wasserstein_order=1,
            method=method,
            save=False,
        )

        data_driven_radii.append(method, load_data(args.data_driven_radii_file, DataDrivenRadii))
    
    fournier_radii = load_data(args.fournier_radii_file, FournierRadii)

    generate_csv(
        methods=methods,
        data_driven_radii=data_driven_radii,
        fournier_radii=fournier_radii,
        args=args,
    )