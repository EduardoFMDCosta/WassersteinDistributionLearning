import os
import itertools
import torch
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from configs.handlers import parse_arguments, load_json
from experiments.datastructures import DataDrivenRadii, FournierRadii

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
    for (N_train, N, M) in data_driven_radii.at(methods[0]).keys():
        row = dict(
            N_train=N_train,
            N=N,
            M=M,
        )
        if (N_train, N, M) in fournier_radii.keys():
            row["fournier"] = f"{fournier_radii.radius_at((N_train, N, M))}"
        else:
            row["fournier"] = "N/A"

        for method in methods:
            if method in data_driven_radii.keys() and (N_train, N, M) in data_driven_radii.at(method).keys():
                row[method] = f"{data_driven_radii.at(method).radius_at((N_train, N, M))}"
            else:
                row[method] = "N/A"

        rows.append(row)

    # Write to CSV
    file_name = f"W{args.wasserstein_order}_{args.distribution.lower()}_num_dims={args.num_dims}_setting={args.setting}_radii.csv"
    csv_path = os.path.join(args.figures_dir, file_name)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    methods = [
        'joint_optimization_milp',
         'joint_full_expansion_milp', 
         'diagonal_constrained_tp', 
         'triangle_inequality_vertex',
    ]

    params = load_json("parameters")
    settings = [(d, int(n), int(s)) for d in params.keys() for n in params[d]["num_dims"].keys() for s in params[d]["num_dims"][n]["settings"].keys()]

    for (distribution, num_dims, setting), wasserstein_order in itertools.product(settings, [1,2]):
        try:
            data_driven_radii = DataDrivenRadiiPerMethod()
            for method in methods:    
                args = parse_arguments(
                    distribution=distribution,
                    num_dims=num_dims,
                    setting=setting,
                    beta=1e-6,
                    wasserstein_order=wasserstein_order,
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
            print(f"Generated CSV for distribution={distribution}, num_dims={num_dims}, setting={setting}, wasserstein_order={wasserstein_order}")

        except Exception as e:
            print(f"Failed to generate CSV for distribution={distribution}, num_dims={num_dims}, setting={setting}, wasserstein_order={wasserstein_order}: {e}")

        