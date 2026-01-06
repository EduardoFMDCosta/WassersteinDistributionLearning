import os
import torch

from configs.handlers import parse_arguments, process_args, num_samples_training_from_num_samples
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations


def format_val(x):
    """Format floats as $x.xxx$ or '--'."""
    if x is None:
        return "--"
    return f"${x:.3f}$"


def compute_radius(args, N, M, d, random_seed_options):
    """Compute mean radius for the current args.method."""
    N_train = num_samples_training_from_num_samples(N)
    combinations = [(N_train, N, M)]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    key = (N_train, N, M)

    if key not in data.keys():
        return None
    return float(data.mean_radius_at(key))


def main_table():
    N_options = [1000, 5000]
    dimensions = [2, 10, 25]
    M_options = [5, 20, 30]
    random_seed_options = list(range(10))

    settings = [
        ('Uniform', 1),
        ('Gaussian', 2)
    ]

    out = {}  # distribution -> rows

    for distribution, wasserstein_order in settings:
        table_rows = []

        for d in dimensions:
            for N in N_options:
                N_train = num_samples_training_from_num_samples(N)

                # Compute Fournier radii
                dummy_args = parse_arguments(
                    distribution=distribution,
                    wasserstein_order=wasserstein_order,
                    num_dims=d,
                    num_samples_training=N_train,
                    num_samples=N,
                    method="triangle_inequality_vertex",
                    beta=1e-6,
                    plot=False,
                    save=False,
                    setting=0
                )
                dummy_args = process_args(dummy_args)

                fournier_radii = fournier_radii_for_combinations(
                    dummy_args,
                    [(N, M) for M in M_options]
                )

                for M in M_options:

                    # Theorem 4.1
                    args_jd = parse_arguments(
                        distribution=distribution,
                        wasserstein_order=wasserstein_order,
                        num_dims=d,
                        num_samples_training=N_train,
                        num_samples=N,
                        method="joint_diagonal_milp",
                        beta=1e-6,
                        plot=False,
                        save=False,
                        setting=0
                    )
                    args_jd = process_args(args_jd)
                    radius_jd = compute_radius(args_jd, N, M, d, random_seed_options)

                    # Proposition 5.2
                    args_ti = parse_arguments(
                        distribution=distribution,
                        wasserstein_order=wasserstein_order,
                        num_dims=d,
                        num_samples_training=N_train,
                        num_samples=N,
                        method="triangle_inequality_vertex",
                        beta=1e-6,
                        plot=False,
                        save=False,
                        setting=0
                    )
                    args_ti = process_args(args_ti)
                    radius_ti = compute_radius(args_ti, N, M, d, random_seed_options)

                    # Fournier
                    key_fournier = (N, M)
                    if key_fournier in fournier_radii.keys():
                        fournier_val = float(fournier_radii.radius_at((N_train, N, M_options[0])))
                    else:
                        fournier_val = None

                    table_rows.append(dict(
                        d=d,
                        N=N,
                        M=M,
                        theorem_41=radius_jd,
                        proposition_52=radius_ti,
                        fournier=fournier_val
                    ))

        # sorting: by d, then N, then M
        table_rows.sort(key=lambda r: (r['d'], r['N'], r['M']))

        out[distribution] = table_rows

    return out


def make_latex_table_per_distribution(distribution, rows):
    """
    Create one LaTeX table with multirow grouping for each distribution.
    """

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{r r r r r r}\n"
        "\\hline\n"
        "$d$ & $N$ & $M$ & Theorem 4.1 & Proposition 5.2 & Fournier \\\\\n"
        "\\hline\n"
    )

    body = ""

    # group rows by (d, N)
    from itertools import groupby

    # group by d
    for d_val, d_group in groupby(rows, key=lambda r: r["d"]):
        d_group = list(d_group)

        # inside each d, group by N
        for N_val, dn_group in groupby(d_group, key=lambda r: r["N"]):
            dn_group = list(dn_group)

            num_rows = len(dn_group)
            first = True

            for row in dn_group:
                M = row["M"]
                t41 = format_val(row["theorem_41"])
                p52 = format_val(row["proposition_52"])
                fval = format_val(row["fournier"])

                if first:
                    body += (
                        f"\\multirow{{{num_rows}}}{{*}}{{$ {d_val} $}} & "
                        f"\\multirow{{{num_rows}}}{{*}}{{$ {N_val} $}} & "
                        f"${M}$ & {t41} & {p52} & {fval} \\\\\n"
                    )
                    first = False
                else:
                    body += f" & & ${M}$ & {t41} & {p52} & {fval} \\\\\n"

    footer = "\\hline\n\\end{tabular}\n"
    footer += f"\\caption{{Bounds comparison table for {distribution}.}}\n"
    footer += "\\end{table}\n"

    return header + body + footer


if __name__ == "__main__":
    all_tables = main_table()

    for distribution, rows in all_tables.items():
        latex_code = make_latex_table_per_distribution(distribution, rows)
        print(f"% ---------- TABLE FOR {distribution} ----------")
        print(latex_code)
        print("\n\n")