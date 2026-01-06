import os
import torch

from configs.handlers import parse_arguments, process_args, num_samples_training_from_num_samples
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations


# ----------------------------------------------------------
# Write LaTeX table into ../tables/
# ----------------------------------------------------------
def write_table_tex(distribution, results):
    """
    results is a list of rows:
    {
        "d": int,
        "N": int,
        "M_opt": int,
        "theorem": float,
        "proposition": float,
        "fournier": float
    }
    """

    # Prepare output directory
    tables_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "tables")
    )
    os.makedirs(tables_dir, exist_ok=True)

    filename = os.path.join(
        tables_dir, f"table_{distribution}_comparison_fournier.tex"
    )

    # Sort rows by dimension then N
    results = sorted(results, key=lambda r: (r["d"], r["N"]))

    # Formatter
    def fmt(x, is_N=False):
        """Format floats, ints, and scientific notation for N."""
        if is_N:
            s = f"{x:.0e}"  # e.g. "1e+06"
            coeff, exp = s.split("e")
            coeff = int(coeff)
            exp = int(exp)

            if coeff == 1:
                return f"$10^{{{exp}}}$"
            else:
                return f"${coeff} \\times 10^{{{exp}}}$"

        if isinstance(x, float):
            return f"${x:.3f}$"

        return f"${x}$"

    # LaTeX lines
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Summary of results for {distribution}}}")
    lines.append(f"\\label{{table:comparison-fournier-{distribution.lower()}}}")
    lines.append("\\vspace{0.3cm}")
    lines.append("\\begin{tabular}{c c c c c c}")
    lines.append("\\hline")

    # -------- Bold header row --------
    header = (
        "\\textbf{$d$} & "
        "\\textbf{$N$} & "
        "\\textbf{$M_{\\text{opt}}$} & "
        "\\textbf{Theorem $4.1$} & "
        "\\textbf{Proposition $4.2$} & "
        "\\textbf{Fournier} \\\\"
    )
    lines.append(header)
    lines.append("\\hline")

    # -------- Table contents with grouping by d --------
    previous_d = None
    for r in results:

        # Insert a thin line when d changes (but not before the first block)
        if previous_d is not None and r["d"] != previous_d:
            lines.append("\\hline")

        # Show dimension only when it changes
        d_str = fmt(r["d"]) if r["d"] != previous_d else ""

        line = (
            f"{d_str} & "
            f"{fmt(r['N'], is_N=True)} & "
            f"{fmt(r['M_opt'])} & "
            f"{fmt(r['theorem'])} & "
            f"{fmt(r['proposition'])} & "
            f"{fmt(r['fournier'])} \\\\"
        )
        lines.append(line)

        previous_d = r["d"]

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Save file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"[Saved] {filename}")


# ----------------------------------------------------------
# CORE COMPUTATION
# ----------------------------------------------------------
def run_single_setting(args):

    N_options = [1000, 10000, 100000, 1000000]
    dimensions = [2, 10, 50, 100]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0]  # TODO extend to 9 if needed

    table_results = []

    for N in N_options:
        args.num_samples = N
        N_train = num_samples_training_from_num_samples(N)
        combinations = [(N_train, N, M) for M in M_options]

        for d in dimensions:
            args.num_dims = d
            args.num_samples_training = N_train
            process_args(args)

            # Load data-driven radii
            data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
            fournier_radii = fournier_radii_for_combinations(
                args, [(N, M) for M in M_options]
            )

            # Find best M minimizing the radius
            best_val = float("inf")
            best_M = None

            for M in M_options:
                key = (N_train, N, M)
                if key in data.keys():
                    r = data.mean_radius_at(key)
                    if r < best_val:
                        best_val = r
                        best_M = M

            # Fournier bound
            fournier_key = (N, M_options[0])
            if fournier_key in fournier_radii.keys():
                fbound = float(fournier_radii.radius_at(fournier_key))
            else:
                fbound = float("inf")

            # Construct table row
            entry = {
                "d": d,
                "N": N,
                "M_opt": best_M,
                "fournier": fbound,
                "theorem": float(best_val) if args.method == "joint_diagonal_milp" else float("nan"),
                "proposition": float(best_val) if args.method == "triangle_inequality_vertex" else float("nan"),
            }

            table_results.append(entry)

    # Save final table
    write_table_tex(args.distribution.lower(), table_results)


# ----------------------------------------------------------
# RUN ALL SETTINGS
# ----------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments(
        distribution="Uniform",
        num_dims=2,
        setting=0,
        wasserstein_order=1,
        num_samples=10000,
        beta=1e-6,
        method="joint_diagonal_milp",
        plot=False,
        save=False,
    )

    settings = [
        ("Uniform", 1, "triangle_inequality_vertex"),
        ("Uniform", 1, "joint_diagonal_milp"),
        ("Gaussian", 2, "triangle_inequality_vertex"),
        ("Gaussian", 2, "joint_diagonal_milp"),
    ]

    for distribution, wasserstein_order, method in settings:
        args.distribution = distribution
        args.wasserstein_order = wasserstein_order
        args.method = method
        args = process_args(args)

        run_single_setting(args)