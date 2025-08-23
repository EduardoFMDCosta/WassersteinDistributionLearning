import torch
import seaborn as sns
from matplotlib import pyplot as plt
from optimization import max_oracle_gradient_descent, nested_gradient_descent, ot_lp_solver, ot_sinkhorn_solver, \
    cutting_plane

import time

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def generate_symmetric_cost(n, low, high):
    upper = (high - low) * torch.rand((n, n)) + low
    upper = torch.triu(upper, diagonal=1)

    cost = upper + upper.T # enforce symmetry
    cost.fill_diagonal_(0) # make diagonal zero
    return cost

def generate_lower_upper(empirical):

    # Generate perturbations for empirical
    rand_lower = torch.rand_like(empirical) * 0.05
    rand_upper = torch.rand_like(empirical) * 0.05

    lower = empirical - rand_lower
    upper = empirical + rand_upper

    lower = lower.clamp(min=0.0)
    upper = upper.clamp(max=1.0)

    assert lower.sum() < 1.0
    assert upper.sum() >= 1.0
    assert (lower < upper).all()

    return lower, upper

def generate_empirical(n):
    return torch.distributions.Dirichlet(0.8 * torch.ones(n)).sample()

if __name__ == '__main__':
    torch.manual_seed(0)

    n = 3
    cost = generate_symmetric_cost(n=n, low=0.5, high=0.9)
    empirical_marginal = generate_empirical(n=n)
    lower, upper = generate_lower_upper(empirical=empirical_marginal)

    print(f"Empirical = {empirical_marginal}")
    print(f"Lower = {lower}")
    print(f"Upper = {upper}")

    if n == 2:
        # Plot surface
        # Feasible interval for w1
        w1_min = max(lower[0].item(), 1 - upper[1].item())
        w1_max = min(upper[0].item(), 1 - lower[1].item())

        # Sample points
        w1_vals = torch.linspace(w1_min, w1_max, 200)
        w2_vals = 1 - w1_vals
        w = torch.stack([w1_vals, w2_vals], dim=-1)

        # Evaluate f
        f_vals = []
        for w_candidate in w:
            f_val = ot_lp_solver(cost, w_candidate, empirical_marginal)[1]
            f_vals.append(f_val)

        # Plot
        plt.plot(w1_vals, f_vals)
        plt.xlabel(r"$\omega_1$")
        plt.ylabel(r"$V(\omega)$")
        plt.show()

    for i in range(3):
        # Max oracle with LP solver
        result = cutting_plane(cost=cost,
                               lower=lower,
                               upper=upper,
                               empirical_marginal=empirical_marginal,
                               num_steps=1000,
                               lr=0.001,
                               ot_solver=ot_lp_solver)

        w_lp = result['final_w']
        obj_lp = result['objective_value']

        print(f"-------------Iteration {i+1}-------------")
        print(f"Final w (LP solver) = {w_lp}")
        print(f"Value (LP solver) = {obj_lp} \n")

        # # Max oracle with Sinkhorn algorithm
        # result = cutting_plane(cost=cost,
        #                        lower=lower,
        #                        upper=upper,
        #                        empirical_marginal=empirical_marginal,
        #                        num_steps=1000,
        #                        lr=0.001,
        #                        ot_solver=ot_sinkhorn_solver)
        #
        # w_sinkhorn = result['final_w']
        # obj_sinkhorn = result['objective_value']
        #
        # print(f"Final w (Sinkhorn) = {w_sinkhorn}")
        # print(f"Value (Sinkhorn) = {obj_sinkhorn} \n")
        #
        # # Gradient descent-ascent
        # result = nested_gradient_descent(cost=cost,
        #                                  lower=lower,
        #                                  upper=upper,
        #                                  empirical_marginal=empirical_marginal,
        #                                  num_steps=10,
        #                                  lr=0.001,
        #                                  tol=1e-8)
        #
        # w_nested = result['final_w']
        # obj_nested = result['objective_value']
        #
        # print(f"Final w (Descent-ascent) = {w_nested}")
        # print(f"Value (Descent-ascent) = {obj_nested} \n")
        #
        # # Plotting
        # sns.set_style("whitegrid")
        # sns.set_context("talk")
        # sns.set_palette("deep")
        # fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        #
        # # --- Scatter plot of w's ---
        # axes[0].scatter(w_lp[0], w_lp[1], label="LP", marker="o", s=80)
        # axes[0].scatter(w_sinkhorn[0], w_sinkhorn[1], label="Sinkhorn", marker="s", s=80)
        # axes[0].scatter(w_nested[0], w_nested[1], label="Nested", marker="^", s=80)
        # axes[0].set_xlabel(r"$\omega_1$")
        # axes[0].set_ylabel(r"$\omega_2$")
        # axes[0].set_xlim(0, 1)
        # axes[0].set_ylim(0, 1)
        # axes[0].set_aspect("equal")
        # axes[0].legend()
        #
        # # --- Bar plot of objectives ---
        # objs = [obj_lp, obj_sinkhorn, obj_nested]
        # labels = ["LP", "Sinkhorn", "Nested"]
        # axes[1].bar(labels, objs, color=["C0", "C1", "C2"])
        # axes[1].set_ylabel("Objective")
        #
        # plt.tight_layout()
        # plt.show()