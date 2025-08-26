import torch
import numpy as np
from matplotlib import pyplot as plt
from optimization import ot_lp_solver, get_omega_space_vertices, cutting_plane, full_search, plain_vanilla_upperbound

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def sample_feasible_points(lower, upper, n_samples=100):
    dim = len(lower)
    points = []
    while len(points) < n_samples:
        # sample from Dirichlet (uniform on simplex)
        x = np.random.dirichlet(np.ones(dim))
        if np.all(x >= lower.numpy()) and np.all(x <= upper.numpy()):
            points.append(x)
    return points

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

    M = 6
    cost = generate_symmetric_cost(n=M, low=0.5, high=1.5)
    empirical_marginal = generate_empirical(n=M)
    lower, upper = generate_lower_upper(empirical=empirical_marginal)

    plot = True

    print(f"Empirical = {empirical_marginal}")
    print(f"Lower = {lower}")
    print(f"Upper = {upper} \n")

    # Analysis 1: If M = 3, plot simplex vertices and linear pieces
    if M == 3 and plot:
        vertices = get_omega_space_vertices(lower, upper)

        # compute f
        f_vals, alphas = [], []
        for v in vertices:
            Pi, objective, duals = ot_lp_solver(cost=cost, w=v, empirical_distribution=empirical_marginal)
            alpha, beta = duals
            f_vals.append(objective)
            alphas.append(alpha / 100) # rescale for visualization

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        verts_np = np.array([v for v in vertices])
        ax.scatter(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2],
                   c=f_vals, cmap='viridis', s=10, label="Vertices")

        # annotate vertices with f values
        for v, val in zip(vertices, f_vals):
            ax.text(v[0].item(), v[1].item(), v[2].item(), f"{val:.4f}", fontsize=8)

        # plot alpha vectors from each vertex
        for v, alpha in zip(vertices, alphas):
            v_np = np.array(v)
            alpha_np = np.array(alpha)
            ax.quiver(
                v_np[0], v_np[1], v_np[2],
                alpha_np[0], alpha_np[1], alpha_np[2],
                color='black', arrow_length_ratio=0.1, linewidth=1.5
            )

        ax.set_xlabel(r"$\omega_1$")
        ax.set_ylabel(r"$\omega_2$")
        ax.set_zlabel(r"$\omega_3$")
        ax.view_init(elev=35.26, azim=45)
        plt.show()

        # sample random points inside feasible region
        rand_points = sample_feasible_points(lower, upper, n_samples=3000)

        # compute f
        rand_vals, rand_alphas = [], []
        for v in rand_points:
            Pi, objective, duals = ot_lp_solver(cost=cost, w=torch.tensor(v), empirical_distribution=empirical_marginal)
            alpha, beta = duals

            rand_vals.append(objective)
            rand_alphas.append(alpha / 100) # rescale for visualization

        # Merge lists
        vertices = [row for row in vertices.numpy()]  # Convert to numpy for compatibility
        vertices = vertices + rand_points
        f_vals = f_vals + rand_vals
        alphas = alphas + rand_alphas

        # Convert to numpy
        points = np.array(vertices)
        alphas = np.array(alphas)

        # Find unique alphas and map each to an index
        unique_alphas, inverse_indices = np.unique(alphas, axis=0, return_inverse=True)

        # Assign colors to each unique alpha
        cmap = plt.get_cmap("tab10")  # or any colormap
        colors = cmap(inverse_indices % cmap.N)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=10)

        ax.set_xlabel(r"$\omega_1$")
        ax.set_ylabel(r"$\omega_2$")
        ax.set_zlabel(r"$\omega_3$")
        ax.view_init(elev=35.26, azim=45)
        plt.show()
        plt.show()


    # Analysis 2: show that V is convex on w
    if M == 2 and plot:
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


    # Analysis 3: Compute optima for different starting points (if applicable)
    for i in range(1):
        # Cutting plane LP solver
        result = cutting_plane(cost=cost,
                        lower=lower,
                        upper=upper,
                        empirical_marginal=empirical_marginal,
                        num_steps=1000,
                        lr=0.001,
                        ot_solver=ot_lp_solver)

        print(f"Final w (Cutting plane) = {result['w_opt']}")
        print(f"Value (Cutting plane) = {result['objective_opt']} \n")

        result = full_search(cost=cost,
                        lower=lower,
                        upper=upper,
                        empirical_marginal=empirical_marginal,
                        num_steps=1000,
                        lr=0.001,
                        ot_solver=ot_lp_solver)

        print(f"Final w (Full search) = {result['w_opt']}")
        print(f"Value (Full search) = {result['objective_opt']} \n")

        result = plain_vanilla_upperbound(cost=cost,
                        lower=lower,
                        upper=upper,
                        empirical_marginal=empirical_marginal)

        print(f"Final w (Plain vanilla) = {result['w_opt']}")
        print(f"Value (Plain vanilla) = {result['objective_opt']} \n")