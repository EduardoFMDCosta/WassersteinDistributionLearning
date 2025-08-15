import torch

from optimization import max_oracle_gradient_descent


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
    return torch.distributions.Dirichlet(torch.ones(n)).sample()

if __name__ == '__main__':
    torch.manual_seed(0)

    n = 10
    cost = generate_symmetric_cost(n=n, low=0.1, high=2.0)
    empirical_marginal = generate_empirical(n=n)
    lower, upper = generate_lower_upper(empirical=empirical_marginal)

    for i in range(10):
        result = max_oracle_gradient_descent(cost=cost,
                                            lower=lower,
                                            upper=upper,
                                            empirical_marginal=empirical_marginal,
                                            num_steps=1000,
                                            lr=0.01,
                                            tol=1e-6)

        print(f"-------------Iteration {i+1}-------------")
        print(f"Initial w = {result['initial_w']}")
        print(f"Final w = {result['final_w']}")
        print(f"Value = {result['objective_value']} \n")