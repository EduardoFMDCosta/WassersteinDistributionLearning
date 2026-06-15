from typing import Callable, Tuple, Optional
import torch
import cvxpy as cp
import gurobipy as gp

from ..templates import DiscreteResult, DiscreteSolver

def solve_milp_min_diagonal_cvxpy(
    cost: torch.Tensor, 
    empirical_distribution: torch.Tensor, 
    lower: torch.Tensor, 
    upper: torch.Tensor
):
    n = len(empirical_distribution)

    # Decision variables
    Pi = cp.Variable((n, n), nonneg=True)
    w = cp.Variable(n)
    m = cp.Variable(n)
    b = cp.Variable(n, boolean=True)

    objective = cp.Maximize(cp.sum(cp.multiply(cost, Pi)))
    constraints = []

    # Column sums
    for j in range(n):
        constraints.append(cp.sum(Pi[:, j]) == empirical_distribution[j])

    # Row sums
    for i in range(n):
        constraints.append(cp.sum(Pi[i, :]) == w[i])

    # Bounds on w
    constraints += [w >= lower, w <= upper]

    # Big-M linearization for min(w[i], empirical_distribution[i])
    M = upper - lower  # tight big-M
    for i in range(n):
        constraints.append(m[i] <= w[i])
        constraints.append(m[i] <= empirical_distribution[i])
        constraints.append(m[i] >= w[i] - M[i] * (1 - b[i]))
        constraints.append(m[i] >= empirical_distribution[i] - M[i] * b[i])

        # Pi[i,i] constraint
        constraints.append(Pi[i, i] >= m[i])

    # w sums to 1
    constraints.append(cp.sum(w) == 1)

    # Solve MILP
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GLPK_MI)

    return prob.value, w.value

def solve_milp_min_diagonal_gurobi(
    cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    device, dtype = cost.device, cost.dtype

    # Convert to NumPy for Gurobi
    c = cost.detach().cpu().numpy()
    p = empirical_distribution.detach().cpu().numpy()
    lo = lower.detach().cpu().numpy()
    up = upper.detach().cpu().numpy()

    n = int(p.size)
    if c.shape != (n, n):
        raise ValueError(f"cost must be (n,n); got {c.shape}")
    if lo.shape != (n,) or up.shape != (n,):
        raise ValueError("lower/upper must be 1D of length n")
    M = up - lo
    if (M < 0).any():
        raise ValueError("upper must be >= lower componentwise")

    m = gp.Model("min_diagonal_milp")
    if not verbose:
        m.setParam("OutputFlag", 0)
    if time_limit is not None:
        m.setParam("TimeLimit", float(time_limit))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    # Variables
    Pi = m.addVars(n, n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="Pi")
    w  = m.addVars(n, lb=lo, ub=up, vtype=gp.GRB.CONTINUOUS, name="w")
    mm = m.addVars(n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="m")
    b  = m.addVars(n, vtype=gp.GRB.BINARY, name="b")

    # Objective: sum_{i,j} c[i,j] * Pi[i,j]
    m.setObjective(gp.quicksum(c[i, j] * Pi[i, j] for i in range(n) for j in range(n)), gp.GRB.MAXIMIZE)

    # Column sums: sum_i Pi[i,j] == p[j]
    m.addConstrs(
        (gp.quicksum(Pi[i, j] for i in range(n)) == float(p[j]) for j in range(n)),
        name="col_sums"
    )

    # Row sums: sum_j Pi[i,j] == w[i]
    m.addConstrs(
        (gp.quicksum(Pi[i, j] for j in range(n)) == w[i] for i in range(n)),
        name="row_sums"
    )

    # Big-M min linearization (elementwise)
    m.addConstrs((mm[i] <= w[i] for i in range(n)), name="m_le_w")
    m.addConstrs((mm[i] <= float(p[i]) for i in range(n)), name="m_le_p")
    m.addConstrs((mm[i] >= w[i] - float(M[i]) * (1 - b[i]) for i in range(n)), name="m_ge_w_minus_M")
    m.addConstrs((mm[i] >= float(p[i]) - float(M[i]) * b[i] for i in range(n)), name="m_ge_p_minus_Mb")

    # Diagonal constraint: Pi[i,i] >= mm[i]
    m.addConstrs((Pi[i, i] >= mm[i] for i in range(n)), name="diag_ge_m")

    # Sum w == 1
    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0, name="sum_w_eq_1")

    # Optimize
    m.setParam("TimeLimit", time_limit if time_limit is not None else gp.GRB.INFINITY)
    
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution within {time_limit} seconds (status {m.Status})")

    obj_val = float(m.ObjVal)

    # Extract w as a torch tensor on the original device/dtype
    w_sol_np = [w[i].X for i in range(n)]
    w_sol = torch.tensor(w_sol_np, device=device, dtype=dtype)

    return obj_val, w_sol
    
class DiagonalConstrainedTP(DiscreteSolver):
    def __init__(
        self, 
        mip_gap: Optional[float] = None,
        verbose: bool = False, 
        use_gurobi: bool = True
    ):
        super().__init__()

        self.mip_gap = mip_gap
        self.verbose = verbose
        self.use_gurobi = use_gurobi
    
    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> DiscreteResult:
        
        if self.use_gurobi:
            objective, w = solve_milp_min_diagonal_gurobi(
                cost=cost,
                empirical_distribution=empirical_marginal,
                lower=lower,
                upper=upper,
                time_limit=self.time_limit,
                mip_gap=self.mip_gap,
                verbose=self.verbose
            )
        else:
            objective, w = solve_milp_min_diagonal_cvxpy(
                cost=cost,
                empirical_distribution=empirical_marginal,
                lower=lower,
                upper=upper
            )
            
        return DiscreteResult(bound=torch.as_tensor(objective).pow(1 / self.wasserstein_order), w_opt=w)