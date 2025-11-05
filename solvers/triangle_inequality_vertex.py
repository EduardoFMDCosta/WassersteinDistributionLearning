import torch
import cvxpy as cp
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from optimization_utils import euclidean_projection_to_vertex, ot_lp_solver
from quantization import UncertainQuantization
from solvers.templates import Solver, Result

def lifted_lp_from_vertex_gurobi(
    cost: torch.Tensor,
    p: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    I: list,
    J: list,
    tol: float = 1e-8,
    verbose: bool = False
):
    # Convert to numpy
    cost_np = cost.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    l_np = lower.detach().cpu().numpy()
    u_np = upper.detach().cpu().numpy()

    n = len(p_np)
    mI, mJ = len(I), len(J)

    # Initialize model
    model = gp.Model()
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.FeasibilityTol = tol
    model.Params.OptimalityTol = tol
    model.Params.IntFeasTol = tol

    # Variables
    w = model.addMVar(n, lb=l_np, ub=u_np, name="w")
    s = model.addMVar(mI, lb=0.0, name="s")
    r = model.addMVar(mJ, lb=0.0, name="r")
    t = model.addMVar((mI, mJ), lb=0.0, name="t")

    # Constraints
    # s_k = w[i] - p[i] for i in I
    for k, i in enumerate(I):
        model.addConstr(s[k] == w[i] - p_np[i], name=f"s_def_{i}")
    # r_k = p[j] - w[j] for j in J
    for k, j in enumerate(J):
        model.addConstr(r[k] == p_np[j] - w[j], name=f"r_def_{j}")

    # Transport marginal constraints
    if mI > 0 and mJ > 0:
        # sum_j t[i, j] = s[i]
        model.addConstrs((gp.quicksum(t[i, j] for j in range(mJ)) == s[i] for i in range(mI)), name="row_sums")
        # sum_i t[i, j] = r[j]
        model.addConstrs((gp.quicksum(t[i, j] for i in range(mI)) == r[j] for j in range(mJ)), name="col_sums")

    # Objective construction
    obj_expr = gp.LinExpr()

    # Diagonal terms for I
    for i in I:
        obj_expr += cost_np[i, i] * w[i]

    # Constant diagonal terms for J
    const_term = sum(cost_np[j, j] * p_np[j] for j in J)

    # Transport term
    if mI > 0 and mJ > 0:
        C_IJ = cost_np[np.ix_(I, J)]
        obj_expr += gp.quicksum(C_IJ[i, j] * t[i, j] for i in range(mI) for j in range(mJ))

    # Maximize
    model.setObjective(obj_expr + const_term, GRB.MAXIMIZE)

    # Solve
    model.optimize()

    # Collect results
    result = {
        "objective": float(model.objVal) if model.status == GRB.OPTIMAL else None,
        "w": torch.tensor(w.X, dtype=torch.float64),
        "status": model.Status
    }
    return result


def lifted_lp_from_vertex_cvxpy(
    cost: torch.Tensor,
    p: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    I: list,
    J: list,
    method: str = "cvxopt",
    tol: float = 1e-8
):
    # Convert to numpy
    cost_np = cost.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    l_np = lower.detach().cpu().numpy()
    u_np = upper.detach().cpu().numpy()

    n = len(p_np)
    mI, mJ = len(I), len(J)

    # Variables
    w = cp.Variable(n)
    s = cp.Variable(mI, nonneg=True)
    r = cp.Variable(mJ, nonneg=True)
    t = cp.Variable((mI, mJ), nonneg=True)

    # Constraints
    cons = []
    cons += [w >= l_np, w <= u_np]

    # s_i = w_i - p_i  for i in I
    for k, i in enumerate(I):
        cons.append(s[k] == w[i] - p_np[i])
    # r_j = p_j - w_j for j in J
    for k, j in enumerate(J):
        cons.append(r[k] == p_np[j] - w[j])

    # transport marginal constraints
    cons += [cp.sum(t, axis=1) == s,  # row sums = s_i
             cp.sum(t, axis=0) == r]  # col sums = r_j

    # Objective
    obj_expr = 0
    # diagonal terms for I
    for i in I:
        obj_expr += cost_np[i, i] * w[i]
    # constant diagonal terms for J
    const_term = sum(cost_np[j, j] * p_np[j] for j in J)
    # transport term
    if mI > 0 and mJ > 0:
        C_IJ = cost_np[np.ix_(I, J)]
        obj_expr += cp.sum(cp.multiply(C_IJ, t))

    objective = cp.Maximize(obj_expr + const_term)

    # Problem
    prob = cp.Problem(objective, cons)
    prob.solve(solver=method, verbose=False, feastol=tol, reltol=tol, abstol=tol)

    result = {
        "objective": float(prob.value),
        "w": torch.tensor(w.value, dtype=torch.float64),
        "status": prob.status,
    }
    return result

def compute_worst_to_vertex(
    quantization: UncertainQuantization,
    vertex: torch.Tensor,
    tol: float,
    use_gurobi: bool
):
    n = vertex.shape[0]
    cost_matrix = (quantization.partition.distance_locs + quantization.partition.radii.unsqueeze(-1)).pow(2).T  # j,i # TODO: CHECK IF TRANSPOSE OR NOT

    # Identify fixed indices
    I_fixed = torch.nonzero(vertex <= quantization.lower_probs + tol).flatten().tolist()
    J_fixed = torch.nonzero(vertex >= quantization.upper_probs - tol).flatten().tolist()
    free = list(set(range(n)) - set(I_fixed) - set(J_fixed))
    if len(free) != 1:
        raise ValueError("There must be exactly one free index")
    free_idx = free[0]

    # Run the LP twice, once assuming free belongs to I, once to J
    best_obj = -float('inf')
    for free_assignment in ["I", "J"]:
        if free_assignment == "I":
            I = I_fixed + [free_idx]
            J = J_fixed
        else:
            I = I_fixed
            J = J_fixed + [free_idx]

        # Convert to lists
        I = list(I)
        J = list(J)
        if len(I) == 0 or len(J) == 0:
            continue

        if not use_gurobi:
            result = lifted_lp_from_vertex_cvxpy(cost=cost_matrix,
                                                 p=vertex,
                                                 lower=quantization.lower_probs,
                                                 upper=quantization.upper_probs,
                                                 I=I,
                                                 J=J)
        else:
            result = lifted_lp_from_vertex_gurobi(cost=cost_matrix,
                                                  p=vertex,
                                                  lower=quantization.lower_probs,
                                                  upper=quantization.upper_probs,
                                                  I=I,
                                                  J=J)
        obj_val = result["objective"]

        if obj_val > best_obj:
            best_obj = obj_val

    return torch.as_tensor(best_obj).pow(0.5)


class TriangleInequalityFromVertex(Solver):
    def __init__(
        self,
        use_gurobi: bool = True
    ):
        super().__init__()

        self.use_gurobi = use_gurobi
        self.tol = 1e-8

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:

        # Get nearest vertex to empirical
        vertex = euclidean_projection_to_vertex(w=quantization.probs, lower=quantization.lower_probs, upper=quantization.upper_probs)

        # Compute moment bound
        moment_bound = compute_worst_to_vertex(quantization=quantization, vertex=vertex, tol=self.tol, use_gurobi=self.use_gurobi)

        # Compute discrete bound
        cost_matrix = quantization.partition.distance_locs.pow(2)
        _, discrete_bound, _ = ot_lp_solver(cost=cost_matrix, w=vertex, empirical_distribution=quantization.probs)
        discrete_bound = torch.as_tensor(discrete_bound).pow(0.5)

        return Result(moment_bound=moment_bound, discrete_bound=discrete_bound)