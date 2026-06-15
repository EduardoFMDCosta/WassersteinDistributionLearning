from typing import Optional
import torch
import cvxpy as cp
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..optimization_utils import euclidean_projection_to_vertex, ot_lp_solver, sample_vertex
from ..quantization import UncertainQuantization
from .templates import Solver, Result
from .discrete_solvers.stochastic_vertice_ascent import StochasticVerticeAscent


def lifted_lp_from_vertex_gurobi(
    cost: torch.Tensor,
    p: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    I: list,
    J: list,
    time_limit: Optional[float],
    tol: float = 1e-8,
):
    cost_np = cost.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()

    n = cost_np.shape[0]

    model = gp.Model("lifted_lp")
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)  # silent

    # Variables
    Pi = model.addVars(n, n, lb=0.0, name="Pi")
    w = model.addVars(n, lb=lower_np, ub=upper_np, name="w")

    # Objective
    model.setObjective(
        gp.quicksum(cost_np[i, j] * Pi[i, j] for i in range(n) for j in range(n)),
        GRB.MAXIMIZE,
    )

    # Constraints
    for j in range(n):
        model.addConstr(gp.quicksum(Pi[i, j] for i in range(n)) == p_np[j],
                        name=f"col_{j}")

    for i in range(n):
        model.addConstr(gp.quicksum(Pi[i, j] for j in range(n)) == w[i],
                        name=f"row_{i}")

    for i in I:
        model.addConstr(Pi[i, i] >= p_np[i] - tol,
                        name=f"diag_p_{i}")

    for j in J:
        model.addConstr(Pi[j, j] >= w[j] - tol,
                        name=f"diag_w_{j}")

    model.optimize()

    result = {
        "objective": float(model.objVal) if model.status == GRB.OPTIMAL else -float('inf'),
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
        tol: float = 1e-8):
    raise NotImplementedError

def identify_sets(vertex: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, tol: float):
    n = vertex.shape[0]

    I = torch.nonzero(vertex <= lower + tol).flatten().tolist()
    J = torch.nonzero(vertex >= upper - tol).flatten().tolist()
    free = list(set(range(n)) - set(I) - set(J))

    if len(free) != 1:
        raise ValueError("vertex must correspond to a valid vertex (one free index)")
    return sorted(I), sorted(J), free[0]

def compute_bound_given_vertex(
    quantization: UncertainQuantization,
        vertex: torch.Tensor,
        wasserstein_order: int,
        tol: float,
        use_gurobi: bool, 
        time_limit: Optional[float],
):
    cost_matrix = quantization.l2_distance_locs_to_region.pow(wasserstein_order)

    I_base, J_base, free_idx = identify_sets(
        vertex=vertex,
        lower=quantization.interval.lower,
        upper=quantization.interval.upper,
        tol=tol,
    )

    # Run the LP twice, once assuming free belongs to I, once to J
    best_obj = -float('inf')
    for free_assignment in ["I", "J"]:
        if free_assignment == "I":
            I = I_base + [free_idx]
            J = J_base
        else:
            I = I_base
            J = J_base + [free_idx]

        # Convert to lists
        I = list(I)
        J = list(J)
        if len(I) == 0 or len(J) == 0:
            continue

        if not use_gurobi:
            result = lifted_lp_from_vertex_cvxpy(cost=cost_matrix,
                                                 p=vertex,
                                                 lower=quantization.interval.lower,
                                                 upper=quantization.interval.upper,
                                                 I=I,
                                                 J=J)
        else:
            result = lifted_lp_from_vertex_gurobi(cost=cost_matrix,
                                                  p=vertex,
                                                  lower=quantization.interval.lower,
                                                  upper=quantization.interval.upper,
                                                  I=I,
                                                  J=J,
                                                  time_limit=time_limit)
        obj_val = result["objective"]

        if obj_val > best_obj:
            best_obj = obj_val

    return {"bound": torch.as_tensor(best_obj)}

def compute_worst_to_vertex(
    quantization: UncertainQuantization,
        initial_vertex: torch.Tensor,
        wasserstein_order: int,
        num_iterations: int,
        tol: float,
        use_gurobi: bool,
        time_limit: Optional[float]
):

    vertex = initial_vertex

    best_obj = float(torch.inf)
    for iteration in range(num_iterations):
        # Compute value with current index sets
        result = compute_bound_given_vertex(quantization=quantization, vertex=vertex, wasserstein_order=wasserstein_order, tol=tol, use_gurobi=use_gurobi, time_limit=time_limit)
        bound = result["bound"]

        # Store if improvement
        if bound < best_obj:
            best_obj = bound

        if num_iterations > 1:
            # Heuristic to generate candidate vertex
            vertex = sample_vertex(lower=quantization.interval.lower, upper=quantization.interval.upper)

    return torch.as_tensor(best_obj).pow(1 / wasserstein_order)


class TriangleInequalityFromVertex(Solver):
    def __init__(
        self,
        use_gurobi: bool = True,
        num_iterations: int = 1,
        tol: float = 1e-8,
    ):
        super().__init__()

        self.use_gurobi = use_gurobi
        self.num_iterations = num_iterations
        self.tol = tol

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result: # TODO to be improved, sequential formulation inconcenient for setting parameters

        # Get nearest vertex to empirical
        vertex = euclidean_projection_to_vertex(
            w=quantization.probs,
            lower=quantization.interval.lower,
            upper=quantization.interval.upper,
        )

        return self.solve_for_vertex(quantization=quantization, vertex=vertex)

    def solve_for_vertex(
        self,
        quantization: UncertainQuantization,
        vertex: torch.Tensor,
    ) -> Result:
        # Compute moment bound
        moment_bound = compute_worst_to_vertex(
            quantization=quantization, 
            initial_vertex=vertex, 
            wasserstein_order=self.wasserstein_order, 
            num_iterations=self.num_iterations, 
            tol=self.tol, 
            use_gurobi=self.use_gurobi, 
            time_limit=self.time_limit
        )

        # Compute discrete bound
        cost_matrix = quantization.l2_distance_locs_to_locs.pow(self.wasserstein_order)
        _, discrete_bound, _ = ot_lp_solver(cost=cost_matrix, w=vertex, empirical_distribution=quantization.probs)
        discrete_bound = torch.as_tensor(discrete_bound).pow(1 / self.wasserstein_order)

        return Result(moment_bound=moment_bound, discrete_bound=discrete_bound)
        

class TriangleInequalityFromVertexBySVA(TriangleInequalityFromVertex):
    def __init__(
        self,
        tol: float = 1e-8,
        num_inits_sva: int = 100, 
        num_steps_sva: int = 5,
    ):
        super().__init__(use_gurobi=True, num_iterations=1, tol=tol)
        self.sva = StochasticVerticeAscent(num_inits=num_inits_sva, num_steps=num_steps_sva, verbose=False)
    
    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:
        
        vertex = self.sva.solve(    
            cost=quantization.l2_distance_locs_to_region.pow(self.wasserstein_order),
            lower=quantization.interval.lower,
            upper=quantization.interval.upper,
            empirical_marginal=quantization.probs
        ).w_opt
        return self.solve_for_vertex(quantization=quantization, vertex=vertex)

    @property
    def time_limit(self) -> Optional[float]:
        return self._time_limit
    
    @time_limit.setter
    def time_limit(self, value: Optional[float]) -> None:
        self._time_limit = value
        self.sva.time_limit = value