from typing import Optional
import torch
import gurobipy as gp
from gurobipy import GRB

from .templates import Solver, Result
from ..quantization import UncertainQuantization

__all__ = ["NoTriangleIneq"]


class NoTriangleIneq(Solver):
    def __init__(self):
        super().__init__()
        self.verbose = False

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:
        
        cost = quantization.l2_distance_locs_to_region.pow(self.wasserstein_order)
        lower = quantization.lower_probs
        upper = quantization.upper_probs
        empirical_marginal = quantization.probs

        n = cost.size(0)

        C = cost.detach().cpu().double().numpy()  # C[j,i] = \sup_{x \in C_j} ||x - c_i||^2
        l = lower.detach().cpu().double().numpy()
        u = upper.detach().cpu().double().numpy()
        pi = empirical_marginal.detach().cpu().double().numpy()
        
        m = gp.Model("dual")
        m.Params.OutputFlag = 1 if self.verbose else 0

        m.setParam("TimeLimit", float(60))

        # μ, ν >= 0
        mu = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="mu")
        nu = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="nu")
        alpha = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="alpha")

        # ---- vals[i] = max_j scores[j, i] ----
        scores = m.addMVar((n, n), lb=-GRB.INFINITY, name="scores")
        for i in range(n):
            for j in range(n):
                m.addConstr(scores[j,i] == C[j,i] + nu[j]  - mu[j] + alpha[j] * (1.0 if j == i else 0.0), name=f"defined_scores_{j}_{i}")

        vals = m.addMVar(n, lb=-GRB.INFINITY, name="vals")
        for i in range(n):
            m.addGenConstrMax(vals[i], [scores[j, i] for j in range(n)], name=f"inner maximization {i}")

        obj = mu @ u - (nu + alpha) @ l + vals @ pi
        m.setObjective(obj, GRB.MINIMIZE)
        m.update()

        m.setParam("TimeLimit", self.time_limit if self.time_limit is not None else GRB.INFINITY)

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi did not find an optimal solution within {self.time_limit} seconds (status {m.Status})")

        return Result(bound=torch.as_tensor(m.ObjVal).pow(1 / self.wasserstein_order))