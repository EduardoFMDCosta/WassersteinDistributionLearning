import torch
import gurobipy as gp
from gurobipy import GRB

from .templates import MaxMinLP, MaxMinLPResult

__all__ = ["NoIneq"]

class NoIneq(MaxMinLP):
    def __init__(self):
        super().__init__()
        self.verbose = False

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> MaxMinLPResult:
        n = cost.size(0)

        C = cost.detach().cpu().double().numpy()  # C[j,i] = \sup_{x \in C_j} ||x - c_i||^2
        l = lower.detach().cpu().double().numpy()
        u = upper.detach().cpu().double().numpy()
        pi = empirical_marginal.detach().cpu().double().numpy()
        
        m = gp.Model("dual")
        m.Params.OutputFlag = 1 if self.verbose else 0

        # μ, ν >= 0
        mu = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="mu")
        nu = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="nu")

        # ---- vals[i] = max_j scores[j, i] ----
        scores = m.addMVar((n, n), lb=-GRB.INFINITY, name="scores")
        for i in range(n):
            for j in range(n):
                m.addConstr(scores[j,i] == C[j,i] + nu[j]  - mu[j], name=f"defined_scores_{j}_{i}")

        vals = m.addMVar(n, lb=-GRB.INFINITY, name="vals")
        for i in range(n):
            m.addGenConstrMax(vals[i], [scores[j, i] for j in range(n)], name=f"inner maximization {i}")

        obj = mu @ u - nu @ l + vals @ pi
        m.setObjective(obj, GRB.MINIMIZE)
        m.update()

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi did not find an optimal solution. status={m.Status}")

        return MaxMinLPResult(objective_opt=float(m.ObjVal))