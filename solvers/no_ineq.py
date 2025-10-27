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

        C = cost.detach().cpu().double().numpy()
        l = lower.detach().cpu().double().numpy()
        u = upper.detach().cpu().double().numpy()
        pi = empirical_marginal.detach().cpu().double().numpy()
        
        # ---- Model ----
        m = gp.Model("dual")
        m.Params.OutputFlag = 1 if self.verbose else 0

        # ---- Variables ----
        alpha = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="alpha")
        beta = m.addMVar(n, lb=0., vtype=GRB.CONTINUOUS, name="beta")

        # ---- Gauge: fix beta[0] = 0 to remove additive freedom ----
        # m.addConstr(beta[0] == 0.0, name="gauge")

        options = m.addMVar((n, n), lb=-GRB.INFINITY, name="options")

        for i in range(n):
            for j in range(n):
                m.addConstr(options[j,i] == C[j,i] + beta[j]  - alpha[j], name=f"dual_feas_{j}_{i}")

        vals = m.addMVar(n, lb=-GRB.INFINITY, name="vals")
        for i in range(n):
            m.addGenConstrMax(vals[i], [options[j, i] for j in range(n)], name=f"inner maximization {i}")

        obj = alpha @ u - beta @ l + vals @ pi
        m.setObjective(obj, GRB.MINIMIZE)
        m.update()

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi did not find an optimal solution. status={m.Status}")

        return MaxMinLPResult(objective_opt=float(m.ObjVal))