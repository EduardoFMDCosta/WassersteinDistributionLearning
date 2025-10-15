import torch
import gurobipy as gp

from .templates import MaxMinLP, MaxMinLPResult


class BlackBox(MaxMinLP):
    def __init__(self, time_limit: int = 60):
        super().__init__()
        self.time_limit = time_limit

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> MaxMinLPResult:
        M = cost.shape[0]

        # Create model
        model = gp.Model("dual_transport")
        model.setParam("NonConvex", 2)
        model.setParam("TimeLimit", self.time_limit)

        # Decision variables
        alpha = model.addVars(M, lb=-gp.GRB.INFINITY, name="alpha")
        beta = model.addVars(M, lb=-gp.GRB.INFINITY, name="beta")
        w = model.addVars(M, lb=0.0, name="w")

        # Bounds on w
        for i in range(M):
            w[i].lb = lower[i]
            w[i].ub = upper[i]
        model.addConstr(gp.quicksum(w[i] for i in range(M)) == 1, name="sum_w")

        # Constraints
        model.addConstrs((alpha[i] + beta[j] <= cost[i, j]
                        for i in range(M) for j in range(M)), name="dual_constr")

        model.addConstr(alpha[0] == 0, name="alpha_anchor")

        # Objective
        obj = QuadExpr()
        for i in range(M):
            obj.addTerms(1.0, alpha[i], w[i])
        for j in range(M):
            obj.add(beta[j] * empirical_marginal[j])
        model.setObjective(obj, gp.GRB.MAXIMIZE)

        # Solve
        model.optimize()

        # Extract solution
        if model.status == gp.GRB.OPTIMAL  or model.status == gp.GRB.SUBOPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            w_opt = torch.tensor([w[i].X for i in range(M)])
            objective_value = model.ObjVal
            return MaxMinLPResult(objective_opt=objective_value, w_opt=w_opt)
        else:
            raise RuntimeError(f"Gurobi ended with status {model.status}")