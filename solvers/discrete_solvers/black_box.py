import torch
import gurobipy as gp

from solvers.templates import DiscreteResult, DiscreteSolver


class BlackBox(DiscreteSolver):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> DiscreteResult:
        M = cost.shape[0]

        # Create model
        model = gp.Model("dual_transport")
        model.setParam("NonConvex", 2)
        model.setParam("TimeLimit", self.time_limit if self.time_limit is not None else gp.GRB.INFINITY)

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
        if model.status == gp.GRB.OPTIMAL  or model.status == gp.GRB.SUBOPTIMAL or model.status == gp.GRB.TIME_LIMIT: # TODO remove acceptance of SUBOPTIMAL and TIME_LIMIT solutions
            w_opt = torch.tensor([w[i].X for i in range(M)])
            objective_value = model.ObjVal
            return DiscreteResult(bound=torch.as_tensor(objective_value).pow(1 / self.wasserstein_order), w_opt=w_opt)
        else:
            raise RuntimeError(f"Gurobi ended with status {model.status}")