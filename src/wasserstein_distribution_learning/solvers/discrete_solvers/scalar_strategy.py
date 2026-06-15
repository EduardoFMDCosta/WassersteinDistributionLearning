from typing import Optional
import torch
import gurobipy as gp
from gurobipy import GRB

from ..templates import DiscreteResult, DiscreteSolver

__all__ = ["ScalarStrategy"]

class ScalarStrategy(DiscreteSolver):
    def __init__(self, strategy: str = 'worst'):
        super().__init__()
        self.verbose = False
        self.strategy = strategy

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> DiscreteResult:
        n = cost.size(0)

        C = cost.detach().cpu().double().numpy()
        l = lower.detach().cpu().double().numpy()
        u = upper.detach().cpu().double().numpy()
        pi = empirical_marginal.detach().cpu().double().numpy()

        r = 1 - l.sum()      
        
        # via addMvar ---------------------------------------------------------------------------------------------------
        # ---- Model ----
        m = gp.Model("scalar_search")
        m.Params.OutputFlag = 1 if self.verbose else 0

        # ---- Variables ----
        alpha = m.addMVar(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
        beta  = m.addMVar(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")

        # ---- Gauge: fix beta[0] = 0 to remove additive freedom ----
        m.addConstr(beta[0] == 0.0, name="gauge")

        # ---- Dual feasibility: alpha_i + beta_j <= C_ij ----
        m.addConstr(alpha[:, None] + beta[None, :] <= C, name="dual_feas")

        if self.strategy == 'worst':
            # ---- Gamma: gamma = max_i alpha_i ----
            gamma = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma")

            m.addGenConstrMax(gamma, [alpha[i] for i in range(n)], name="gamma_is_max_alpha")

            obj = alpha @ l + beta @ pi + r * gamma
        elif self.strategy == 'average':
            gamma = - alpha.sum() / n

            s = m.addMVar(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s")
            m.addConstr(s == gamma + alpha, name="gamma_is_average_alpha")

            t = m.addMVar(n, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
            for i in range(n):
                m.addGenConstrMax(t[i], [s[i], 0.0], name=f"relu[{i}]")

            obj = alpha @ l + beta @ pi + (u - l) @ t -  r * gamma 
        elif self.strategy == 'exact':
            options = m.addMVar(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="options")

            for i in range(n):
                s = m.addMVar(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"s{i}")
                m.addConstr(s == alpha - alpha[i], name=f"s_def[{i}]")

                t = m.addMVar(n, lb=0.0, vtype=GRB.CONTINUOUS, name=f"t{i}")

                for j in range(n):
                    m.addGenConstrMax(t[j], [s[j], 0.0], name=f"relu[{i},{j}]")

                m.addConstr(options[i] == (u - l) @ t  + r * alpha[i] , name=f"min_alpha_def[{i}]")

            val = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="val")
            m.addGenConstrMin(val, [options[i] for i in range(n)], name="inner minimization")
            
            obj = alpha @ l + beta @ pi + val

            # # pairwise differences and ReLU epigraph
            # s = m.addMVar((n, n), lb=-GRB.INFINITY, name="s")
            # t = m.addMVar((n, n), lb=0.0, name="t")  # t >= 0 already

            # options = m.addMVar(n, lb=-GRB.INFINITY, name="options")
            # val = m.addVar(lb=-GRB.INFINITY, name="val")

            # # s_ij = alpha_j - alpha_i (vectorized)
            # m.addConstr(s == alpha[None, :] - alpha[:, None], name="s_def")

            # # t_ij = max(s_ij, 0) via epigraph (no binaries, no genconstr)
            # m.addConstr(t >= s, name="relu_link")  # with lb(t) = 0 this yields t = max(s,0) at optimum

            # # options[i] = c^T t[i,:] + r * alpha[i]  (vectorized)
            # m.addConstr(options == t @ (u - l) + r * alpha, name="options_def")

            # # val = min_i options[i]  -> val <= options[i] for all i; maximize val tightens it to min
            # m.addConstr(val <= options, name="val_le_options")

            # # --- objective ---
            # obj = alpha @ l + beta @ pi + val
        else:
            raise NotImplementedError('Only "worst" strategy is implemented for ScalarSearch solver.')

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()

        m.setParam("TimeLimit", self.time_limit if self.time_limit is not None else GRB.INFINITY)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi did not find an optimal solution within {self.time_limit} seconds (status {m.Status})")

        # Checks -------------------------------------------------------------------------------------------------------
        # alpha_t = torch.from_numpy(alpha.X).to(dtype=lower.dtype)
        # beta_t  = torch.from_numpy(beta.X).to(dtype=lower.dtype)

        # val, w_star = o_maximization(alpha_t, lower, upper)  # val = max_{ω∈Ω} α^T ω
        # obj_val = (val + beta_t @ torch.as_tensor(empirical_marginal, dtype=lower.dtype)).item()

        # print(f"check: val + beta^T pi = {obj_val:.10f},   problem.value = {float(m.ObjVal):.10f},   gap = {obj_val - float(m.ObjVal):.3e}")

        # # # extra checks:
        # print(f"lower >= w_star: {(w_star >= lower).all()}")
        # print(f"upper <= w_star: {(w_star <= upper).all()}")
        # print(f"dual feasibility check (alpha_i + beta_j <= cost_ij): {(alpha_t.unsqueeze(1) + beta_t.unsqueeze(0) <= cost).all()}")

        # vertices = torch.stack(get_vertices(lower, upper, max_vertices=10000, tol=1e-9))
        # print(f"w_star on vertice? {torch.isclose(vertices, w_star.unsqueeze(0)).all(dim=-1).any()}")

        return DiscreteResult(bound=torch.as_tensor(m.ObjVal).pow(1 / self.wasserstein_order))