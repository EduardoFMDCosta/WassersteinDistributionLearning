import torch
from tqdm import tqdm

from solvers.templates import DiscreteResult, DiscreteSolver
from optimization_utils import ot_lp_solver, o_maximization


__all__ = ["StochasticVerticeAscent"]


def project_to_omega_subspace(
    w: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 1000
) -> torch.Tensor:
    """Project a vector onto the capped probability simplex.

    Solve:  minimize ||y - w||_2  subject to  lower <= y <= upper (elementwise), sum(y)=1.

    The KKT conditions imply the solution has the form
        y_i = clip(w_i - λ, lower_i, upper_i)
    for a single scalar Lagrange multiplier λ enforcing the sum constraint. Define
        S(λ) = sum_i clip(w_i - λ, lower_i, upper_i).
    S is continuous and strictly decreasing in λ, so we locate λ* with bisection.

    Parameters
    ---------
    w : torch.Tensor (n,)
        Point to project.
    lower, upper : torch.Tensor (n,)
        Elementwise bounds (must satisfy lower <= upper and
        sum(lower) <= 1 <= sum(upper) for feasibility).
    tol : float
        Absolute tolerance on the sum constraint.
    max_iter : int
        Maximum bisection iterations.
    """

    # First clamp into the box; if the sum hits 1 already we're done.
    y = torch.clamp(w, min=lower, max=upper)
    s = y.sum().item()
    if abs(s - 1.0) <= tol:
        final_y = y
    else:
        # Establish bisection interval [low, high] for λ.
        # Let low = min_i (w_i - upper_i). Then for every i: w_i - low >= upper_i, so after clipping y = upper and S(low)=sum(upper).
        # Let high = max_i (w_i - lower_i). Then for every i: w_i - high <= lower_i, so after clipping y = lower and S(high)=sum(lower).
        # Feasibility (sum(lower) <= 1 <= sum(upper)) guarantees the root λ* with S(λ*)=1 lies in [low, high].
        low = float(torch.min(w - upper).item())    # S(low) = sum(upper) ≥ 1
        high = float(torch.max(w - lower).item())   # S(high) = sum(lower) ≤ 1

        # Adaptive effective tolerance: don't demand more than floating precision permits.
        effective_tol = max(tol, torch.finfo(w.dtype).eps * w.numel())

        final_y = None
        best_y = None
        best_res = float('inf')

        # Bisection on S(λ) - 1 = 0. S decreases with λ.
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            y = torch.clamp(w - mid, min=lower, max=upper)
            s = y.sum().item()
            res = abs(s - 1.0)

            # Track best iterate always
            if res < best_res:
                best_res = res
                best_y = y

            # Terminate on residual or bracket size.
            if res <= effective_tol:
                final_y = y
                break

            if s > 1.0:
                # Sum too large ⇒ λ too small ⇒ increase lower bound
                low = mid
            else:
                # Sum too small ⇒ λ too large ⇒ decrease upper bound
                high = mid

        if final_y is None:
            # Fall back to best iterate obtained
            final_y = best_y
            # Only raise if we are *far* outside user-requested tolerance.
            if best_res > effective_tol:
                raise RuntimeError(
                    f"Bisection did not reach tolerance: residual={best_res:.3e}, interval=({low:.3e},{high:.3e}), tol={tol:.1e}, effective_tol={effective_tol:.1e}"
                )

        # Use effective_tol for internal assertion; still ensure within a modest multiple of user tol
        assert abs(final_y.sum() - 1.0) <= effective_tol
        assert (final_y >= lower).all() & (final_y <= upper).all() 

    return final_y


class StochasticVerticeAscent(DiscreteSolver):
    def __init__(
        self, 
        num_inits: int = 1000,
        num_steps: int = 1000, 
        verbose: bool = False
    ):
        super().__init__()
        self.num_inits = num_inits
        self.num_steps = num_steps
        self.verbose = verbose

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> DiscreteResult:
        M = cost.shape[0]
        delta = 1e-3

        objective_opt = -float("inf")
        w_opt = None

        pbar = tqdm(total=self.num_steps, desc="Cutting Plane Outer Loop")

        for d in range(self.num_inits):
            msg = f"[CuttingPlane] iteration={d}"
            pbar.set_postfix_str(msg)

            # Initialize w
            w = empirical_marginal.clone() + torch.rand_like(empirical_marginal) * delta
            w = project_to_omega_subspace(w=w, lower=lower, upper=upper, max_iter=10_000)

            for step in range(self.num_steps):
                # Solve for primal (w^{(k)}) and dual (alpha^{(k)}, beta^{(k)})
                Pi, objective, duals = ot_lp_solver(cost=cost, w=w, empirical_distribution=empirical_marginal)
                alpha, beta = duals

                alpha = alpha.float()
                beta = beta.float()

                # O-maximization (w^{(k+1)})
                _, w_next = o_maximization(cost=alpha, lower=lower, upper=upper)

                epsilon = torch.einsum('i,i->', alpha, w_next - w)
                msg_inner = f"[Inner] step={step+1}, epsilon={epsilon:.2e}, objective={objective:.4f}"
                pbar.set_postfix_str(msg + ", " + msg_inner)

                if epsilon < 1e-5:
                    if self.verbose:
                        pbar.write(f"{msg} converged after {step + 1} iterations. Final objective: {objective:.4f}")
                    break

                # Update w
                w = w_next

            #Update highest objective
            if objective_opt < objective:
                objective_opt = objective
                w_opt = w

            pbar.update(1)

        pbar.close()

        return DiscreteResult(bound=torch.as_tensor(objective_opt).pow(1 / self.wasserstein_order), w_opt=w_opt)
    

