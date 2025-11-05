from .joint_optimization_milp import JointOptimizationMilp
from .triangle_inequality_vertex import TriangleInequalityFromVertex
from .independent import IndependentSolver
from .discrete_solvers import get_discrete_solver
from .templates import Solver, Result

__all__ = ['get_solver', 'get_discrete_solver', 'Solver', 'Result']


class GetSolver:
    mapping = dict(
        joint_optimization_milp=JointOptimizationMilp,
        triangle_inequality_vertex=TriangleInequalityFromVertex
    )

    def __call__(
        self,
        method: str, 
        compute_discrete_bound: bool = True,
        compute_moment_bound: bool = True,
        **kwargs
    ):
        if method in self.mapping:
            solver = self.mapping[method](**kwargs)
        else:
            solver = IndependentSolver(
                discrete_solver=get_discrete_solver(method=method, **kwargs)
            )
        solver.compute_discrete_bound = compute_discrete_bound
        solver.compute_moment_bound = compute_moment_bound
        return solver
    
    @property
    def supported_methods(self):
        return list(self.mapping.keys())
    
    
get_solver = GetSolver()