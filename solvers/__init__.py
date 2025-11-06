from .joint_optimization_milp import JointOptimizationMilp
from .triangle_inequality_vertex import TriangleInequalityFromVertex
from .independent import IndependentSolver
from .discrete_solvers import get_discrete_solver
from .templates import Solver, Result
from .no_triangle_inequality import NoTriangleIneq

__all__ = ['get_solver', 'get_discrete_solver', 'Solver', 'Result']


class GetSolver:
    mapping = dict(
        joint_optimization_milp=JointOptimizationMilp,
        triangle_inequality_vertex=TriangleInequalityFromVertex,
        no_triangle_inequality=NoTriangleIneq,
    )

    def __call__(
        self,
        method: str,
        **kwargs
    ):
        if method in self.mapping:
            solver = self.mapping[method](**kwargs)
        else:
            discrete_solver = get_discrete_solver(method=method, **kwargs)
            solver = IndependentSolver(discrete_solver=discrete_solver)
        return solver
    
    @property
    def supported_methods(self):
        return list(self.mapping.keys())
    
    
get_solver = GetSolver()