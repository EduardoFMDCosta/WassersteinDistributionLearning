from .joint_optimization_milp import JointOptimizationMilp
from .triangle_inequality_vertex import TriangleInequalityFromVertex
from .independent import IndependentSolver
from .discrete_solvers import get_discrete_solver
from .templates import Solver

__all__ = ['get_solver', 'get_discrete_solver', 'Solver']


class GetSolver:
    mapping = dict(
        joint_optimization_milp=JointOptimizationMilp,
        triangle_inequality_vertex=TriangleInequalityFromVertex
    )

    def __call__(self, method: str, **kwargs):
        if method in self.mapping:
            return self.mapping[method](**kwargs)
        else:
            return IndependentSolver(
                discrete_solver=get_discrete_solver(method=method, **kwargs)
            )
    
    @property
    def supported_methods(self):
        return list(self.mapping.keys())
    
    
get_solver = GetSolver()