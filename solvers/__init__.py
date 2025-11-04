from .templates import MaxMinLP
from .joint_optimization_milp import JointOptimizationMilp
from .separate_optimization_milp import SeparateOptimizationMilp
from .triangle_inequality_vertex import TriangleInequalityFromVertex

__all__ = ['get_solver']


class GetSolver:
    mapping = dict(
        joint_optimization_milp=JointOptimizationMilp,
        separate_optimization_milp=SeparateOptimizationMilp,
        triangle_inequality_vertex=TriangleInequalityFromVertex
    )

    def __call__(self, method: str, **kwargs):
        if method not in self.mapping:
            raise ValueError('Unknown optimization method.')
        return self.mapping[method](**kwargs)
    
    @property
    def supported_methods(self):
        return list(self.mapping.keys())
    
    
get_solver = GetSolver()