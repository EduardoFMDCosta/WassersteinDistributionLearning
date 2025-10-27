from .templates import MaxMinLP
from .full_search import FullSearch
from .stochastic_vertice_ascent import StochasticVerticeAscent
from .plain_vanilla import PlainVanilla
from .diagonal_constrained_tp import DiagonalConstrainedTP
from .max_oracle_gradient_descent import MaxOracleGradientDescent
from .black_box import BlackBox
from .no_ineq import NoIneq

__all__ = ['get_solver']


class GetSolver:
    mapping = dict(
        full_search=FullSearch,
        stochastic_vertice_ascent=StochasticVerticeAscent,
        plain_vanilla=PlainVanilla,
        diagonal_constrained_tp=DiagonalConstrainedTP,
        max_oracle_gradient_descent=MaxOracleGradientDescent,
        black_box=BlackBox
    )

    def __call__(self, method: str, **kwargs):
        if method not in self.mapping:
            raise ValueError('Unknown optimization method.')
        return self.mapping[method](**kwargs)
    
    @property
    def supported_methods(self):
        return list(self.mapping.keys())
    
    
get_solver = GetSolver()