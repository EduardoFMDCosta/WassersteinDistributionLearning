from .black_box import BlackBox
from .full_search import FullSearch
from .max_oracle_gradient_descent import MaxOracleGradientDescent
from .plain_vanilla import PlainVanilla
from .stochastic_vertice_ascent import StochasticVerticeAscent
from .diagonal_constrained_tp import DiagonalConstrainedTP


class GetDiscreteSolver:
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

get_discrete_solver = GetDiscreteSolver()

__all__ = [
    'get_discrete_solver'
]