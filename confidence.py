import torch
import math
from scipy.special import comb
from scipy.optimize import brentq

class DuchiConfidence:
    def __init__(self,
                 beta: float,
                 n_set: int,
                 n: int):

        assert 0.0 < beta < 1.0
        assert 0 <= n_set <= n

        self.beta = beta
        self.empirical_proba = n_set / n
        self.n = n

        self.epsilon = self._get_epsilon()

    def _get_epsilon(self):
        #See Proposition 2 in Duchi, 2025 (https://arxiv.org/pdf/2503.00220)

        first_term = 4/3 * math.log(1/self.beta) / self.n
        second_term = (4/3 * math.log(1/self.beta) / self.n) ** 2 + 2 * (1 - self.empirical_proba) * self.empirical_proba * math.log(1/self.beta) / self.n
        return first_term + second_term ** 0.5

class HoeffdingConfidence:
    def __init__(self,
                 beta: float,
                 n_set: int,
                 n: int):

        assert 0.0 < beta < 1.0
        assert 0 <= n_set <= n

        self.beta = beta
        self.empirical_proba = n_set / n
        self.n = n

        self.epsilon = self._get_epsilon()

    def _get_epsilon(self):
        return (math.log(2/self.beta) / (2 * self.n)) ** 0.5