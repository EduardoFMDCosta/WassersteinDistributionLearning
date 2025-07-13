import math
from scipy.special import comb
from scipy.optimize import brentq

class Confidence:
    def __init__(self, beta: float, n_set: int, n: int):
        assert 0.0 < beta < 1.0
        assert 0 <= n_set <= n

        self.n_set = n_set
        self.n = n
        self.beta = beta
        self.empirical_proba = n_set / n

        self.lower_proba = self.empirical_proba - self._get_epsilon()
        self.upper_proba = self.empirical_proba + self._get_epsilon()

    def _get_epsilon(self) -> float:
        pass

class DuchiConfidence(Confidence):
    def _get_epsilon(self):
        #See Proposition 2 in Duchi, 2025 (https://arxiv.org/pdf/2503.00220)

        first_term = 4/3 * math.log(1/self.beta) / self.n
        second_term = (4/3 * math.log(1/self.beta) / self.n) ** 2 + 2 * (1 - self.empirical_proba) * self.empirical_proba * math.log(1/self.beta) / self.n
        return first_term + second_term ** 0.5

class HoeffdingConfidence(Confidence):
    def __init__(self, beta: float, n_set: int, n: int):
        super().__init__(beta=beta, n_set=n_set, n=n)

    def _get_epsilon(self):
        return (math.log(2/self.beta) / (2 * self.n)) ** 0.5

class ClopperPearsonConfidence(Confidence):
    def __init__(self, beta: float, n_set: int, n: int):
        super().__init__(beta=beta, n_set=n_set, n=n)

    def _binomial_tail_prob(self, p):
        return sum(comb(self.n, i) * (p ** i) * ((1 - p) ** (self.n - i)) for i in range(self.n_set, self.n + 1))

    def _find_p(self, tol=1e-8):
        target = self.beta / 2

        def root_func(p):
            return self._binomial_tail_prob(p) - target

        p_sol = brentq(root_func, 1e-10, 1 - 1e-10, xtol=tol)
        return p_sol

    def _get_epsilon(self):
        return self.empirical_proba - self._find_p()