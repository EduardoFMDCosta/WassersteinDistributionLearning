import math
from scipy.stats import beta

class Confidence:
    def __init__(self, beta: float, n_set: int, n: int):
        assert 0.0 < beta < 1.0
        assert 0 <= n_set <= n

        self.n_set = n_set
        self.n = n
        self.beta = beta
        self.empirical_proba = n_set / n

        self.lower_proba = self._get_lower_proba()
        self.upper_proba = self._get_upper_proba()

    def _get_lower_proba(self) -> float:
        pass

    def _get_upper_proba(self) -> float:
        pass

class DuchiConfidence(Confidence):
    def __init__(self, beta: float, n_set: int, n: int):
        super().__init__(beta=beta, n_set=n_set, n=n)

    def _get_epsilon(self):
        #See Proposition 2 in Duchi, 2025 (https://arxiv.org/pdf/2503.00220)

        first_term = 4/3 * math.log(1/self.beta) / self.n
        second_term = (4/3 * math.log(1/self.beta) / self.n) ** 2 + 2 * (1 - self.empirical_proba) * self.empirical_proba * math.log(1/self.beta) / self.n
        return first_term + second_term ** 0.5

    def _get_lower_proba(self):
        return max(self.empirical_proba - self._get_epsilon(), 0.0)

    def _get_upper_proba(self):
        return min(self.empirical_proba + self._get_epsilon(), 1.0)

class HoeffdingConfidence(Confidence):
    def __init__(self, beta: float, n_set: int, n: int):
        super().__init__(beta=beta, n_set=n_set, n=n)

    def _get_epsilon(self):
        return (math.log(2/self.beta) / (2 * self.n)) ** 0.5

    def _get_lower_proba(self):
        return max(self.empirical_proba - self._get_epsilon(), 0.0)

    def _get_upper_proba(self):
        return min(self.empirical_proba + self._get_epsilon(), 1.0)

class ClopperPearsonConfidence(Confidence):
    def __init__(self, beta: float, n_set: int, n: int):
        super().__init__(beta=beta, n_set=n_set, n=n)

    def _get_lower_proba(self):
        if self.n_set == 0:
            return 0.0
        else:
            return beta.ppf(self.beta / 2, self.n_set, self.n - self.n_set + 1)

    def _get_upper_proba(self):
        if self.n_set == self.n:
            return 1.0
        else:
            return beta.ppf(1 - self.beta / 2, self.n_set + 1, self.n - self.n_set)