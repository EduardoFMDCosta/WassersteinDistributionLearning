import math
import torch
from scipy.stats import beta as scipy_beta

class Confidence:
    def __init__(self, beta: float, n_set: torch.Tensor, n: int):
        assert 0.0 < beta < 1.0
        assert torch.all((n_set >= 0) & (n_set <= n))

        self.n_set = n_set
        self.n = n
        self.beta = beta
        self.empirical_proba = n_set / n

        self.lower_proba = self._get_lower_proba()
        self.upper_proba = self._get_upper_proba()

    def _get_lower_proba(self) -> torch.Tensor:
        pass

    def _get_upper_proba(self) -> torch.Tensor:
        pass

    

class ClopperPearsonConfidence(Confidence):
    def __init__(self, beta: float, n_set: torch.Tensor, n: int):
        super().__init__(beta=beta, n_set=n_set, n=n)

    def _get_lower_proba(self):

        lower = torch.zeros_like(self.n_set, dtype=torch.float32)

        non_zero_mask = self.n_set > 0
        n_set_non_zero = self.n_set[non_zero_mask]

        lower_probs = torch.tensor(
            scipy_beta.ppf(self.beta / 2, n_set_non_zero.numpy(), (self.n - n_set_non_zero + 1).numpy()),
            dtype=torch.float32
        )

        lower[non_zero_mask] = lower_probs
        return lower

    def _get_upper_proba(self):

        upper = torch.ones_like(self.n_set, dtype=torch.float32)

        non_full_mask = self.n_set < self.n
        n_set_valid = self.n_set[non_full_mask]

        upper_probs = torch.tensor(
            scipy_beta.ppf(1 - self.beta / 2, (n_set_valid + 1).numpy(), (self.n - n_set_valid).numpy()),
            dtype=torch.float32
        )

        upper[non_full_mask] = upper_probs
        return upper