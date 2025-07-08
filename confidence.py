import torch

class DuchiConfidence:
    def __init__(self,
                 delta: torch.Tensor,
                 alpha: torch.Tensor,
                 num_samples: int):

        assert (0.0 < delta < 1.0).all()
        assert (0.0 <= alpha <= 1.0).all()

        self.delta = delta
        self.alpha = alpha
        self.num_samples = num_samples

        self.gamma = self._get_gamma()

    def _get_gamma(self):
        #See Proposition 2 in Duchi, 2025 (https://arxiv.org/pdf/2503.00220)

        first_term = 4/3 * torch.log(1/self.delta) / self.num_samples
        second_term = (4/3 * torch.log(1/self.delta) / self.num_samples) ** 2 + 2 * self.alpha * (1 - self.alpha) * torch.log(1/self.delta) / self.num_samples

        return first_term + second_term ** 0.5

class HoeffdingConfidence:
    def __init__(self,
                 delta: torch.Tensor,
                 alpha: torch.Tensor,
                 num_samples: int):

        assert (0.0 < delta < 1.0).all()
        assert (0.0 <= alpha <= 1.0).all()

        self.delta = delta
        self.alpha = alpha
        self.num_samples = num_samples

        self.gamma = self._get_gamma()

    def _get_gamma(self):
        return (torch.log(2/self.delta) / (2 * self.num_samples)) ** 0.5