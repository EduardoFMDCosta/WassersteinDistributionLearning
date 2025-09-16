import torch

class PolynomialDecayScheduler:
    def __init__(self, base_lr=1.0, p=0.75):
        """
        Learning rate schedule: lr_t = base_lr / (t+1)^p
        with p in (0.5, 1] ensures:
          - lr_t not summable (sum diverges)
          - lr_t^2 summable (sum converges)
        """
        assert 0.5 < p <= 1.0, "p must be in (0.5, 1] for conditions to hold"
        self.base_lr = base_lr
        self.p = p
        self.t = 0
        self.s1 = 0.0  # sum of lr
        self.s2 = 0.0  # sum of lr^2

    def step(self):
        lr = self.base_lr / ((self.t + 1) ** self.p)
        self.t += 1
        self.s1 += lr
        self.s2 += lr * lr
        return lr, self.s1, self.s2

class PolyakScheduler:
    def __init__(self, f_star: float):

        self.f_star = f_star
        self.t = 0
        self.s1 = 0.0  # sum of lr
        self.s2 = 0.0  # sum of lr^2

    def step(self, f, grad):
        lr = abs(f - self.f_star) / torch.norm(grad) ** 2
        self.t += 1
        self.s1 += lr
        self.s2 += lr * lr
        return lr, self.s1, self.s2