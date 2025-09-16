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
    def __init__(self, f_star: float, hessian_factor: float = 1.0):

        self.f_star = f_star
        self.hessian_factor = hessian_factor
        self.t = 0
        self.s1 = 0.0  # sum of lr
        self.s2 = 0.0  # sum of lr^2

    def step(self, f, grad):
        lr = self.hessian_factor * (abs(f - self.f_star) / torch.norm(grad) ** 2)
        self.t += 1
        self.s1 += lr
        self.s2 += lr * lr
        return lr, self.s1, self.s2


class HybridScheduler:
    def __init__(self, f_star, hessian_factor=1.0, poly_base_lr=1.0, poly_p=0.75,
                 stagnation_window=5, stagnation_tol=1e-6):
        self.polyak = PolyakScheduler(f_star=f_star, hessian_factor=hessian_factor)
        self.poly_decay = None
        self.poly_p = poly_p

        self.stagnation_window = stagnation_window
        self.stagnation_tol = stagnation_tol
        self.lr_history = []
        self.use_poly_decay = False
        self.t = 0

    def step(self, f=None, grad=None):
        if self.use_poly_decay:
            lr, s1, s2 = self.poly_decay.step()
        else:
            lr, s1, s2 = self.polyak.step(f, grad)
            self.lr_history.append(lr)

            # Check for stagnation
            if len(self.lr_history) > self.stagnation_window:
                recent = self.lr_history[-self.stagnation_window:]
                if max(recent) - min(recent) < self.stagnation_tol:
                    # Initialize PolynomialDecay starting from last Polyak lr
                    self.poly_decay = PolynomialDecayScheduler(base_lr=lr, p=self.poly_p)
                    self.use_poly_decay = True
                    print(f"Switching to PolynomialDecay at step {self.t}, lr={lr:.6f}")

        self.t += 1
        return lr, s1, s2