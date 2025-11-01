# ====== Optimizer: Lion ======
import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                m = state.setdefault("exp_avg", torch.zeros_like(p))
                if wd:
                    p.add_(p, alpha=-lr * wd)
                u = m.mul(beta1).add(p.grad, alpha=1 - beta1)
                p.add_(torch.sign(u), alpha=-lr)
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
