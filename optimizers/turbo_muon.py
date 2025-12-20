import torch
import torch.distributed as dist


@torch.compile
def newton_schulz(G, iter=4, epsilon: float = 1e-7, dtype=torch.bfloat16):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ][-iter:]
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # First NS iteration
    a = ns_consts[0][0]
    b = ns_consts[0][1]
    c = ns_consts[0][2]
    A = X @ X.mT
    s = torch.rsqrt(torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=epsilon))
    X = X * s.unsqueeze(-1)
    A = A * s.unsqueeze(-1) * s.unsqueeze(-2)

    B = b * A + c * A @ A
    X = a * X + B @ X

    # Perform the last NS iterations
    for a, b, c in ns_consts[1:]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class TurboMuon(torch.optim.Optimizer):
    """
    TurboMuon variant for usage in non-distributed settings.
    """

    def __init__(
        self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=4
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                update = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                if update.ndim == 4:
                    update = update.view(len(update), -1)
                update = newton_schulz(update, iter=group["ns_steps"])
                update *= max(1, g.size(-2) / g.size(-1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
