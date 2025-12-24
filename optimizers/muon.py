import torch
import torch.nn.functional as F

# coeffs for polar express 
# not pre_computed, same as modded-nanoGPT 
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile()
def zeropower_polar_express(G:torch.Tensor, steps: int = 5,):
    """Polar express as replacement for Newton-Schulz iteration"""
    assert G.ndim >= 2
    assert steps <= len(coeffs_list)

    X = G.bfloat16()
    # X = G.half()

    transpose_needed = G.size(-2) > G.size(-1) # transposing if tall matrix
    if transpose_needed: 
        X = X.mT 
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7) # safety factor
    
    coeffs = coeffs_list[:steps]
    for a , b, c in coeffs:
        A = X @ X.mT 
        A2 = A @ A 
        B = b * A + c * A2
        X = a * X + B @ X  # Right-multiplication for left polar factor
    
    if transpose_needed: 
        X = X.mT 
    
    return X # orthogonalized 




class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_polar_express(g, steps=group["ns_steps"]) # steps are 5 for both ns and pe
                g = g.to(p.dtype)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
