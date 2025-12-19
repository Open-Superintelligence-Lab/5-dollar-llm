import torch
import torch.nn.functional as F
import sys
from itertools import repeat


def _maybe_compile(fn):
    """Apply torch.compile only on supported Python versions."""
    if sys.version_info >= (3, 14):
        # torch.compile not supported on Python 3.14+
        return fn
    try:
        return torch.compile(fn)
    except Exception:
        return fn


# =============================================================================
# Newton-Schulz Iteration (Original)
# =============================================================================

@_maybe_compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.half()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


# =============================================================================
# Polar Express (2.5x faster convergence)
# Paper: arXiv:2505.16932 - "The Polar Express"
# =============================================================================

# Optimal quintic coefficients with safety factor for bfloat16 stability
POLAR_EXPRESS_COEFFS = [
    (8.205456755589732, -22.842964278313602, 16.53016098365233),
    (4.067029218854677, -2.8586875888494163, 0.5285577863545636),
    (3.9091927550319346, -2.8235663524375814, 0.5354241497412851),
    (3.2854650071978226, -2.416325766152358, 0.4950484358563564),
    (2.277922792035462, -1.6202010739353955, 0.4063241899628388),
    (1.873565739394452, -1.2309561656876105, 0.36551797420143887),
    (1.8564371095578693, -1.2138138175914251, 0.36396906439285626),
    (1.875, -1.25, 0.375),  # Final coefficient (no safety factor)
]


@_maybe_compile
def zeropower_via_polar_express(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Polar Express iteration for matrix orthogonalization.

    Uses adaptive polynomial sequences optimized via the Remez algorithm.
    Converges faster than Newton-Schulz (8 vs 20 iterations to machine precision).

    Args:
        G: Input tensor (ndim >= 2)
        steps: Number of iterations (default 5, recommended 5-6)

    Returns:
        Orthogonalized matrix

    Reference: arXiv:2505.16932
    """
    assert G.ndim >= 2
    X = G.bfloat16()  # Use bfloat16 for speed and stability

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize with safety factor
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    # Get coefficient sequence (reuse last if steps > len)
    hs = POLAR_EXPRESS_COEFFS[:steps] + list(
        repeat(POLAR_EXPRESS_COEFFS[-1], max(0, steps - len(POLAR_EXPRESS_COEFFS)))
    )

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


# =============================================================================
# Muon Optimizer (Original)
# =============================================================================

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 use_polar_express=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                       use_polar_express=use_polar_express)
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

                # Choose orthogonalization method
                if group["use_polar_express"]:
                    g = zeropower_via_polar_express(g, steps=group["ns_steps"])
                else:
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


# =============================================================================
# NorMuon Optimizer (11-21% efficiency gain)
# Paper: arXiv:2510.05491 - "NorMuon: Making Muon more efficient and scalable"
# =============================================================================

class NorMuon(torch.optim.Optimizer):
    """
    NorMuon - Muon with Neuron-wise Normalization

    Adds per-neuron adaptive learning rates via row-wise second-order statistics.
    This balances the spectral properties after orthogonalization.

    Performance: 11-21% efficiency gain over standard Muon with 2.9% overhead.

    Reference: arXiv:2510.05491
    """
    def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95,
                 nesterov=True, ns_steps=5, use_polar_express=True):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2,
                       nesterov=nesterov, ns_steps=ns_steps,
                       use_polar_express=use_polar_express)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize state buffers
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    # Per-neuron second-order statistics (shape: [m, 1] for [m, n] weight)
                    state["second_momentum_buffer"] = torch.zeros(
                        g.size(0), 1, device=g.device, dtype=g.dtype
                    )

                buf = state["momentum_buffer"]
                second_buf = state["second_momentum_buffer"]

                # First-order momentum
                buf.lerp_(g, 1 - group["momentum"])
                update = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf.clone()

                # Orthogonalization
                if group["use_polar_express"]:
                    update = zeropower_via_polar_express(update, steps=group["ns_steps"])
                else:
                    update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])

                # ============ NorMuon Addition ============
                # Save original norm
                vnorm = update.norm(dim=(-2, -1), keepdim=True)

                # Row-wise second-order statistics (per-neuron variance)
                # Cast to float32 for stable accumulation
                v_mean = torch.mean(update.float() * update.float(), dim=-1, keepdim=True)

                # Update second momentum (EMA of per-neuron stats)
                second_buf.lerp_(v_mean, 1 - group["beta2"])

                # Per-neuron adaptive step size
                step_size = 1.0 / (second_buf.sqrt() + 1e-10)

                # Apply neuron-wise normalization
                update = update * step_size

                # Restore original Frobenius norm (preserve update magnitude)
                vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
                update = update * (vnorm / (vnorm_new + 1e-10))
                # ==========================================

                p.add_(update.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


# =============================================================================
# MuonPolarExpress (Convenience alias)
# =============================================================================

class MuonPolarExpress(Muon):
    """Muon with Polar Express enabled by default."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, use_polar_express=True)
