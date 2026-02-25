import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepDeltaRes(nn.Module):
    def __init__(
        self,
        d_model: int,
        beta_dim: int,
        k_eps: float = 1e-6,
        sigmoid_scale: float = 4.0,
    ):
        super().__init__()
        self.k_eps = k_eps
        self.sigmoid_scale = sigmoid_scale

        self.beta_in = nn.Linear(d_model, beta_dim, bias=False)
        self.beta_out = nn.Linear(beta_dim, 1, bias=True)
        self.v_proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor, k_in: torch.Tensor, contxt: torch.Tensor):
        k = F.normalize(k_in, p=1, dim=-1, eps=self.k_eps)
        beta_logits = self.beta_out(F.tanh(self.beta_in(contxt)))
        beta = 2.0 * F.sigmoid(beta_logits)
        proj = torch.sum(k * x, dim=-1, keepdim=True)
        v = F.sigmoid(self.v_proj(x)) * self.sigmoid_scale
        delta = beta * (v - proj)
        update = delta * k
        return x * update
