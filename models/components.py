import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))


class TopKRouter(nn.Module):
    """Router that selects top-k experts for each token"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1  # Standard deviation for noise during training

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss
        aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight


class ZeroComputeMoe(nn.Module):
    """Mixture of Experts layer from https://arxiv.org/pdf/2509.01322v1"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_zero_experts: int | None = None,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_zero_experts = num_zero_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

        # Create router
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape
        total_num_of_experts = (
            self.num_experts
            if self.num_zero_experts is None
            else self.num_experts + self.num_zero_experts
        )

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(total_num_of_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(
                dim=-1
            )  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                # This code might not work **please test it greatly**
                # Tell me if it doesn't work
                # RohanKhanBD
                if self.num_zero_experts is None or expert_idx < self.num_experts:
                    expert_output = self.experts[expert_idx](expert_input)
                else:
                    expert_output = expert_input

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = expert_indices == expert_idx  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = (
                    router_weights[expert_mask]
                    .gather(-1, positions.unsqueeze(-1))
                    .squeeze(-1)
                )

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss
        aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self, router_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight
