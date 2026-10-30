from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


TARGET_GRAD_NORM = 0.0


class AMPDiscriminator(nn.Module):

    def __init__(self, input_dim: int, hidden_layer_sizes: list[int], device: str):
        super().__init__()
        self.device = device
        self.input_dim = input_dim

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h in hidden_layer_sizes:
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU()]
            in_dim = h

        self.feature_extractor = nn.Sequential(*layers).to(device)
        self.head = nn.Linear(in_dim, 1).to(device)

        self.feature_extractor.train()
        self.head.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_extractor(x))

    def compute_grad_pen(
        self, expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        disc_grad_penalty: float = 10.0) -> torch.Tensor:
        """
        Only penalizes gradients on expert data
        Adapted from:
        Copyright (c) 2021, ETH Zurich, Nikita Rudin
        Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES
        https://github.com/escontra/AMP_for_hardware/tree/main"""
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)
        d = self.forward(expert_data)
        ones = torch.ones_like(d)
        grad = autograd.grad(
            outputs=d, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True,
        )[0]
        return disc_grad_penalty * (grad.norm(2, dim=1) - TARGET_GRAD_NORM).pow(2).mean()

    def compute_logit_regularization(self) -> torch.Tensor:
        """L2 penalty on style_head weights to prevent discriminator score explosion."""
        return torch.sum(torch.square(self.head.weight))

    def compute_grad_pen_averaged(
        self,
        expert_state:      torch.Tensor,
        expert_next_state: torch.Tensor,
        policy_state:      torch.Tensor,
        policy_next_state: torch.Tensor,
        disc_grad_penalty: float = 10.0,
    ) -> torch.Tensor:
        """Gradient penalty on expert and policy data separately, averaged.
        Inspired from MimicKit - Peng,2025
        https://github.com/xbpeng/MimicKit
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        policy_data = torch.cat([policy_state, policy_next_state], dim=-1)

        expert_data.requires_grad_(True)
        policy_data.requires_grad_(True)

        expert_d = self.forward(expert_data)
        policy_d = self.forward(policy_data)

        expert_grad = autograd.grad(
            outputs=expert_d,
            inputs=expert_data,
            grad_outputs=torch.ones_like(expert_d),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        policy_grad = autograd.grad(
            outputs=policy_d,
            inputs=policy_data,
            grad_outputs=torch.ones_like(policy_d),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        expert_grad_penalty = torch.sum(torch.square(expert_grad), dim=-1).mean()
        policy_grad_penalty = torch.sum(torch.square(policy_grad), dim=-1).mean()

        return disc_grad_penalty * 0.5 * (expert_grad_penalty + policy_grad_penalty)

    @torch.no_grad()
    def compute_discriminator_scores(self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        self.eval()
        if normalizer is not None:
            state = normalizer.normalize_torch(state, self.device)
            next_state = normalizer.normalize_torch(next_state, self.device)
        d = self.forward(torch.cat([state, next_state], dim=-1))
        self.train()
        return d

