from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from legged_gym.utils.amp.motion_loader import AMPLoader
from omegaconf import DictConfig

from .discriminator import AMPDiscriminator
from xRL.utils.torch_util import Normalizer, TaskRewardNormalizer


@dataclass
class AMPManager:
    env: Any
    cfg: DictConfig

    device:           torch.device = field(init=False)
    discriminator:    AMPDiscriminator = field(init=False)
    amp_data:         AMPLoader = field(init=False)
    amp_normalizer:   Normalizer = field(init=False)
    task_reward_normalizer: TaskRewardNormalizer = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.cfg.device)
        self.disc_reward_weight  = self.cfg.alg.disc_reward_weight
        self.disc_reward_scale   = self.cfg.alg.disc_reward_scale
        self.disc_grad_penalty   = self.cfg.alg.disc_grad_penalty
        self.disc_logit_reg_coef = self.cfg.alg.disc_logit_reg_coef

        self.amp_data = AMPLoader(
            device                 = str(self.device),
            time_between_frames    = self.env.dt,
            preload_transitions    = True,
            num_preload_transitions= self.cfg.alg.num_preload_transitions,
            motion_files           = list(self.env.amp_files_path),
        )

        obs_dim = self.amp_data.observation_dim

        self.discriminator = AMPDiscriminator(
            input_dim         = obs_dim * 2,
            hidden_layer_sizes= list(self.cfg.alg.disc_hidden_dims),
            device            = str(self.device),
        ).to(self.device)

        self.amp_normalizer = Normalizer(obs_dim)
        self.task_reward_normalizer = TaskRewardNormalizer()

    def blend_rewards(self, disc_r: torch.Tensor, task_r: torch.Tensor) -> torch.Tensor:
        self.task_reward_normalizer.update(task_r)
        task_r_norm = self.task_reward_normalizer.normalize(task_r)
        return self.disc_reward_weight * disc_r + (1.0 - self.disc_reward_weight) * task_r_norm

    def compute_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.discriminator.compute_discriminator_scores(state, next_state, normalizer)

        if self.cfg.alg.disc_loss_type == "lsgan":
            #target +-1
            disc_reward = self.disc_reward_scale * torch.clamp(1 - 0.25 * (d - 1).pow(2), min=0)
        else:
            # GAN-theoretic reward dual to BCE loss: -log(1 - σ(d))
            # Clamped to avoid log(0) when σ(d) → 1
            p = torch.sigmoid(d)
            disc_reward = self.disc_reward_scale * -torch.log(torch.clamp(1 - p, min=1e-4))

        reward_b = self.blend_rewards(disc_reward, task_reward.unsqueeze(-1))
        return reward_b.squeeze(-1), disc_reward.squeeze(-1)

    def compute_lsgan_loss(self, disc_score: torch.Tensor, target: float) -> torch.Tensor:
        #  Least-Squares GAN LSGAN (MSE)
        # AMP - Nikita Rudin, 2021
        target = torch.full_like(disc_score, fill_value=target)
        return torch.nn.MSELoss()(disc_score, target)

    def compute_bce_loss(self, disc_score: torch.Tensor, target: float) -> torch.Tensor:
        # Binary Cross-Entropy loss with logits (numerically stable sigmoid + BCE).
        # AMP - Peng, 2021
        target = torch.full_like(disc_score, fill_value=target)
        return torch.nn.BCEWithLogitsLoss()(disc_score, target)

    def compute_discr_loss(
        self,
        policy_state:      torch.Tensor,
        policy_next_state: torch.Tensor,
        expert_state:      torch.Tensor,
        expert_next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_type = self.cfg.alg.disc_loss_type

        policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

        if loss_type == "lsgan":
            expert_loss   = self.compute_lsgan_loss(expert_d, target=1.0)
            policy_loss   = self.compute_lsgan_loss(policy_d, target=-1.0)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, disc_grad_penalty=self.disc_grad_penalty)
        elif loss_type == "bce":
            expert_loss   = self.compute_bce_loss(expert_d, target=1.0)
            policy_loss   = self.compute_bce_loss(policy_d, target=0.0)
            grad_pen_loss = self.discriminator.compute_grad_pen_averaged(
                expert_state, expert_next_state,
                policy_state, policy_next_state,
                disc_grad_penalty=self.disc_grad_penalty)
        else:
            raise ValueError(
                f"Unknown disc_loss_type: {loss_type!r}. "
                "Expected 'lsgan' or 'bce'."
            )

        discr_loss = 0.5 * (expert_loss + policy_loss)
        return discr_loss, grad_pen_loss, policy_d, expert_d

    def compute_discriminator_accuracy(
        self,
        policy_scores: torch.Tensor,
        expert_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        policy_acc = (policy_scores < 0).float().mean()
        expert_acc = (expert_scores > 0).float().mean()
        return policy_acc, expert_acc

    def compute_regularization(
        self,
        policy_state:      torch.Tensor,
        policy_next_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        discr_loss_coef = self.cfg.alg.discr_loss_coef
        grad_pen_loss_coef = self.cfg.alg.grad_pen_loss_coef

        expert_state, expert_next_state = self.amp_data.feed_forward_generator(
            1,
            policy_state.size(0),
        )

        expert_state = expert_state.to(self.device)
        expert_next_state = expert_next_state.to(self.device)

        if self.amp_normalizer is not None:
            self.amp_normalizer.update(policy_state)
            self.amp_normalizer.update(expert_state)
            with torch.no_grad():
                policy_state      = self.amp_normalizer.normalize_torch(policy_state,      self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state      = self.amp_normalizer.normalize_torch(expert_state,      self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

        discr_loss, grad_pen_loss, policy_d, expert_d = self.compute_discr_loss(
            policy_state,
            policy_next_state,
            expert_state,
            expert_next_state,
        )

        total_reg = (
            discr_loss_coef * discr_loss
            + grad_pen_loss_coef * grad_pen_loss
        )

        if self.disc_logit_reg_coef > 0 and self.cfg.alg.disc_loss_type != "lsgan":
            logit_reg_loss = self.discriminator.compute_logit_regularization()
            total_reg += self.disc_logit_reg_coef * logit_reg_loss

        policy_acc, expert_acc = self.compute_discriminator_accuracy(policy_d, expert_d)

        return {
            "total_reg": total_reg,
            "discr_loss": discr_loss,
            "grad_pen_loss": grad_pen_loss,
            "policy_acc": policy_acc,
            "expert_acc": expert_acc,
            "policy_d": policy_d.detach(),
            "expert_d": expert_d.detach(),
        }


