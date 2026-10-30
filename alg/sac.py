from copy import deepcopy
from dataclasses import dataclass

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from xRL.alg.ac_base import ActorCriticBase
from xRL.utils.torch_util import autocast_if_cuda


@dataclass
class SACEntropy:
    cfg: DictConfig
    action_dim: int
    device: torch.device

    def __post_init__(self):
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optim = None
        if self.cfg.alg.alpha is None:
            self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.cfg.alg.alpha_lr)
        self.target_entropy = -self.action_dim

    def alpha(self, detach=True, scalar=False):
        alpha = self.log_alpha.exp() if self.cfg.alg.alpha is None else self.cfg.alg.alpha
        if detach and isinstance(alpha, torch.Tensor):
            alpha = alpha.detach()
        if scalar and isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        return alpha

    def update_alpha(self, log_prob, optimizer_update_fn):
        if self.alpha_optim is None:
            return 0.0
        with autocast_if_cuda(self.device):
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        optimizer_update_fn(self.alpha_optim, alpha_loss)
        return alpha_loss.item()


@dataclass
class SACAgent(ActorCriticBase):

    def __post_init__(self):
        super().__post_init__()
        self.critic_target = deepcopy(self.critic)
        self.entropy = SACEntropy(cfg=self.cfg, action_dim=self.action_dim, device=self.device)

    def get_actions(self, obs, sample=True):
        if self.cfg.alg.obs_norm:
            obs = self.obs_rms.normalize(obs)
        return self.actor(obs, sample=sample)
