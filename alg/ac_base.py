from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.utils import clip_grad_norm_
try:
    from torch.amp import GradScaler
except ImportError:  # torch < 2.3
    from torch.cuda.amp import GradScaler

from xRL.models import model_name_to_path
from xRL.utils.common import Tracker, load_class_from_path
from xRL.utils.torch_util import RunningMeanStd


@dataclass
class ActorCriticBase:
    env: Any
    cfg: DictConfig
    obs_dim: int = None
    action_dim: int = None

    def __post_init__(self):
        self.device = torch.device(self.cfg.device)

        self.obs = None
        if self.obs_dim is None:
            self.obs_dim = int(self.env.observation_space.shape[-1])
        if self.action_dim is None:
            self.action_dim = self.env.action_dim

        alg = self.cfg.alg
        actor_class = load_class_from_path(alg.actor_class, model_name_to_path[alg.actor_class])
        critic_class = load_class_from_path(alg.critic_class, model_name_to_path[alg.critic_class])
        self.actor = actor_class(self.obs_dim, self.action_dim, alg.act_hidden_layers, alg.act_activation).to(self.device)
        self.critic = critic_class(self.obs_dim, self.action_dim, alg.cri_hidden_layers, alg.cri_activation).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), alg.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), alg.critic_lr)

        self.return_tracker = Tracker(alg.tracker_len)
        self.disc_return_tracker = Tracker(alg.tracker_len)
        self.step_tracker = Tracker(alg.tracker_len)
        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.device)
        self.current_disc_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.device)

        self.obs_rms = RunningMeanStd(shape=self.obs_dim, device=self.device) if alg.obs_norm else None

        # Initialize GradScaler
        self.scaler = GradScaler()

    def reset_agent(self):
        self.obs, _ = self.env.reset()

    def update_additional_tracker(self, disc_reward, done):
        self.current_disc_returns += disc_reward
        env_done_indices = torch.where(done)[0]
        self.disc_return_tracker.update(self.current_disc_returns[env_done_indices])
        self.current_disc_returns[env_done_indices] = 0

    def update_tracker(self, reward, done):
        self.current_returns += reward
        self.current_lengths += 1
        env_done_indices = torch.where(done)[0]
        self.return_tracker.update(self.current_returns[env_done_indices])
        self.step_tracker.update(self.current_lengths[env_done_indices])
        self.current_returns[env_done_indices] = 0
        self.current_lengths[env_done_indices] = 0

    def optimizer_update_mix_precision(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(objective).backward()
        self.scaler.unscale_(optimizer)
        grad_norm = None
        if self.cfg.alg.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.cfg.alg.max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        return grad_norm
