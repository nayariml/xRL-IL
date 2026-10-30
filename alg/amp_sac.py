from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from torch import nn

from xRL.alg.sac import SACAgent
from xRL.amp_scripts.manager import AMPManager
from xRL.memory.nstep_replay import TransitionBuffer
from xRL.utils.torch_util import autocast_if_cuda, soft_update


@dataclass
class AMPSACLearner:
    actor: nn.Module
    critic: nn.Module
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    critic_target: nn.Module
    entropy_module: object
    optimizer_update_fn: callable
    amp_manager: AMPManager
    cfg: DictConfig
    device: torch.device

    def compute_actor_loss(self, obs):
        with autocast_if_cuda(self.device):
            actions, _, log_prob = self.actor.sample_actions_logprob(obs)
            q_min = self.critic.min_q(obs, actions)
            actor_loss = (self.entropy_module.alpha() * log_prob - q_min).mean()
        return actor_loss, log_prob

    def compute_critic_loss(self, obs, action, reward, next_obs, discount):
        with torch.no_grad():
            with autocast_if_cuda(self.device):
                next_actions, _, log_prob = self.actor.sample_actions_logprob(next_obs)
                target_q = self.critic_target.min_q(next_obs, next_actions) - self.entropy_module.alpha() * log_prob
                target_q = reward + discount * target_q

        with autocast_if_cuda(self.device):
            current_q1, current_q2 = self.critic.forward_q1_q2(obs, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        return critic_loss

    def step_update(self, obs, action, reward, next_obs, discount, amp_obs, amp_next_obs):
        amp_stats = self.amp_manager.compute_regularization(amp_obs, amp_next_obs)

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, discount)
        self.optimizer_update_fn(self.critic_optimizer, critic_loss)

        self.critic.requires_grad_(False)
        base_actor_loss, log_prob = self.compute_actor_loss(obs)
        with autocast_if_cuda(self.device):
            total_reg = torch.as_tensor(amp_stats["total_reg"], dtype=base_actor_loss.dtype, device=self.device)
            actor_loss = base_actor_loss + total_reg
        self.optimizer_update_fn(self.actor_optimizer, actor_loss)
        self.critic.requires_grad_(True)

        alpha_loss_value = self.entropy_module.update_alpha(log_prob, self.optimizer_update_fn)

        soft_update(self.critic_target, self.critic, self.cfg.alg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.entropy_module.alpha(scalar=True),
            "alpha_loss": alpha_loss_value,
            "discr_loss": amp_stats["discr_loss"].item(),
            "grad_pen_loss": amp_stats["grad_pen_loss"].item(),
            "policy_acc": amp_stats["policy_acc"].item(),
            "expert_acc": amp_stats["expert_acc"].item(),
            "policy_d": amp_stats["policy_d"],
            "expert_d": amp_stats["expert_d"],
        }


@dataclass
class AMPSACAgent(SACAgent):
    """SAC agent with AMP regularization."""

    def __post_init__(self):
        super().__post_init__()

        self.amp_obs_dim = self.env.amp_observation_dim
        self.amp_manager = AMPManager(env=self.env, cfg=self.cfg)

        self.replay = TransitionBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            add_obs_dim=self.amp_obs_dim,
            num_envs=self.cfg.num_envs,
            nstep=self.cfg.alg.nstep,
            device=self.device,
            gamma=self.cfg.alg.gamma,
        )

        params = [
            {"params": self.actor.parameters(), "name": "actor"},
            {"params": self.amp_manager.discriminator.feature_extractor.parameters(),
             "weight_decay": 1e-3, "name": "feature_extractor"},
            {"params": self.amp_manager.discriminator.head.parameters(),
             "weight_decay": 1e-2, "name": "head"},
        ]
        self.actor_optimizer = torch.optim.AdamW(params, lr=self.cfg.alg.actor_lr)

        self.learner = AMPSACLearner(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            critic_target=self.critic_target,
            entropy_module=self.entropy,
            optimizer_update_fn=self.optimizer_update_mix_precision,
            amp_manager=self.amp_manager,
            cfg=self.cfg,
            device=self.device,
        )

    @torch.no_grad()
    def collect_transitions(self, env, timesteps: int, random: bool):
        n = self.cfg.num_envs
        batch_obs = torch.empty((n, timesteps, self.obs_dim), device=self.device)
        batch_actions = torch.empty((n, timesteps, self.action_dim), device=self.device)
        batch_rewards = torch.empty((n, timesteps, 1), device=self.device)
        batch_next_obs = torch.empty((n, timesteps, self.obs_dim), device=self.device)
        batch_terminated = torch.empty((n, timesteps, 1), dtype=torch.bool, device=self.device)
        batch_episode_end = torch.empty((n, timesteps, 1), dtype=torch.bool, device=self.device)
        amp_batch_obs = torch.empty((n, timesteps, self.amp_obs_dim), device=self.device)
        amp_batch_next_obs = torch.empty((n, timesteps, self.amp_obs_dim), device=self.device)

        ep_infos = []
        obs = self.obs
        amp_obs = env.get_amp_observations().to(self.device)

        for t in range(timesteps):
            if self.cfg.alg.obs_norm:
                self.obs_rms.update(obs)

            if random:
                action = torch.rand((n, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            reset_env_ids = info["reset_env_ids"]

            # Terminal-state correction
            next_amp_obs = env.get_amp_observations().to(self.device)
            next_amp_obs_term = next_amp_obs.clone()
            next_obs_term = next_obs.clone()
            if reset_env_ids.numel() > 0:
                next_amp_obs_term[reset_env_ids] = info["terminal_amp_obs"].to(self.device)
                next_obs_term[reset_env_ids] = info["terminal_next_obs"]
                ep_infos.append(info["episode"])

            episode_end = done.bool()
            terminated = episode_end & ~info["time_outs"].bool()

            reward_b, disc_reward = self.amp_manager.compute_reward(
                amp_obs, next_amp_obs_term, reward, normalizer=self.amp_manager.amp_normalizer,
            )
            self.update_tracker(reward_b, episode_end)
            self.update_additional_tracker(disc_reward, episode_end)

            batch_obs[:, t] = obs
            batch_actions[:, t] = action
            batch_rewards[:, t, 0] = reward_b.to(self.device)
            batch_next_obs[:, t] = next_obs_term
            batch_terminated[:, t, 0] = terminated
            batch_episode_end[:, t, 0] = episode_end
            amp_batch_obs[:, t] = amp_obs
            amp_batch_next_obs[:, t] = next_amp_obs_term

            obs = next_obs
            amp_obs = next_amp_obs

        self.obs = obs

        data = self.replay.push_sequence(
            batch_obs,
            batch_actions,
            self.cfg.alg.reward_scale * batch_rewards,
            batch_next_obs,
            batch_terminated,
            batch_episode_end,
            amp_batch_obs,
            amp_batch_next_obs,
        )

        # release memory -good practice
        del batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminated, batch_episode_end
        del amp_batch_obs, amp_batch_next_obs
        torch.cuda.empty_cache()

        return data, timesteps * n, ep_infos

    def update(self, memory):
        stats = []
        for _ in range(self.cfg.alg.update_times):
            obs, action, reward, next_obs, discount, amp_obs, amp_next_obs = memory.sample(self.cfg.alg.batch_size)

            obs, action, reward, next_obs, discount = (
                x.to(self.device) for x in (obs, action, reward, next_obs, discount))
            amp_obs = amp_obs.to(self.device)
            amp_next_obs = amp_next_obs.to(self.device)

            if self.cfg.alg.obs_norm:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)

            stats.append(self.learner.step_update(obs, action, reward, next_obs, discount, amp_obs, amp_next_obs))

        def mean(key):
            return float(np.mean([s[key] for s in stats]))

        policy_d = torch.cat([s["policy_d"] for s in stats]).squeeze(-1).float().cpu().numpy()
        expert_d = torch.cat([s["expert_d"] for s in stats]).squeeze(-1).float().cpu().numpy()

        torch.cuda.empty_cache()

        # numpy arrays go to histograms, everything else is a scalar
        return {
            "discriminator/policy_logits": policy_d,
            "discriminator/expert_logits": expert_d,
            "discriminator/policy_score_mean": float(policy_d.mean()),
            "discriminator/expert_score_mean": float(expert_d.mean()),
            "train/critic_loss": mean("critic_loss"),
            "train/actor_loss": mean("actor_loss"),
            "train/return": self.return_tracker.mean(),
            "train/return_disc": self.disc_return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/entropy": mean("alpha"),
            "train/alpha_loss": mean("alpha_loss"),
            "train/discr_loss": mean("discr_loss"),
            "train/grad_pen_loss": mean("grad_pen_loss"),
            "train/policy_acc": mean("policy_acc"),
            "train/expert_acc": mean("expert_acc"),
        }
