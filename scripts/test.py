"""Roll out a trained policy with deterministic actions and report episode stats.

python xRL/scripts/test.py artifact=<run dir | model.pth> num_envs=64

"""
from __future__ import annotations

import isaacgym  # before torch

import _legged_gym_path  # noqa: F401

# motion files are globbed relative to the cwd, import before hydra changes it
import legged_gym
import legged_gym.envs

import json
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

import xRL
from xRL.alg import alg_name_to_path
from xRL.utils.common import load_class_from_path
from xRL.utils.model_util import load_model
from xRL.wrappers.create_environment import create_task_env

MAX_IMITATION_SAMPLES = 20000


def _mean_nn_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> float:
    """Mean distance from each row of a to its nearest row of b."""
    return torch.cat([torch.cdist(a[i:i + chunk], b).min(dim=1).values
                      for i in range(0, a.shape[0], chunk)]).mean().item()


def imitation_distance(policy: torch.Tensor, expert: torch.Tensor) -> dict[str, float]:
    mean, std = expert.mean(dim=0), expert.std(dim=0).clamp(min=1e-3)
    policy, expert = (policy - mean) / std, (expert - mean) / std
    # policy->expert: is every motion natural, expert->policy: is the reference covered
    to_expert, to_policy = _mean_nn_dist(policy, expert), _mean_nn_dist(expert, policy)
    return {"imitation_dist": 0.5 * (to_expert + to_policy),
            "imitation_policy_to_expert": to_expert,
            "imitation_expert_to_policy": to_policy}


@hydra.main(config_path=xRL.LIB_PATH.joinpath("cfg").as_posix(), config_name="default")
def main(cfg: DictConfig):
    if cfg.artifact is None:
        raise ValueError("Set artifact=<run dir or .pth> to test a policy.")
    logger.info(f"legged_gym: {legged_gym.LEGGED_GYM_ROOT_DIR}")

    env = create_task_env(cfg)
    alg_name = cfg.alg.name if cfg.alg.name.endswith("Agent") else cfg.alg.name + "Agent"
    agent = load_class_from_path(alg_name, alg_name_to_path[alg_name])(env=env, cfg=cfg)
    load_model(agent.actor, "actor", cfg)
    if cfg.alg.obs_norm:
        load_model(agent.obs_rms, "obs_rms", cfg)
    agent.actor.eval()

    num_steps = int(cfg.get("test_steps") or env.max_episode_steps)
    ep_return = torch.zeros(env.num_envs, device=env.device)
    ep_length = torch.zeros(env.num_envs, device=env.device)
    returns, lengths, falls = [], [], []
    vel_err = []
    amp_transitions = []
    keep_per_step = max(1, MAX_IMITATION_SAMPLES // num_steps)

    obs, _ = env.reset()
    amp_obs = env.get_amp_observations()
    with torch.no_grad():
        for _ in range(num_steps):
            obs, reward, done, info = env.step(agent.get_actions(obs, sample=False))
            ep_return += reward
            ep_length += 1
            vel_err.append(env.get_velocity_tracking()["velocity/track_err_lin_vel_x"])

            next_amp_obs = env.get_amp_observations()
            amp_next_term = next_amp_obs.clone()
            reset_ids = info["reset_env_ids"]
            if reset_ids.numel() > 0:
                amp_next_term[reset_ids] = info["terminal_amp_obs"]
            keep = torch.randperm(env.num_envs, device=env.device)[:keep_per_step]
            amp_transitions.append(torch.cat([amp_obs, amp_next_term], dim=-1)[keep])
            amp_obs = next_amp_obs

            ended = done.bool()
            if ended.any():
                returns += ep_return[ended].tolist()
                lengths += ep_length[ended].tolist()
                falls += (~info["time_outs"].bool()[ended]).tolist()
                ep_return[ended] = 0.0
                ep_length[ended] = 0.0

        policy_amp = torch.cat(amp_transitions).to(agent.amp_manager.device)
        expert_s, expert_sn = agent.amp_manager.amp_data.feed_forward_generator(1, policy_amp.shape[0])
        metrics = imitation_distance(policy_amp, torch.cat([expert_s, expert_sn], dim=-1))

    # episodes still running count with their partial length, a policy that never falls has no ended episodes
    lengths_all = lengths + ep_length[ep_length > 0].tolist()
    metrics.update({
        "num_steps": num_steps,
        "num_envs": env.num_envs,
        "finished_episodes": len(returns),
        "episode_return": float(np.mean(returns)) if returns else float(ep_return.mean().item()),
        "episode_length": float(np.mean(lengths_all)),
        "fall_rate": float(np.mean(falls)) if falls else 0.0,
        "vel_x_track_err": float(np.mean(vel_err)),
    })

    logger.info(f"Tested {num_steps} steps x {env.num_envs} envs, {len(returns)} finished episodes")
    for key, value in metrics.items():
        logger.info(f"{key:<28} {value:.4f}" if isinstance(value, float) else f"{key:<28} {value}")

    out_path = os.path.join(HydraConfig.get().runtime.output_dir, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"metrics -> {out_path}")


if __name__ == "__main__":
    main()
