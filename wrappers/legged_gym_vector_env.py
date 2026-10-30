from __future__ import annotations

import contextlib
import io
import sys
from typing import Any

import numpy as np
import torch
import gymnasium as gym


def _parse_legged_gym_args(task: str):
    """AMP_Zoo's parser takes the task positionally, upstream legged_gym's as --task."""
    from legged_gym.utils import get_args

    # hide hydra args from legged gym parser
    argv = sys.argv
    try:
        for task_args in (["xRL", task], ["--task", task]):
            sys.argv = argv[:1] + task_args
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    return get_args()
            except SystemExit:
                continue
    finally:
        sys.argv = argv
    raise RuntimeError(f"legged_gym get_args() rejected task {task!r}")


class LeggedGymWrapper:
    def __init__(self, task: str, num_envs: int, seed: int, headless: bool = True,
                 sim_device: str = "cuda:0"):
        from legged_gym.envs import task_registry

        args = _parse_legged_gym_args(task)

        device_id = int(sim_device.split(":")[1]) if ":" in sim_device else 0
        args.num_envs = num_envs
        args.seed = seed
        args.headless = headless
        args.sim_device_type = "cuda"
        args.compute_device_id = device_id
        args.sim_device_id = device_id
        args.sim_device = f"cuda:{device_id}"
        args.rl_device = args.sim_device
        args.graphics_device_id = -1 if headless else device_id

        self.env, self.env_cfg = task_registry.make_env(name=task, args=args)

        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.dt = self.env.dt
        self.max_episode_steps = int(self.env.max_episode_length)
        self.action_dim = self.env.num_actions

        obs_dim = self.env.get_observations().shape[-1]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        self.amp_observation_dim = self.env.get_amp_observations().shape[-1]
        self.amp_files_path = list(self.env.amp_motion_files_path)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, _ = self.env.reset()
        return obs, {}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, _, reward, reset_buf, extras, reset_env_ids, terminal_amp_states = self.env.step(actions)
        info = dict(extras)
        info["reset_env_ids"] = reset_env_ids
        info["terminal_amp_obs"] = terminal_amp_states
        info["terminal_next_obs"] = info.pop("terminal_obs")
        return obs, reward, reset_buf.float(), info

    def get_amp_observations(self) -> torch.Tensor:
        return self.env.get_amp_observations()

    def get_terrain_level(self) -> float | None:
        if not self.env.cfg.terrain.curriculum:
            return None
        return self.env.terrain_levels.float().mean().item()

    def get_terrain_levels(self) -> np.ndarray | None:
        if not self.env.cfg.terrain.curriculum:
            return None
        return self.env.terrain_levels.cpu().numpy()

    def get_velocity_tracking(self) -> dict[str, float]:
        cmd_vx = self.env.commands[:, 0]
        act_vx = self.env.base_lin_vel[:, 0]
        return {
            "velocity/cmd_lin_vel_x": cmd_vx.mean().item(),
            "velocity/actual_lin_vel_x": act_vx.mean().item(),
            "velocity/track_err_lin_vel_x": (act_vx - cmd_vx).abs().mean().item(),
        }

    def get_velocity_histograms(self) -> dict[str, np.ndarray]:
        return {
            "velocity_hist/cmd_lin_vel_x": self.env.commands[:, 0].cpu().numpy(),
            "velocity_hist/actual_lin_vel_x": self.env.base_lin_vel[:, 0].cpu().numpy(),
        }
