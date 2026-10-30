import isaacgym  # before torch

import _legged_gym_path  # noqa: F401

# motion files are globbed relative to the cwd, import before hydra changes it
import legged_gym
import legged_gym.envs

import time
from itertools import count

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

import xRL
from xRL.alg import alg_name_to_path
from xRL.memory.replay_buffer import ReplayBuffer
from xRL.utils.common import WandbLogger, capture_keyboard_interrupt, create_folder, load_class_from_path
from xRL.utils.model_util import load_model, save_model
from xRL.wrappers.create_environment import create_task_env


def _create_tracker(cfg):
    if not cfg.tracking_training.wandb:
        return None
    try:
        return WandbLogger(cfg)
    except Exception as e:
        logger.warning(f"wandb init failed ({e}), continuing without tracking.")
        return None


def _episode_stats(ep_infos):
    if not ep_infos:
        return {}
    return {f"episode/{key}": float(np.mean([float(ep[key]) for ep in ep_infos]))
            for key in ep_infos[0]}


@hydra.main(config_path=xRL.LIB_PATH.joinpath("cfg").as_posix(), config_name="default")
def main(cfg: DictConfig):
    capture_keyboard_interrupt()
    logger.info(f"legged_gym: {legged_gym.LEGGED_GYM_ROOT_DIR}")
    tracker = _create_tracker(cfg)

    log_dir = cfg.env_name
    create_folder(log_dir)

    env = create_task_env(cfg)

    alg_name = cfg.alg.name if cfg.alg.name.endswith("Agent") else cfg.alg.name + "Agent"
    logger.info(f"Initializing agent: {alg_name}")
    agent = load_class_from_path(alg_name, alg_name_to_path[alg_name])(env=env, cfg=cfg)

    if cfg.artifact is not None:
        load_model(agent.actor, "actor", cfg)
        load_model(agent.critic, "critic", cfg)
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        if cfg.alg.obs_norm:
            load_model(agent.obs_rms, "obs_rms", cfg)

    agent.reset_agent()
    logger.info(f"obs_dim={agent.obs_dim}  action_dim={agent.action_dim}  amp_obs_dim={agent.amp_obs_dim}")

    memory = ReplayBuffer(
        capacity=int(cfg.alg.memory_size),
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        add_obs_dim=agent.amp_obs_dim,
        device=cfg.rl_buffer_device,
    )

    start_time = time.time()
    trajectory, global_steps, _ = agent.collect_transitions(env, cfg.warm_up, random=True)
    if trajectory is not None:
        memory.add(trajectory)

    for iter_t in count():
        trajectory, steps, ep_infos = agent.collect_transitions(env, cfg.alg.num_interac_per_env, random=False)
        global_steps += steps
        if trajectory is not None:
            memory.add(trajectory)
        log_info = agent.update(memory)

        if iter_t % cfg.log_freq == 0:
            elapsed = time.time() - start_time
            fps = global_steps / max(elapsed, 1.0)
            log_info.update(_episode_stats(ep_infos))
            log_info.update(env.get_velocity_tracking())
            log_info.update({"system/fps": fps, "system/elapsed_s": elapsed,
                             "system/buffer_size": memory.size, "system/iteration": iter_t})
            terrain_level = env.get_terrain_level()
            if terrain_level is not None:
                log_info["terrain/level_mean"] = terrain_level

            logger.info(
                f"iter={iter_t:8d}  steps={global_steps:10.2e}  "
                f"fps={fps:6.0f}  time={elapsed:7.1f}s  "
                f"return={log_info['train/return']:8.2f}  "
                f"critic_loss={log_info['train/critic_loss']:8.4f}  "
                f"actor_loss={log_info['train/actor_loss']:8.4f}  "
                f"alpha={log_info['train/entropy']:6.4f}  "
                f"vel_x_err={log_info['velocity/track_err_lin_vel_x']:.2f}"
                + (f"  terrain={terrain_level:.2f}" if terrain_level is not None else "")
            )
            if tracker is not None:
                tracker.log(log_info, step=global_steps)
                hists = {k: v for k, v in log_info.items() if isinstance(v, np.ndarray)}
                hists.update(env.get_velocity_histograms())
                terrain_levels = env.get_terrain_levels()
                if terrain_levels is not None:
                    hists["terrain/levels"] = terrain_levels
                for key, values in hists.items():
                    tracker.log_histogram(key, values, step=global_steps)

        finished = (
            (cfg.max_step is not None and global_steps > cfg.max_step)
            or (cfg.max_time is not None and time.time() - start_time > cfg.max_time)
            or (cfg.max_iterations is not None and iter_t >= cfg.max_iterations - 1)
        )

        if iter_t % cfg.save_interval == 0 or finished:
            save_path = f"{log_dir}/model.pth" if finished else f"{log_dir}/_{iter_t}_model.pth"
            save_model(
                path=save_path,
                actor=agent.actor.state_dict(),
                critic=agent.critic.state_dict(),
                rms=agent.obs_rms.get_states() if cfg.alg.obs_norm else None,
            )
            if tracker is not None and cfg.tracking_training.upload_artifacts:
                tracker.upload_model("final_model" if finished else f"checkpoint_{iter_t}", save_path)

        if finished:
            logger.info(f"Training finished after {iter_t + 1} iterations, {global_steps:.2e} steps.")
            break

    if tracker is not None:
        tracker.close()


if __name__ == "__main__":
    main()
