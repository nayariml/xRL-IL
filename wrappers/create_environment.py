from xRL.utils.torch_util import set_device, set_seed
from xRL.wrappers.legged_gym_vector_env import LeggedGymWrapper


def create_task_env(cfg):
    device = set_device(cfg.device, cfg.device_id) if cfg.device == "cuda" else cfg.device
    set_seed(cfg.seed)
    return LeggedGymWrapper(task=cfg.env_name, num_envs=cfg.num_envs, seed=cfg.seed,
                            headless=cfg.headless, sim_device=device)
