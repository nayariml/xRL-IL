import glob
import os

import torch
from hydra.utils import to_absolute_path
from loguru import logger

def _resolve_checkpoint(artifact: str, env_name: str) -> str:
    # relative to where the script was launched, hydra changes the cwd
    artifact = to_absolute_path(artifact)
    if os.path.isfile(artifact):
        return artifact

    # train.py saves into <run dir>/<env_name>
    folders = (artifact, os.path.join(artifact, env_name))
    for folder in folders:
        path = os.path.join(folder, "model.pth")
        if os.path.isfile(path):
            return path
    candidates = [c for folder in folders for pattern in ("*.pth", "*.pt")
                  for c in glob.glob(os.path.join(folder, pattern))]
    if not candidates:
        raise FileNotFoundError(f"No weights found in {artifact}")
    return max(candidates, key=os.path.getmtime)


def load_model(model, model_type, cfg):
    """
    Loads weights into a model from a checkpoint file or run dir.
    """
    path = _resolve_checkpoint(cfg.artifact, cfg.env_name)
    logger.info(f'Loading {model_type} from {path} to {cfg.device}')
    checkpoint = torch.load(path, map_location=cfg.device)

    weights = checkpoint[model_type]
    if weights is None:
        logger.warning(f'{model_type} not stored in the checkpoint.')
        return model
    model.load_state_dict(weights)
    return model


def save_model(path, actor, critic, rms):
    torch.save({'obs_rms': rms, 'actor': actor, 'critic': critic}, path)
