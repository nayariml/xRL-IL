import ast
import importlib.util
import os
import signal
import sys
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf


class Tracker:
    def __init__(self, max_len):
        self.moving_average = deque([0 for _ in range(max_len)], maxlen=max_len)

    def __repr__(self):
        return self.moving_average.__repr__()

    def update(self, value):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self):
        return np.mean(self.moving_average)


def _to_float(v) -> Optional[float]:
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.item())
    return None


class WandbLogger:
    def __init__(self, cfg):
        import wandb

        self._wandb = wandb
        wandb_cfg = cfg.logging.wandb
        task_name = f"{cfg.env_name} | {cfg.alg.name}"
        tags = list(wandb_cfg.get("tags", []))
        disc_loss_type = cfg.alg.get("disc_loss_type", None)
        if disc_loss_type is not None:
            tags.append(f"disc:{disc_loss_type}")

        wandb_dir = wandb_cfg.get("dir", None)
        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
            os.environ.setdefault("WANDB_DATA_DIR", str(wandb_dir))

        self.run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.get("entity", None),
            name=task_name,
            group=wandb_cfg.get("group", None),
            tags=tags or None,
            dir=wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        )
        logger.info(f"[Wandb] Run '{task_name}' initialised (project: {wandb_cfg.project})")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        log_dict = {k: _to_float(v) for k, v in metrics.items()}
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        if log_dict:
            self._wandb.log(log_dict, step=step)

    def log_histogram(self, key: str, values: np.ndarray, step: int) -> None:
        self._wandb.log({key: self._wandb.Histogram(values)}, step=step)

    def upload_model(self, name: str, path: str) -> None:
        artifact = self._wandb.Artifact(name=name.replace(" ", "_"), type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def close(self) -> None:
        self._wandb.finish()


def load_class_from_path(cls_name, path):
    mod_name = 'MOD%s' % cls_name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def list_class_names(dir_path):
    cls_name_to_path = dict()
    for py_file in Path(dir_path).rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        node = ast.parse(py_file.read_text())
        for cls in (n for n in node.body if isinstance(n, ast.ClassDef)):
            cls_name_to_path[cls.name] = py_file
    return cls_name_to_path


def capture_keyboard_interrupt():
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
