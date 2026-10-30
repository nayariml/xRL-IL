# xRL

AMP-SAC for the Unitree Go2 for legged_gym (Isaac Gym), flat and rough terrain.

xRL holds only the learning side (agent, discriminator, replay buffer, logging).

In this version, the repository AMP_for_hardware was used as baseline and to load Isaac gym.


## Highlights

To support a large replay buffer, thousands of parallel environments and several updates per iteration, xRL manages precision, memory and devices explicitly:

- **Mixed precision:** on CUDA, the actor, critic and temperature losses run under `torch.autocast`. `GradScaler` scales each loss and unscales the gradients before clipping (`alg/ac_base.py`).
- **Memory cleanup:** rollouts are collected without gradients, and temporary rollout tensors are deleted once they are in the n-step buffer. `torch.cuda.empty_cache()` runs after collection and after each round of updates, and gradients are cleared with `set_to_none=True` (`alg/amp_sac.py`).
- **CPU and GPU:** simulation, networks and the n-step buffer stay on the GPU. The replay buffer stays on the CPU by default (`rl_buffer_device`), because 10⁷ transitions run out of GPU memory. New transitions move to the CPU when stored, and only each sampled batch moves back to the GPU (`memory/replay_buffer.py`).



## Installation

Clone this repository next to AMP_for_hardware, and run the commands in this README from their parent folder. Isaac Gym needs python 3.8.

```bash
git clone https://github.com/nayariml/xRL-IL.git
git clone https://github.com/nayariml/AMP_for_hardware.git

conda create -n xrl python=3.8
conda activate xrl

# Isaac Gym Preview 4, downloaded from NVIDIA
pip install -e <path/to/isaacgym>/python

pip install torch==2.2.2 numpy==1.24.4 gymnasium==0.29.1 hydra-core==1.3.2 \
    loguru wandb scipy pyquaternion ninja matplotlib termcolor \
    pybullet pygame lxml transformations tensorboard

pip install optuna pandas    # optional, only for hp_search_go2_amp.py


pip install -e xRL-IL

```

## How to use

Do not `pip install` AMP_for_hardware or its rsl_rl. 

The scripts add them to `sys.path` themselves (see `scripts/_legged_gym_path.py`). An installed
legged_gym from another checkout could otherwise get imported in their place.

Log in to wandb once with `wandb login`, or train with `tracking_training.wandb=False`.

By default the scripts use the `AMP_for_hardware` folder next to `xRL-IL`. To use another checkout, set `LEGGED_GYM_ROOT=<path/to/AMP_for_hardware>`.


AMP_for_hardware registers two tasks for xRL in `legged_gym/envs/__init__.py`:

Reference motions are loaded from `AMP_for_hardware/datasets/mocap_motions_go2`.


Both the env (reference state init) and the discriminator read them with
`legged_gym/utils/amp/motion_loader.py` (`AMPLoader`). Clips are sampled by the
`MotionWeight` in each file.

To check the setup without starting a simulation:

```bash
(cd xRL-IL/scripts && python -c "import isaacgym, _legged_gym_path, legged_gym, legged_gym.envs; print(legged_gym.LEGGED_GYM_ROOT_DIR)")
```

This should print the AMP_for_hardware path.

## Training

```bash
python xRL-IL/scripts/train.py env_name=go2_ampoff
python xRL-IL/scripts/train.py env_name=go2_ampoff_terrain
```

Config values can be overridden from the command line, e.g. `num_envs=4096 tracking_training.wandb=False`.
Resume with `artifact=<checkpoint>`.

The main configs are `cfg/default.yaml` (run settings), `cfg/alg/amp_sac.yaml`
(agent and discriminator) and `cfg/logging/default.yaml` (wandb project).

Runs are written to `outputs/<date>/<time>/<env_name>/`, with a checkpoint every
`save_interval` iterations and `model.pth` at the end.

## Testing

```bash
python xRL-IL/scripts/test.py env_name=go2_ampoff artifact=<run dir | model.pth> num_envs=64
```

This writes `test_metrics.json` (velocity tracking error, fall rate, imitation
distance) to the hydra run dir. Pass `headless=False` to watch the policy.

## Hyperparameter search

```bash
python xRL-IL/scripts/hp_search_go2_amp.py --env-name go2_ampoff --n-trials 30
```

Every trial runs a training and a test as separate processes, and they inherit `LEGGED_GYM_ROOT`.

See `--help` for the other options.

## Notes

- Import errors from libstdc++ when Isaac Gym loads: set the `LD_PRELOAD`.

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

```bash
export LD_LIBRARY_PATH=$(conda info --base)/envs/xrl/lib/:$LD_LIBRARY_PATH
``` 

## Acknowledgements

xRL builds on **[PQL](https://github.com/Improbable-AI/pql)** (Improbable AI Lab, MIT License, 2023), from which it inherits the `ActorCriticBase` structure, SAC skeletons, `SquashedNormal`, `TanhTransform`, `RunningMeanStd`, and the Hydra search-path plugin.


## Citation:

```bibtex
@inproceedings{lessa2026unlocking,
  title={Unlocking the Potential of Soft Actor--Critic for Imitation Learning},
  author={Lessa, Nayari Marie and Boukheddimi, Melya and Kirchner, Frank},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6105--6112},
  year={2026},
  organization={IEEE}
}

```

