#!/usr/bin/env python3
"""Hyperparameter search for AMP-SAC imitation on the Go2 legged_gym envs.

Each trial trains with scripts/train.py, then evaluates the final policy with
scripts/test.py (deterministic actions) and minimises

    imitation_dist + w_vel * vel_x_track_err + w_fall * fall_rate


Usage (from any directory, legged_gym root picked as in train.py):
    LEGGED_GYM_ROOT=~/projects/AMP_for_hardware \\
    python xRL/scripts/hp_search_go2_amp.py --n-trials 30 --num-envs 4096 --max-iterations 2000

The study is stored in <output-dir>/study.db, rerunning the same command
resumes it. Per trial: trial_NNN/params.json, metrics.json and the train/test
run dirs with their logs. The best config ends up in <output-dir>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from loguru import logger
from optuna.samplers import TPESampler

SCRIPTS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPTS_DIR / "train.py"
TEST_SCRIPT = SCRIPTS_DIR / "test.py"


def _round_sig(x: float, sig: int = 2) -> float:
    return round(x, sig - 1 - math.floor(math.log10(abs(x)))) if x else 0.0


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Hydra overrides for one trial."""
    params = {
        # SAC
        "alg.actor_lr": _round_sig(trial.suggest_float("actor_lr", 1e-4, 2e-3, log=True)),
        "alg.critic_lr": _round_sig(trial.suggest_float("critic_lr", 1e-4, 2e-3, log=True)),
        "alg.alpha_lr": _round_sig(trial.suggest_float("alpha_lr", 1e-4, 3e-3, log=True)),
        "alg.gamma": _round_sig(trial.suggest_float("gamma", 0.95, 0.995), 3),
        "alg.tau": _round_sig(trial.suggest_float("tau", 0.005, 0.1, log=True)),
        "alg.batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "alg.update_times": trial.suggest_int("update_times", 2, 8),
        "alg.nstep": trial.suggest_int("nstep", 1, 5),
        # AMP discriminator
        "alg.disc_reward_scale": _round_sig(trial.suggest_float("disc_reward_scale", 0.3, 3.0, log=True)),
        "alg.disc_loss_type": trial.suggest_categorical("disc_loss_type", ["bce", "lsgan"]),
        "alg.disc_grad_penalty": _round_sig(trial.suggest_float("disc_grad_penalty", 1.0, 50.0, log=True)),
        "alg.discr_loss_coef": _round_sig(trial.suggest_float("discr_loss_coef", 0.01, 0.5, log=True)),
        "alg.grad_pen_loss_coef": _round_sig(trial.suggest_float("grad_pen_loss_coef", 1e-3, 0.1, log=True)),
        "alg.disc_logit_reg_coef": _round_sig(trial.suggest_float("disc_logit_reg_coef", 5e-3, 0.1, log=True)),
    }
    # style vs task blend, fixed by default: optimising it against an imitation
    # objective just pushes it towards pure imitation
    params["alg.disc_reward_weight"] = (
        _round_sig(trial.suggest_float("disc_reward_weight", 0.2, 0.9))
        if args.tune_disc_weight else args.disc_reward_weight
    )
    return params


def _run(cmd: list[str], log_path: Path, timeout: float | None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"exit code {proc.returncode}, see {log_path}")


def run_seed(params: dict, seed: int, seed_dir: Path, args: argparse.Namespace) -> dict:
    common = [f"env_name={args.env_name}", f"seed={seed}",
              *args.overrides]
    overrides = [f"{k}={v}" for k, v in params.items()]

    train_dir = seed_dir / "train"
    _run([sys.executable, str(TRAIN_SCRIPT), *common, *overrides,
          f"num_envs={args.num_envs}", f"max_iterations={args.max_iterations}",
          f"save_interval={args.max_iterations}",
          f"tracking_training.wandb={args.wandb}",
          f"logging.wandb.group={args.study_name}",
          f"hydra.run.dir={train_dir}"],
         seed_dir / "train.log", args.trial_timeout)

    # train.py saves into <run dir>/<env_name>/model.pth
    model = train_dir / args.env_name / "model.pth"
    if not model.is_file():
        raise RuntimeError(f"no model at {model}")

    test_dir = seed_dir / "test"
    _run([sys.executable, str(TEST_SCRIPT), *common, *overrides,
          f"num_envs={args.test_envs}", f"artifact={model}",
          "tracking_training.wandb=False",
          *([f"+test_steps={args.test_steps}"] if args.test_steps else []),
          f"hydra.run.dir={test_dir}"],
         seed_dir / "test.log", args.trial_timeout)
    return json.loads((test_dir / "test_metrics.json").read_text())


def objective_value(m: dict, args: argparse.Namespace) -> float:
    return m["imitation_dist"] + args.w_vel * m["vel_x_track_err"] + args.w_fall * m["fall_rate"]


def run_trial(trial: optuna.Trial, args: argparse.Namespace) -> float:
    params = suggest_params(trial, args)
    trial_dir = Path(args.output_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "params.json").write_text(json.dumps(params, indent=2))
    logger.info(f"trial {trial.number}: " + "  ".join(f"{k[4:]}={v}" for k, v in params.items()))

    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        m = run_seed(params, seed, trial_dir / f"seed_{seed}", args)
        m["objective"] = objective_value(m, args)
        per_seed.append(m)
        logger.info(f"trial {trial.number} seed {seed}: objective={m['objective']:.4f}  "
                    f"imitation={m['imitation_dist']:.3f}  vel_err={m['vel_x_track_err']:.3f}  "
                    f"fall={m['fall_rate']:.2%}")

    metrics = {k: float(np.mean([m[k] for m in per_seed]))
               for k in per_seed[0] if isinstance(per_seed[0][k], float)}
    metrics["elapsed_s"] = time.time() - t0
    (trial_dir / "metrics.json").write_text(json.dumps({"mean": metrics, "per_seed": per_seed}, indent=2))
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return metrics["objective"]


def write_summary(study: optuna.Study, args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("no trial completed, check trial_*/seed_*/train.log")
        return

    best = study.best_trial
    best_overrides = json.loads((out / f"trial_{best.number:03d}" / "params.json").read_text())
    override_str = " ".join(f"{k}={v}" for k, v in best_overrides.items())
    summary = {
        "objective": f"imitation_dist + {args.w_vel}*vel_x_track_err + {args.w_fall}*fall_rate",
        "best_trial": best.number,
        "best_value": best.value,
        "best_metrics": best.user_attrs,
        "best_overrides": best_overrides,
        "n_completed": len(completed),
        "n_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "reproduce_cmd": f"python xRL/scripts/train.py env_name={args.env_name} {override_str}",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"best trial {best.number}: objective={best.value:.4f}  "
                f"imitation={best.user_attrs.get('imitation_dist', float('nan')):.3f}")
    logger.info(f"reproduce: {summary['reproduce_cmd']}")
    logger.info(f"summary -> {out / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--env-name", default="go2_ampoff", help="go2_ampoff | go2_ampoff_terrain")
    ap.add_argument("--num-envs", type=int, default=4096, help="training envs per trial")
    ap.add_argument("--max-iterations", type=int, default=2000, help="training iterations per trial")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="seeds per trial, objective is their mean")
    ap.add_argument("--test-envs", type=int, default=256)
    ap.add_argument("--test-steps", type=int, default=None, help="default: one episode length")
    ap.add_argument("--w-vel", type=float, default=1.0, help="objective weight of vel_x tracking error [m/s]")
    ap.add_argument("--w-fall", type=float, default=2.0, help="objective weight of the fall rate")
    ap.add_argument("--disc-reward-weight", type=float, default=0.4, help="fixed style/task blend")
    ap.add_argument("--tune-disc-weight", action="store_true", help="search disc_reward_weight too")
    ap.add_argument("--trial-timeout", type=float, default=None, help="seconds per train/test run")
    ap.add_argument("--wandb", action="store_true", help="log training runs to wandb, grouped by study")
    ap.add_argument("--output-dir", default="hp_search_go2_amp")
    ap.add_argument("--study-name", default="go2_amp_sac")
    ap.add_argument("overrides", nargs="*", help="extra hydra overrides for every run, e.g. alg.memory_size=1e6")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    args.output_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=8),
        storage=f"sqlite:///{out / 'study.db'}",
        load_if_exists=True,
    )
    remaining = args.n_trials - len(study.trials)
    logger.info(f"study {args.study_name}: {len(study.trials)} trials done, running {max(remaining, 0)} more "
                f"({args.num_envs} envs x {args.max_iterations} iterations, seeds {args.seeds}) -> {out}")
    if remaining > 0:
        study.optimize(lambda t: run_trial(t, args), n_trials=remaining, catch=(Exception,))
    write_summary(study, args)


if __name__ == "__main__":
    main()
