"""Upgraded PPO training for LunarLander with resume + eval.

Uruchom bez parametrów lub z nadpisaniem:
    python train.py --total-timesteps 500000 --n-envs 8 --device auto
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="LunarLander-v3", help="Gymnasium environment id")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total train steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Vectorized environments")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Steps between evals (env steps)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--model-path", default="models/PPO/latest", help="Checkpoint path (without .zip)")
    return parser.parse_args()


def main():
    args = parse_args()

    models_dir = os.path.dirname(args.model_path) or "."
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Deterministic-ish runs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Vectorized env speeds up PPO data collection
    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
    eval_env = gym.make(args.env_id)

    if os.path.exists(args.model_path + ".zip"):
        print(f"Loading existing checkpoint {args.model_path}.zip and resuming...")
        model = PPO.load(args.model_path, env=env, device=args.device)
    else:
        print("Training new model from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
        )

    # EvalCallback saves best model based on mean reward
    eval_freq_vec_steps = max(1, args.eval_freq // args.n_envs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=eval_freq_vec_steps,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, reset_num_timesteps=False)

    # Save latest checkpoint; best model saved by callback
    model.save(args.model_path)
    print(f"Training complete. Latest checkpoint: {args.model_path}.zip")
    print(f"Best model (by EvalCallback) in: {models_dir}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
