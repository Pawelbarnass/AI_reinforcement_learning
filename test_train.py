import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# 1. Directory to save logs and models
models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 2. Create the Environment
# We use make_vec_env to create multiple parallel environments. 
# This speeds up training significantly on a CPU, feeding data to the GPU.
env_id = "LunarLander-v3"
n_envs = 4  # Number of parallel games to play at once
env = make_vec_env(env_id, n_envs=n_envs)

# 3. Define the Model (The Agent)
# MlpPolicy = Multi-Layer Perceptron (Standard Neural Network)
# device="auto" will use CUDA or MPS if available
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir, 
    device="cpu" 
)

print("-----------------------------------------")
print(f"Training on device: {model.device}")
print("-----------------------------------------")

# 4. Train the Model
# LunarLander usually takes between 100k to 1M timesteps to solve perfectly.
TIMESTEPS = 100000 
iters = 0

while True:
    iters += 1
    
    # Train for a chunk of time
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # Save the model
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    
    print(f"Iteration {iters} complete. Model saved.")
    
    # Stop after some iterations (e.g., 1 million steps total)
    if TIMESTEPS * iters >= 1000000:
        break