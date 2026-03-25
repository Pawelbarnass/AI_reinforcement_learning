# Ewaluacja wytrenowanego agenta RL z renderowaniem (LunarLander-v3)
# Uruchom: python evaluate.py

import gymnasium as gym
from stable_baselines3 import PPO
import time

ENV_ID = "LunarLander-v3"  # Upewnij się, że zgadza się z treningiem
MODEL_PATH = "ppo_lunarlander"
EPISODES = 5

# Tworzenie środowiska z renderowaniem
env = gym.make(ENV_ID, render_mode="human")

# Wczytanie modelu
model = PPO.load(MODEL_PATH)

for ep in range(EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.02)  # Spowolnienie dla lepszej widoczności
    print(f"Epizod {ep+1}: suma nagród = {total_reward:.2f}")

env.close()
