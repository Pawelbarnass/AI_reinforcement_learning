import gymnasium as gym
import sys
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

sys.modules['gym'] = gym

# 1. Download the model file from Hugging Face
# This downloads the file to your computer and returns the local path
checkpoint_path = load_from_hub(
	repo_id="sb3/ppo-LunarLander-v2",
	filename="ppo-LunarLander-v2.zip",
)
print(f"Model downloaded to: {checkpoint_path}")

# 2. Load the environment
# Note: The pre-trained model was likely trained on v2. 
# Gymnasium has v3, but they are physically very similar.
# If v3 crashes, try installing `gymnasium[box2d]` and using "LunarLander-v2"
env = gym.make("LunarLander-v3", render_mode="human")

# 3. Load the model
# We use custom_objects to ensure it loads even if the python versions differ slightly
model = PPO.load(checkpoint_path, env=env, device="cpu", custom_objects={"observation_space": env.observation_space, "action_space": env.action_space})

# 4. Enjoy
episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    score = 0
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        
    print(f"Episode {ep+1} Score: {score}")

env.close()