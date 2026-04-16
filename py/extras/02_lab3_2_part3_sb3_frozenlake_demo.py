# Local support imports for this extracted multi-cell file.
# Notebook-derived blocks below keep per-cell source markers for easier copy-back.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*60}")
print(f"Device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Gymnasium Version: {gym.__version__}")
print(f"Torch Threads: {torch.get_num_threads()}")
print(f"{'='*60}\n")

# Source: notebooks/Lab3.2-st.ipynb code cell 32

# PART 3: DQN with Stable-Baselines3
### 3.1: Import Stable-Baselines3

print("\n" + "="*60)
print("PART 3: STABLE-BASELINES3 DQN ON FROZENLAKE-V1")
print("="*60)

print("\n3.1: Import Stable-Baselines3")
print("-" * 60)

try:
    from stable_baselines3 import DQN
    sb3_available = True
    print("✓ Stable-Baselines3 imported successfully")
except Exception as e:
    sb3_available = False
    print("⚠ SB3 import failed in this environment")
    print(f"  Error: {type(e).__name__}: {e}")
    print("  Please check package compatibility in env nt549-1-phatpt.")

# Source: notebooks/Lab3.2-st.ipynb code cell 34

# PART 3: DQN with Stable-Baselines3
### 3.2: Train SB3 DQN on FrozenLake-v1 (Simple)

if sb3_available:
    print("\n3.2: Training SB3 DQN on FrozenLake-v1")
    print("-" * 60)

    ### YOU NEED TO WRITE YOUR CODE BELOW ###
    # 1) Create FrozenLake-v1 environment
    # 2) Initialize SB3 DQN model
    # 3) Train model with suitable timesteps
    env_sb3 = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

    model = DQN(
        policy='MlpPolicy',
        env=env_sb3,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=200,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=250,
        verbose=0
    )

    total_timesteps = 10000
    print(f"Training for {total_timesteps} timesteps...\n")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed!\n")

    print("3.3: Evaluation of SB3 DQN on FrozenLake-v1")
    print("-" * 60)

    eval_episodes = 30
    sb3_eval_rewards = []

    for _ in range(eval_episodes):
        obs, _ = env_sb3.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, reward, terminated, truncated, _ = env_sb3.step(action)
            done = terminated or truncated
            episode_reward += reward

        sb3_eval_rewards.append(episode_reward)

    success_rate = sum(1 for r in sb3_eval_rewards if r > 0) / eval_episodes * 100
    print("SB3 Evaluation Results:")
    print(f"  Average Reward: {np.mean(sb3_eval_rewards):.2f}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Min Reward: {np.min(sb3_eval_rewards):.2f}")
    print(f"  Max Reward: {np.max(sb3_eval_rewards):.2f}")

    env_sb3.close()
else:
    print("\n⚠ Skipping SB3 training because SB3 import was not successful.")

# Source: notebooks/Lab3.2-st.ipynb code cell 36

# PART 3: DQN with Stable-Baselines3
### 3.4: Key Advantages of Stable-Baselines3

print("\n3.4: Key Advantages of Stable-Baselines3")
print("="*60)

print("""
1. Optimized Implementation:
   - Highly optimized algorithms (better than manual implementation)
   - Less error-prone (extensively tested)
   - Better hyperparameter tuning utilities

2. Features:
   - Built-in support for multiple algorithms
   - Vectorized environment support
   - Automatic GPU acceleration
   - Callbacks and monitoring

3. Production Ready:
   - Used in industry and research
   - Active community and support
   - Benchmarked and validated

4. When to Use:
   - Quick prototyping: Use SB3
   - Learning fundamentals: Use manual implementation
   - Production systems: Use SB3
   - Custom algorithms: Use PyTorch/TensorFlow directly
""")

print("="*60)
