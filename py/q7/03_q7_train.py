# Source: notebooks/Lab3.2-st.ipynb code cell 39

# PART 4: Student Exercise - SB3 DQN on VacuumCleanerEnv
### 4.1: Implement SB3 DQN for VacuumCleanerEnv

print("\n" + "="*60)
print("PART 4: SB3 DQN ON VACUUMCLEANERENV")
print("="*60)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from logger import setup_csv_logger
# Step 1: Create environment
### YOU NEED TO WRITE YOUR CODE BELOW ###
env_sb3_vacuum = make_vec_env(lambda: VacuumCleanerEnv(), n_envs=1)

# Step 2: Build SB3 DQN model
# Suggested parameters to try:
# - learning_rate: 1e-3 or 5e-4
# - buffer_size: 10000 or 20000
# - batch_size: 32 or 64
### YOU NEED TO WRITE YOUR CODE BELOW ###
model_sb3_vacuum = DQN(
    "MlpPolicy",
    env_sb3_vacuum,
    learning_rate=5e-4,
    buffer_size=20000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=0,
    device=device
)

# Step 3: Train
### YOU NEED TO WRITE YOUR CODE BELOW ###
setup_csv_logger(model_sb3_vacuum)

total_timesteps = 50000
model_sb3_vacuum.learn(total_timesteps=total_timesteps)

print("Training finished (if implemented).")
