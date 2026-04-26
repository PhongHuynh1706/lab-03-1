# Source: notebooks/Lab3.2-st.ipynb code cell 41

# PART 4: Student Exercise - SB3 DQN on VacuumCleanerEnv
### 4.2: Evaluate SB3 on VacuumCleanerEnv

# Evaluation template
### YOU NEED TO WRITE YOUR CODE BELOW ###
# Suggested structure:
# eval_episodes = 30
# rewards = []
# clean_ratios = []
# completions = []
# for _ in range(eval_episodes):
#
#     # collect metrics from environment
eval_episodes = 30
rewards = []
clean_ratios = []
completions = []

env = VacuumCleanerEnv()

for _ in range(eval_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model_sb3_vacuum.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward

    rewards.append(episode_reward)

    cleaned = (env.num_cells - 1) - len(env.dirty_cells)
    clean_ratio = cleaned / (env.num_cells - 1)
    clean_ratios.append(clean_ratio)

    completions.append(1 if len(env.dirty_cells) == 0 else 0)

success_rate = np.mean(completions) * 100

print(f"Average Reward: {np.mean(rewards):.2f}")
print(f"Average Clean Ratio: {np.mean(clean_ratios):.2f}")
print(f"Success Rate: {success_rate:.1f}%")

print("Please complete evaluation code for Part 4.")

