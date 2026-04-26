# Source: notebooks/Lab3.1-st.ipynb code cell 25

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1
### 2.5: Evaluation on FrozenLake

print("\n2.5: Evaluation with Greedy Policy")
print("-" * 60)

eval_episodes = 30
eval_rewards = []

print(f"\nEvaluating on {eval_episodes} episodes (greedy policy)...\n")

for episode in range(eval_episodes):
    state, _ = env_train.reset()
    episode_reward = 0
    done = False

    while not done:
        # Instruction:
        # Use greedy action (training=False), interact with env, and accumulate reward.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        action = agent.select_action(state, training=False)
        
        next_state, reward, terminated, truncated, _ = env_train.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state

    eval_rewards.append(episode_reward)

success_rate = sum(1 for r in eval_rewards if r > 0) / eval_episodes * 100
print(f"Evaluation Results:")
print(f"  Average Reward: {np.mean(eval_rewards):.2f}")
print(f"  Success Rate: {success_rate:.1f}%")
print(f"  Min Reward: {np.min(eval_rewards):.2f}")
print(f"  Max Reward: {np.max(eval_rewards):.2f}")


# Source: notebooks/Lab3.1-st.ipynb code cell 27

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1
### 2.6: Visualization of Training Progress

print("\n2.6: Visualization")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Plot 1: Training rewards with moving average
axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
ma_window = 50
ma_rewards = np.convolve(episode_rewards, np.ones(ma_window)/ma_window, mode='valid')
axes[0].plot(range(ma_window-1, len(episode_rewards)), ma_rewards, label=f'MA({ma_window})', color='red', linewidth=2)
axes[0].set_xlabel('Episode', fontsize=11)
axes[0].set_ylabel('Episode Reward', fontsize=11)
axes[0].set_title('Training Progress: FrozenLake-v1', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Evaluation reward distribution
axes[1].bar(['Failed', 'Success'], [sum(1 for r in eval_rewards if r == 0), sum(1 for r in eval_rewards if r > 0)])
axes[1].set_ylabel('Number of Episodes', fontsize=11)
axes[1].set_title(f'Evaluation Results (Success Rate: {success_rate:.1f}%)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close(fig)
