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
# =========================
# 2.6: Visualization (FIXED - SEPARATE LR PLOTS)
# =========================

learning_rates = [1e-2, 1e-3, 5e-4]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for idx, lr in enumerate(learning_rates):

    rewards_all_seeds = results[lr]["reward"]

    # average over seeds (nếu có nhiều seed)
    max_len = min(len(r) for r in rewards_all_seeds)
    rewards_all_seeds = [r[:max_len] for r in rewards_all_seeds]

    avg_rewards = np.mean(rewards_all_seeds, axis=0)

    # moving average
    ma_window = 50
    if len(avg_rewards) >= ma_window:
        ma_rewards = np.convolve(
            avg_rewards,
            np.ones(ma_window) / ma_window,
            mode='valid'
        )
        axes[idx].plot(
            range(ma_window - 1, len(avg_rewards)),
            ma_rewards,
            color='red',
            label=f'MA({ma_window})'
        )

    axes[idx].plot(avg_rewards, alpha=0.4, label='Reward')
    axes[idx].set_title(f'Learning Rate = {lr}')
    axes[idx].set_xlabel('Episode')
    axes[idx].set_ylabel('Reward')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for idx, lr in enumerate(learning_rates):

    losses_all_seeds = results[lr]["loss"]

    # cắt cùng length
    max_len = min(len(l) for l in losses_all_seeds)
    losses_all_seeds = [l[:max_len] for l in losses_all_seeds]

    avg_loss = np.mean(losses_all_seeds, axis=0)

    axes[idx].plot(avg_loss, label="Loss")
    axes[idx].set_title(f"Loss Curve (lr={lr})")
    axes[idx].set_xlabel("Training Step")
    axes[idx].set_ylabel("MSE Loss")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

plt.tight_layout()
plt.show()