# Source: notebooks/Lab3.2-st.ipynb code cell 17

# PART 1: DQN on VacuumCleanerEnv
### 1.3: Evaluation and Visualization

# Tier-based evaluation:
# - Basic success: cleaned >= 20%
# - Good success : cleaned >= 50%
# - Full success : cleaned 100%

print("\n1.3: Evaluation on VacuumCleanerEnv")
print("-" * 60)

eval_episodes = 30
eval_rewards = []
eval_clean_ratios = []
eval_full_completed = []

print(f"\nEvaluating on {eval_episodes} episodes (greedy policy)...\n")

for _ in range(eval_episodes):
    state, _ = env1.reset()
    episode_reward = 0.0
    done = False
    terminated = False

    while not done:
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        action = None
        next_state, reward, terminated, truncated, _ = env1.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state

    cleaned = (env1.num_cells - 1) - len(env1.dirty_cells)
    clean_ratio = cleaned / (env1.num_cells - 1)
    eval_rewards.append(episode_reward)
    eval_clean_ratios.append(clean_ratio)
    eval_full_completed.append(bool(terminated))

# Instruction:
# Compute average reward, average clean ratio, and success rates for each tier...
### YOU NEED TO WRITE YOUR CODE BELOW ###
basic_success_rate = None
good_success_rate = None
full_completion_rate = None
avg_clean_ratio = None

print("Evaluation Results:")
print(f"  Average Reward: {np.mean(eval_rewards):.2f}")
print(f"  Average Clean Ratio: {avg_clean_ratio:.1f}%")
print(f"  Basic Success Rate (>=20% cleaned): {basic_success_rate:.1f}%")
print(f"  Good Success Rate (>=50% cleaned): {good_success_rate:.1f}%")
print(f"  Full Completion Rate (100% cleaned): {full_completion_rate:.1f}%")
print(f"  Min/Max Reward: {np.min(eval_rewards):.2f} / {np.max(eval_rewards):.2f}")

# Source: notebooks/Lab3.2-st.ipynb code cell 19

# PART 1: DQN on VacuumCleanerEnv
### 1.4: Policy Rollout Demo (Greedy)

# Show one greedy rollout to visualize the learned policy behavior
print("\n1.4: Policy Rollout Demo (Greedy)")
print("-" * 60)

action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

state, _ = env1.reset(seed=123)
done = False
demo_reward = 0.0
demo_max_steps = 12  # Keep output compact but informative
step_id = 0

print("Initial state:")
print(env1.render(mode='ansi'))
print()

while (not done) and (step_id < demo_max_steps):
    action = agent1.select_action(state, training=False)
    next_state, reward, terminated, truncated, _ = env1.step(action)
    done = terminated or truncated
    step_id += 1
    demo_reward += reward

    print(f"Step {step_id:02d} | Action: {action_names[action]:>5} | Reward: {reward:>5.1f} | Done: {done}")
    print(env1.render(mode='ansi'))
    print()

    state = next_state

cleaned = (env1.num_cells - 1) - len(env1.dirty_cells)
clean_ratio = cleaned / (env1.num_cells - 1)
print("Demo summary:")
print(f"  Total reward: {demo_reward:.2f}")
print(f"  Cleaned: {cleaned}/{env1.num_cells - 1} ({clean_ratio:.1%})")
print(f"  Full completion: {'Yes' if len(env1.dirty_cells) == 0 else 'No'}")

# Source: notebooks/Lab3.2-st.ipynb code cell 20

# PART 1: DQN on VacuumCleanerEnv
### 1.4: Policy Rollout Demo (Greedy)

# Left plot: training rewards + moving average
# Right plot: distribution by cleaning-quality tiers

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Plot 1: training curve
axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
ma_window = 30
ma_rewards = np.convolve(episode_rewards, np.ones(ma_window)/ma_window, mode='valid')
axes[0].plot(range(ma_window-1, len(episode_rewards)), ma_rewards, label=f'MA({ma_window})', color='red', linewidth=2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Episode Reward')
axes[0].set_title('Part 1: VacuumCleanerEnv Training Progress', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: quality tiers based on clean ratio
clean_ratios = np.array(eval_clean_ratios)
tier_labels = ['<20%', '20-50%', '50-99%', '100%']
tier_counts = [
    int(np.sum(clean_ratios < 0.20)),
    int(np.sum((clean_ratios >= 0.20) & (clean_ratios < 0.50))),
    int(np.sum((clean_ratios >= 0.50) & (clean_ratios < 1.00))),
    int(np.sum(clean_ratios >= 1.00))
]
tier_colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']

axes[1].bar(tier_labels, tier_counts, color=tier_colors)
axes[1].set_ylabel('Number of Episodes')
axes[1].set_title(
    f'Evaluation Quality Tiers | Avg Clean: {avg_clean_ratio:.1f}% | Basic Success: {basic_success_rate:.1f}%',
    fontsize=10,
    fontweight='bold'
)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close(fig)  # Free figure memory in notebook runtime
