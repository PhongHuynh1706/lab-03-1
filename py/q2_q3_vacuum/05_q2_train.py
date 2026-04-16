# Source: notebooks/Lab3.2-st.ipynb code cell 15

# PART 1: DQN on VacuumCleanerEnv
### 1.2: Training DQN on VacuumCleanerEnv

print("\n" + "="*60)
print("PART 1: DQN ON VACUUMCLEANERENV")
print("="*60)

print("\n1.2: Training on Custom VacuumCleanerEnv")
print("-" * 60)

env1 = VacuumCleanerEnv(grid_size=5)
agent1 = DQNAgent(state_size=50, action_size=4, learning_rate=1e-3)

num_episodes = 250
batch_size = 32

episode_rewards = []
episode_lengths = []

print(f"\nTraining for {num_episodes} episodes...\n")

for episode in range(num_episodes):
    state, _ = env1.reset()
    episode_reward = 0
    step_count = 0
    done = False

    while not done:
        # Instruction:
        # 1) Select action.
        # 2) Step environment.
        # 3) Store transition and train.
        # 4) Update state and counters.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        action = agent1.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env1.step(action)
        done = terminated or truncated
        agent1.store_transition(state, action, reward, next_state, done)
        agent1.train_step(batch_size)
        episode_reward += reward
        step_count += 1
        state = next_state

    agent1.decay_epsilon()
    episode_rewards.append(episode_reward)
    episode_lengths.append(step_count)

    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode+1:3d}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f} | Epsilon: {agent1.epsilon:.3f}")

print(f"\nTraining completed!")
print(f"Final average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.2f}")
