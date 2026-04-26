# Source: notebooks/Lab3.1-st.ipynb code cell 23

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1
### 2.4: Training DQN on FrozenLake

# Create environment
env_train = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
agent = DQNAgent(state_size=16, action_size=4, learning_rate=1e-3)

# Training parameters
num_episodes = 500
batch_size = 32

episode_rewards = []
episode_lengths = []

print(f"\n2.4: Training DQN on FrozenLake")
print("-" * 60)
print(f"\nTraining for {num_episodes} episodes...\n")

for episode in range(num_episodes):
    state, _ = env_train.reset()

    episode_reward = 0
    step_count = 0
    done = False

    while not done:
        # Instruction:
        # 1) Select action with epsilon-greedy policy.
        # 2) Step environment.
        # 3) Store transition and train one step.
        # 4) Update state, reward, counters.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env_train.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step(batch_size)

        episode_reward += reward
        step_count += 1
        state = next_state

    agent.decay_epsilon()
    episode_rewards.append(episode_reward)
    episode_lengths.append(step_count)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1:3d}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

print(f"\nTraining completed!")
print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
