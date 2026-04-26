# Source: notebooks/Lab3.1-st.ipynb code cell 23

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1
### 2.4: Training DQN on FrozenLake

# Create environment
learning_rates = [1e-2, 1e-3, 5e-4]
seeds = [40, 41, 42, 43, 44]

num_episodes = 500
batch_size = 32

results = {}

for lr in learning_rates:
    results[lr] = {"reward": [], "loss": []}

    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env_train = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

        agent = DQNAgent(state_size=16, action_size=4, learning_rate=lr)

        episode_rewards = []

        for episode in range(num_episodes):

            state, _ = env_train.reset()
            done = False
            ep_reward = 0

            while not done:

                action = agent.select_action(state, training=True)

                next_state, reward, terminated, truncated, _ = env_train.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step(batch_size)

                state = next_state
                ep_reward += reward

            agent.decay_epsilon()
            episode_rewards.append(ep_reward)

        results[lr]["reward"].append(episode_rewards)
        results[lr]["loss"].append(agent.loss_history)