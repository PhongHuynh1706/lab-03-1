# Source: notebooks/Lab3.1-st.ipynb code cell 21

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1
### 2.3: DQN Agent - Complete Algorithm

class DQNAgent:
    """Deep Q-Learning Agent for training on environments.

    Key components:
    - Q-network: estimates Q-values
    - Target network: stabilizes training
    - Replay buffer: stores transitions
    - Epsilon-greedy: exploration vs exploitation
    """

    def __init__(self, state_size=16, action_size=4, learning_rate=1e-3):
        """Initialize DQN agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
        """

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Instruction:
        # 1) Create q_network and target_network as instances of DQNNetwork.
        # 2) Move networks to device (CPU or GPU).
        # 3) Initialize target_network weights to match q_network and set to eval mode.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        self.q_network = None
        self.target_network = None
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.update_counter = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Instruction:
        # Build one-hot state tensor and select greedy action from q_network.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        state_onehot = None
        with torch.no_grad():
            q_values = None
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        # Instruction:
        # Convert both state and next_state to one-hot vectors,
        # then push transition to replay buffer.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        state_onehot = np.zeros(self.state_size, dtype=np.float32)
        next_state_onehot = np.zeros(self.state_size, dtype=np.float32)
        self.replay_buffer.push(state_onehot, action, reward, next_state_onehot, done)

    def train_step(self, batch_size=32):
        """Update Q-network with one mini-batch from replay buffer.

        DQN Update Rule:
            Q_target(s,a) = r + gamma * max_a' Q_target(s',a')
            Loss = (Q_target(s,a) - Q_network(s,a))^2

        Args:
            batch_size: Size of mini-batch for training
        """

        if len(self.replay_buffer) < batch_size:
            return  # Not enough samples to train

        # Sample a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Instruction:
        # 1) Compute q_values for taken actions.

        ### YOU NEED TO WRITE YOUR CODE BELOW ###

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2) Compute target_q_values using target network.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = None

        # 3) Compute loss, backward, optimizer.step().
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        loss = None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4) Update target network every 100 steps.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            # HERE
            pass

    def decay_epsilon(self):
        # Instruction:
        # Decay epsilon after each episode, ensuring it does not go below epsilon_min.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        self.epsilon = None


print("✓ DQNAgent class defined")
