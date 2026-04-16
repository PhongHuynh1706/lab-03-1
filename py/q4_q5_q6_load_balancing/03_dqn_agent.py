# Source: notebooks/Lab3.2-st.ipynb code cell 10

# Shared Components for DQN
### Shared Component 3: DQN Agent


class DQNAgent:
    """Reusable Deep Q-Learning agent for discrete-action environments."""

    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99):
        """Initialize model, optimizer, and replay memory."""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

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

        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        state_array = None
        state_tensor = None

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        self.replay_buffer.push(state_array, action, reward, next_state_array, done)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        q_values = None

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = None

        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        loss = None
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % 100 == 0:
            # HERE
            pass

    def decay_epsilon(self):
        self.epsilon = None


print("✓ DQNAgent class defined")
