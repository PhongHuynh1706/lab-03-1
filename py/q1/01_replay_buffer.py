# Source: notebooks/Lab3.1-st.ipynb code cell 17

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1

### 2.1: Core Component - Replay Buffer
print("\n" + "="*60)
print("PART 2: DQN ON FROZENLAKE-V1")
print("="*60)

class ReplayBuffer:
    """Store and sample experiences for training DQN.

    Experience replay allows the agent to learn from past experiences,
    decorrelating data and improving learning stability.
    """

    def __init__(self, capacity=10000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """

        # Sample random transitions from the buffer
        # Instruction:
        # 1) Use random.sample to get a batch of transitions from self.buffer
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        batch = random.sample(self.buffer, batch_size)
        
        # 2) Convert lists to numpy arrays first to avoid slow tensor creation warnings
        states_np = np.array([t[0] for t in batch], dtype=np.float32)
        actions_np = np.array([t[1] for t in batch], dtype=np.int64)
        rewards_np = np.array([t[2] for t in batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in batch], dtype=np.float32)
        dones_np = np.array([t[4] for t in batch], dtype=np.float32)

        # 3) Convert numpy arrays to PyTorch tensors and move to device
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        states = torch.from_numpy(states_np)
        actions = torch.from_numpy(actions_np)
        rewards = torch.from_numpy(rewards_np)
        next_states = torch.from_numpy(next_states_np)
        dones = torch.from_numpy(dones_np)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)

print("✓ ReplayBuffer class defined")