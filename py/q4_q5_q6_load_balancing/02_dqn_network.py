# Source: notebooks/Lab3.2-st.ipynb code cell 8

# Shared Components for DQN
### Shared Component 2: DQN Network


class DQNNetwork(nn.Module):
    """Simple MLP-based Deep Q-Network."""

    def __init__(self, state_size, action_size, hidden_size=128):
        """Initialize network layers.

        Args:
            state_size: Number of features in state vector.
            action_size: Number of discrete actions.
            hidden_size: Width of hidden layer.
        """
        super(DQNNetwork, self).__init__()
        # Instruction:
        # 1) Define a linear layer (self.fc1) that maps state_size to
        #    a hidden layer of size 64
        # 2) Define a ReLU activation (self.relu)
        # 3) Define a linear layer (self.fc2) that maps the hidden layer
        #    to action_size (output Q-values for each action)
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        self.fc1 = None
        self.relu = None
        self.fc2 = None

    def forward(self, state):
        """Compute Q-values for each possible action given a state batch."""
        x = self.fc1(state)
        x = self.relu(x)
        q_values = self.fc2(x)
        return q_values


print("✓ DQNNetwork class defined")
