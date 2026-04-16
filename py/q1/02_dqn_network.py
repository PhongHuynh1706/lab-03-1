# Source: notebooks/Lab3.1-st.ipynb code cell 19

# PART 2: Deep Q-Network (DQN) on FrozenLake-v1
### 2.2: Core Component - DQN Network

class DQNNetwork(nn.Module):
    """Deep Q-Network for approximating Q-values.

    Architecture for FrozenLake:
        Input (16, one-hot encoded) -> Linear(64) -> ReLU -> Linear(4) -> Output

    The output represents Q-values for each of 4 actions.
    """

    def __init__(self, state_size=16, action_size=4):
        """Initialize DQN network.

        Args:
            state_size: Size of state input (16 for FrozenLake 4x4)
            action_size: Number of possible actions (4 for FrozenLake)
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
        """Forward pass: compute Q-values for all actions.

        Args:
            state: One-hot encoded state or batch of states

        Returns:
            Q-values for each action
        """
        x = self.fc1(state)      # Pass through hidden layer
        x = self.relu(x)         # Apply ReLU activation
        q_values = self.fc2(x)   # Compute Q-values
        return q_values


print("✓ DQNNetwork class defined")
