# Source: notebooks/Lab3.1-st.ipynb code cell 3

### Imports and Setup

# Fix for OpenMP conflict on macOS (numpy + torch + gymnasium)
# This prevents: "OMP: Error #15: Initializing libiomp5.dylib already initialized"
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Limit PyTorch threading for kernel stability on macOS
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Gymnasium Version: {gym.__version__}")
print(f"Torch Threads: {torch.get_num_threads()}")
print(f"{'='*60}\n")

# Source: notebooks/Lab3.1-st.ipynb code cell 6

# PART 1: Introduction to PyTorch
### 1.1: Tensor Creation and Basic Operations

print("\n" + "="*60)
print("PART 1: INTRODUCTION TO PYTORCH")
print("="*60)

# Create tensors with gradient tracking
# Instruction:
# 1) Create tensor x with values [1.0, 2.0, 3.0], requires_grad=True
# 2) Create tensor y with values [4.0, 5.0, 6.0], requires_grad=True
### YOU NEED TO WRITE YOUR CODE BELOW ###
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

print(f"\nTensor x: {x}")
print(f"Tensor y: {y}")
print(f"x shape: {x.shape}")
print(f"x device: {x.device}")

# Basic operations
# Instruction:
# - Compute z = x + y
# - Compute w = x * y
### YOU NEED TO WRITE YOUR CODE BELOW ###
z = x + y
w = x * y

print(f"\nx + y = {z}")
print(f"x * y = {w}")

# Source: notebooks/Lab3.1-st.ipynb code cell 8

# PART 1: Introduction to PyTorch
### 1.2: Automatic Differentiation (Autograd)

print("\n1.2: Automatic Differentiation with Autograd")
print("-" * 60)

# Define a computation graph
### YOU NEED TO WRITE YOUR CODE BELOW ###
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"\nInput x: {x}")
print(f"requires_grad: {x.requires_grad}")

# Forward pass
# y = x^2, z = sum(y)
### YOU NEED TO WRITE YOUR CODE BELOW ###
y = x * x
z = sum(y)

print(f"y = x^2: {y}")
print(f"z = sum(y): {z}")

# Backward pass (compute gradients)
### YOU NEED TO WRITE YOUR CODE BELOW ###

z.backward()

print(f"\nGradients (dz/dx): {x.grad}")
print(f"Expected: [4.0, 6.0] (since d/dx[x^2] = 2x)")

# Source: notebooks/Lab3.1-st.ipynb code cell 10

# PART 1: Introduction to PyTorch
### 1.3: Building a Simple Neural Network

print("\n1.3: Building a Simple Neural Network")
print("-" * 60)

class SimpleNet(nn.Module):
    """A simple 2-layer neural network for demonstration.
    
    Architecture:
        Input (4) -> Linear -> Hidden (64) -> ReLU -> Linear -> Output (2)
    """

    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        """Initialize network layers.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden layer
            output_size: Dimension of output
        """
        super(SimpleNet, self).__init__()
        
        # Instruction:
        # 1) Define a linear layer (self.fc1) that maps input_size to hidden_size
        # 2) Define a ReLU activation (self.relu)
        # 3) Define a linear layer (self.fc2) that maps hidden_size to output_size
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Instruction:
        # 1) Pass x through self.fc1, then apply self.relu
        # 2) Pass the result through self.fc2 to get the output
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        x = self.fc1(x)   # Linear
        x = self.relu(x)  # Activation
        x = self.fc2(x)   # Output layer
                
        return x

# Create network instance
net = SimpleNet(input_size=4, hidden_size=64, output_size=2)
print(f"\nNetwork Architecture:")
print(net)

# Source: notebooks/Lab3.1-st.ipynb code cell 11

# PART 1: Introduction to PyTorch
### 1.3: Building a Simple Neural Network

# Count parameters
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\nTotal Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")

# Forward pass with dummy input
dummy_input = torch.randn(1, 4)  # Batch size 1, 4 features
output = net(dummy_input)
print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output values: {output}")

# Source: notebooks/Lab3.1-st.ipynb code cell 13

# PART 1: Introduction to PyTorch
### 1.4: Training Loop with Optimizer

print("\n1.4: Training Loop with Optimizer")
print("-" * 60)

# Create network and optimizer
# Instruction:
# 1) Create an instance of SimpleNet
# 2) Use Adam optimizer with learning rate 0.01
# 3) Use Mean Squared Error loss function
net = SimpleNet(input_size=4, hidden_size=64, output_size=2)
### YOU NEED TO WRITE YOUR CODE BELOW ###
optimizer = optim.Adam(net.parameters(), lr=0.01)  # Adam optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error loss

# Create dummy training data
X_train = torch.randn(10, 4)   # 10 samples, 4 features
y_train = torch.randn(10, 2)   # 10 targets, 2 output dimensions

print(f"\nTraining on {len(X_train)} samples for 5 iterations...")

# Source: notebooks/Lab3.1-st.ipynb code cell 14

# PART 1: Introduction to PyTorch
### 1.4: Training Loop with Optimizer

# Training loop
losses = []

for epoch in range(5):
    # 1. Forward pass: compute predictions
    predictions = net(X_train)

    # 2. Compute loss
    loss = loss_fn(predictions, y_train)

    # 3. Backward pass: compute gradients
    optimizer.zero_grad()  # reset gradients
    loss.backward()        # compute gradients

    # 4. Update weights
    optimizer.step()       # update parameters

    losses.append(loss.item())
    print(f"Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

print(f"\nTraining completed! Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
print("\n" + "="*60 + "\n")
