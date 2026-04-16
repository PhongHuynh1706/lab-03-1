# Local support imports for this extracted multi-cell file.
# Notebook-derived blocks below keep per-cell source markers for easier copy-back.

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Gymnasium Version: {gym.__version__}")
print(f"Torch Threads: {torch.get_num_threads()}")
print(f"{'='*60}\n")

# Local support block reused so the extracted MountainCar demo can stay readable.

class ReplayBuffer:
    """Store and sample experiences for training DQN."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        batch = None

        states_np = np.array([t[0] for t in batch], dtype=np.float32)
        actions_np = np.array([t[1] for t in batch], dtype=np.int64)
        rewards_np = np.array([t[2] for t in batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in batch], dtype=np.float32)
        dones_np = np.array([t[4] for t in batch], dtype=np.float32)

        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        states = None
        actions = None
        rewards = None
        next_states = None
        dones = None

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Source: notebooks/Lab3.1-st.ipynb code cell 30

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.1: Environment Setup

print("\n" + "="*60)
print("PART 3: DQN ON MOUNTAINCAR-V0")
print("="*60)

print("\n3.1: Environment Setup")
print("-" * 60)

env_mc = gym.make('MountainCar-v0')
state, _ = env_mc.reset()

print(f"\nMountainCar Environment:")
print(f"  State space: {env_mc.observation_space} (position, velocity)")
print(f"  Action space: {env_mc.action_space} (3 discrete actions)")
print(f"  Sample state: {state}")
print(f"  Reward: +1 at goal, -1 per step")
print(f"  Max steps: 200")

# Source: notebooks/Lab3.1-st.ipynb code cell 32

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.2: Define DQN Network for Continuous State

class DQNNetworkContinuous(nn.Module):
    """DQN network for continuous state spaces."""

    def __init__(self, hidden_sizes=(64, 64)):
        super(DQNNetworkContinuous, self).__init__()
        layers = []
        prev_size = 2

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


print("\n3.2: DQN Network Setup")
print("-" * 60)
print("✓ DQNNetworkContinuous class defined")

# Source: notebooks/Lab3.1-st.ipynb code cell 34

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.3: DQN Agent for MountainCar

class DQNAgentMountainCar:
    """DQN Agent adapted for MountainCar environment."""

    def __init__(self, hidden_sizes=(64, 64), learning_rate=1e-3, name="Agent"):
        self.action_size = 3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.name = name

        self.q_network = DQNNetworkContinuous(hidden_sizes).to(device)
        self.target_network = DQNNetworkContinuous(hidden_sizes).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.update_counter = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        )

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


print("✓ DQNAgentMountainCar class defined")

# Source: notebooks/Lab3.1-st.ipynb code cell 36

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.4: Train Simple Network (64, 64)

print("\n3.4: Experiment 1 - Simple Network (64, 64)")
print("-" * 60)

agent_simple = DQNAgentMountainCar(hidden_sizes=(64, 64), learning_rate=1e-3, name="SimpleNet")

print(f"\nTraining simple network for 300 episodes...\n")

simple_rewards = []
simple_lengths = []

for episode in range(300):
    state, _ = env_mc.reset()
    episode_reward = 0
    step_count = 0
    done = False

    while not done:
        action = agent_simple.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env_mc.step(action)
        done = terminated or truncated

        agent_simple.store_transition(state, action, reward, next_state, done)
        agent_simple.train_step(batch_size=32)

        episode_reward += reward
        step_count += 1
        state = next_state

    agent_simple.decay_epsilon()
    simple_rewards.append(episode_reward)
    simple_lengths.append(step_count)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(simple_rewards[-100:])
        success = sum(1 for r in simple_rewards[-100:] if r > -200)
        print(f"Episode {episode+1:3d}/300 | Avg Reward: {avg_reward:.2f} | Success (>-200): {success}/100")

print(f"\nSimple Network - Final Results:")
print(f"  Avg Reward (last 50 eps): {np.mean(simple_rewards[-50:]):.2f}")
print(f"  Success Rate (>-200): {sum(1 for r in simple_rewards[-100:] if r > -200) / 100 * 100:.1f}%")

# Source: notebooks/Lab3.1-st.ipynb code cell 38

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.5: Train Improved Network (128, 128, 64)

print("\n3.5: Experiment 2 - Improved Network (128, 128, 64)")
print("-" * 60)

agent_improved = DQNAgentMountainCar(hidden_sizes=(128, 128, 64), learning_rate=5e-4, name="ImprovedNet")

print(f"\nTraining improved network for 300 episodes...\n")

improved_rewards = []
improved_lengths = []

for episode in range(300):
    state, _ = env_mc.reset()
    episode_reward = 0
    step_count = 0
    done = False

    while not done:
        action = agent_improved.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env_mc.step(action)
        done = terminated or truncated

        agent_improved.store_transition(state, action, reward, next_state, done)
        agent_improved.train_step(batch_size=32)

        episode_reward += reward
        step_count += 1
        state = next_state

    agent_improved.decay_epsilon()
    improved_rewards.append(episode_reward)
    improved_lengths.append(step_count)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(improved_rewards[-100:])
        success = sum(1 for r in improved_rewards[-100:] if r > -200)
        print(f"Episode {episode+1:3d}/300 | Avg Reward: {avg_reward:.2f} | Success (>-200): {success}/100")

print(f"\nImproved Network - Final Results:")
print(f"  Avg Reward (last 50 eps): {np.mean(improved_rewards[-50:]):.2f}")
print(f"  Success Rate (>-200): {sum(1 for r in improved_rewards[-100:] if r > -200) / 100 * 100:.1f}%")

# Source: notebooks/Lab3.1-st.ipynb code cell 40

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.6: Evaluation and Comparison

print("\n3.6: Evaluation and Comparison")
print("-" * 60)

eval_episodes = 20

simple_eval_rewards = []
for _ in range(eval_episodes):
    state, _ = env_mc.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent_simple.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env_mc.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state
    simple_eval_rewards.append(episode_reward)

improved_eval_rewards = []
for _ in range(eval_episodes):
    state, _ = env_mc.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent_improved.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env_mc.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state
    improved_eval_rewards.append(episode_reward)

print(f"\nEvaluation Results ({eval_episodes} episodes, greedy policy):\n")
print(f"Simple Network (64, 64):")
print(f"  Avg Reward: {np.mean(simple_eval_rewards):.2f}")
print(f"  Success Rate: {sum(1 for r in simple_eval_rewards if r > -200) / eval_episodes * 100:.1f}%")

print(f"\nImproved Network (128, 128, 64):")
print(f"  Avg Reward: {np.mean(improved_eval_rewards):.2f}")
print(f"  Success Rate: {sum(1 for r in improved_eval_rewards if r > -200) / eval_episodes * 100:.1f}%")

# Source: notebooks/Lab3.1-st.ipynb code cell 42

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.7: Visualization - Training Curves

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(simple_rewards, alpha=0.3, label='Simple (64,64)', color='blue')
ma_simple = np.convolve(simple_rewards, np.ones(50)/50, mode='valid')
axes[0, 0].plot(range(49, len(simple_rewards)), ma_simple, color='blue', linewidth=2, label='MA(50)')

axes[0, 0].plot(improved_rewards, alpha=0.3, label='Improved (128,128,64)', color='green')
ma_improved = np.convolve(improved_rewards, np.ones(50)/50, mode='valid')
axes[0, 0].plot(range(49, len(improved_rewards)), ma_improved, color='green', linewidth=2)

axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Episode Reward')
axes[0, 0].set_title('Training: Simple vs Improved Networks', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=-200, color='red', linestyle='--', alpha=0.5)

axes[0, 1].boxplot([simple_eval_rewards, improved_eval_rewards], labels=['Simple\n(64,64)', 'Improved\n(128,128,64)'])
axes[0, 1].axhline(y=-200, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_ylabel('Episode Reward')
axes[0, 1].set_title('Evaluation: Reward Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

window = 50
simple_success = np.convolve([1 if r > -200 else 0 for r in simple_rewards], np.ones(window)/window, mode='valid')
improved_success = np.convolve([1 if r > -200 else 0 for r in improved_rewards], np.ones(window)/window, mode='valid')

axes[1, 0].plot(range(window-1, len(simple_rewards)), simple_success, label='Simple (64,64)', color='blue', linewidth=2)
axes[1, 0].plot(range(window-1, len(improved_rewards)), improved_success, label='Improved (128,128,64)', color='green', linewidth=2)
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Success Rate')
axes[1, 0].set_title(f'Success Rate (MA {window}): Threshold = -200', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

axes[1, 1].plot(simple_lengths, alpha=0.3, label='Simple (64,64)', color='blue')
ma_simple_len = np.convolve(simple_lengths, np.ones(50)/50, mode='valid')
axes[1, 1].plot(range(49, len(simple_lengths)), ma_simple_len, color='blue', linewidth=2)

axes[1, 1].plot(improved_lengths, alpha=0.3, label='Improved (128,128,64)', color='green')
ma_improved_len = np.convolve(improved_lengths, np.ones(50)/50, mode='valid')
axes[1, 1].plot(range(49, len(improved_lengths)), ma_improved_len, color='green', linewidth=2)

axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Episode Length (steps)')
axes[1, 1].set_title('Episode Length: Simple vs Improved', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close(fig)

# Source: notebooks/Lab3.1-st.ipynb code cell 44

# PART 3: DQN on MountainCar-v0 - Demonstrating DQN Limitations
### 3.8: Analysis and Conclusions

print("\n3.8: Conclusion and Analysis")
print("="*60)

print("\n🔍 KEY OBSERVATIONS:")
print("-" * 60)

print(f"\n1. Success Rates (Target: > -200 reward):")
print(f"   - Simple Network:   {sum(1 for r in simple_eval_rewards if r > -200) / eval_episodes * 100:.1f}%")
print(f"   - Improved Network: {sum(1 for r in improved_eval_rewards if r > -200) / eval_episodes * 100:.1f}%")

print(f"\n2. Average Rewards:")
print(f"   - Simple Network:   {np.mean(simple_eval_rewards):.2f}")
print(f"   - Improved Network: {np.mean(improved_eval_rewards):.2f}")

print(f"\n3. Reward Variance:")
print(f"   - Simple Network:   {np.std(simple_eval_rewards):.2f}")
print(f"   - Improved Network: {np.std(improved_eval_rewards):.2f}")

print("\n" + "="*60)
print("📌 CONCLUSIONS:")
print("="*60)

print("""
1. DQN Struggles with MountainCar:
   - Both simple and improved networks show VERY LOW success rates
   - Reason: Sparse reward signal (only +1 at goal, -1 per step)
   - The environment gives little guidance to the agent

2. Network Architecture Helps But Not Enough:
   - Improved network slightly outperforms simple network
   - But both remain ineffective for this problem
   - Demonstrates that deeper networks alone don't solve all problems

3. Why DQN Fails Here:
   - Continuous state space: DQN works better with discrete states
   - Sparse rewards: DQN needs frequent reward signals
   - High-dimensional learning: 2D state with credit assignment difficulty

4. Better Approaches for MountainCar:
   - Policy Gradient methods (e.g., Actor-Critic, PPO)
   - Reward shaping: Add intermediate rewards to guide learning
   - Intrinsic motivation: Use curiosity or empowerment
   - Dueling DQN or other DQN improvements

5. Takeaway:
   - Choose algorithms based on environment characteristics
   - DQN is excellent for: Discrete state/action, dense rewards (games)
   - Avoid DQN for: Continuous states, sparse rewards, complex dynamics
""")

env_mc.close()

print("="*60)
print("END OF LAB 3.1")
print("="*60)
