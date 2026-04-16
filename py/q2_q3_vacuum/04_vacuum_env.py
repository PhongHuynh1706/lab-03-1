# Source: notebooks/Lab3.2-st.ipynb code cell 13

# PART 1: DQN on VacuumCleanerEnv
### 1.1: Define VacuumCleanerEnv

# Problem setup:
#+ Grid world 5x5. Agent starts at cell 0.
#+ Every other cell is initially dirty.
#+ Goal: move and clean as many cells as possible before max_steps.


class VacuumCleanerEnv(gym.Env):
    """Grid-based vacuum cleaning task with discrete movement actions."""

    def __init__(self, grid_size=5):
        """Initialize environment dynamics and spaces."""
        super().__init__()
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation = [one-hot position (25)] + [dirty-mask (25)] => length 50
        self.observation_space = spaces.Box(low=0, high=1, shape=(50,), dtype=np.float32)

        self.agent_pos = 0
        self.dirty_cells = set(range(1, self.num_cells))
        self.steps = 0
        self.max_steps = 100
        self.total_reward = 0.0

    def _get_state(self):
        """Encode current environment state as a float32 vector."""
        state = np.zeros(50, dtype=np.float32)

        # Position channel (one-hot)
        state[self.agent_pos] = 1.0

        # Dirt channel (binary mask)
        for cell in self.dirty_cells:
            state[25 + cell] = 1.0

        return state

    def reset(self, seed=None):
        """Reset to initial configuration for a new episode."""
        super().reset(seed=seed)
        self.agent_pos = 0
        self.dirty_cells = set(range(1, self.num_cells))
        self.steps = 0
        self.total_reward = 0.0
        return self._get_state(), {}

    def step(self, action):
        """Apply one action and return Gymnasium transition tuple."""
        self.steps += 1
        row, col = divmod(self.agent_pos, self.grid_size)

        # Apply bounded movement
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.grid_size - 1, col + 1)

        self.agent_pos = row * self.grid_size + col

        # Reward design: +10 for cleaning new dirty cell, -0.1 otherwise
        if self.agent_pos in self.dirty_cells:
            self.dirty_cells.remove(self.agent_pos)
            reward = 10.0
        else:
            reward = -0.1

        self.total_reward += reward

        terminated = len(self.dirty_cells) == 0
        truncated = self.steps >= self.max_steps

        return self._get_state(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Render environment as text grid.

        mode='human': print to notebook output
        mode='ansi' : return string for custom logging/animation
        """
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark dirty cells
        for cell in self.dirty_cells:
            r, c = divmod(cell, self.grid_size)
            grid[r][c] = 'D'

        # Mark agent
        r, c = divmod(self.agent_pos, self.grid_size)
        grid[r][c] = 'A'

        lines = [' '.join(row) for row in grid]
        cleaned = (self.num_cells - 1) - len(self.dirty_cells)
        clean_ratio = cleaned / (self.num_cells - 1)
        lines.append(f"Steps: {self.steps}/{self.max_steps} | Cleaned: {cleaned}/{self.num_cells - 1} ({clean_ratio:.1%}) | Total reward: {self.total_reward:.2f}")
        text = '\n'.join(lines)

        if mode == 'ansi':
            return text
        print(text)
        print()


print("✓ VacuumCleanerEnv class defined")
