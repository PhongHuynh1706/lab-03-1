# Source: notebooks/Lab3.2-st.ipynb code cell 24

# PART 2: DQN on LoadBalancingEnv
### 2.1: Define LoadBalancingEnv


class Task:
    """Simple task object with processing demand."""

    def __init__(self, demand=10):
        self.demand = demand


class Server:
    """Server with queue and fixed processing speed per step."""

    def __init__(self, speed=1.0):
        self.speed = speed
        self.queue = 0

    def add_task(self, demand):
        self.queue += demand

    def process(self):
        self.queue = max(0, self.queue - self.speed)

    def get_utilization(self):
        return min(1.0, self.queue / 100.0)


class LoadBalancingEnv(gym.Env):
    """Gymnasium environment for dynamic multi-server load balancing."""

    def __init__(self, num_servers=3):
        super().__init__()
        self.num_servers = num_servers
        self.servers = [Server(speed=1.0) for _ in range(num_servers)]
        self.action_space = spaces.Discrete(num_servers)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_servers,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 300

    def _get_state(self):
        return np.array([s.get_utilization() for s in self.servers], dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.servers = [Server(speed=1.0) for _ in range(self.num_servers)]
        self.steps = 0
        return self._get_state(), {}

    def step(self, action):
        self.steps += 1
        task_demand = np.random.randint(5, 15)
        self.servers[action].add_task(task_demand)
        for server in self.servers:
            server.process()
        total_queue = sum(s.queue for s in self.servers)
        reward = -total_queue / 100.0
        terminated = False
        truncated = self.steps >= self.max_steps
        return self._get_state(), reward, terminated, truncated, {}


print("✓ LoadBalancingEnv class defined")
