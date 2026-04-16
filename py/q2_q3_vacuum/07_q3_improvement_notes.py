# Source: notebooks/Lab3.2-st.ipynb markdown cell 21

# PART 1: DQN on VacuumCleanerEnv
### Part 1 Improvement Challenge (Target: Success Rate > 80%)

# Part 1 Improvement Challenge (Target: Success Rate > 80%)
#
# Use your DQN implementation and try to improve performance with the following ideas:
#
# 1. Reward redesign
#    - Add stronger penalty for wall hits / repeated visits to old cells.
#    - Add larger bonus when the agent cleans all cells.
#
# 2. Q-network architecture
#    - Try hidden sizes: 64 / 128 / 256.
#    - Try deeper MLP (more hidden layers).
#
# 3. Hyperparameter tuning
#    - learning_rate
#    - batch_size = 32 / 64
#    - epsilon_decay = 0.99 / 0.995 / 0.998
#
# Document the best setting that helps you reach success rate > 80%.
#
# This file is intentionally kept light in the first split because the current
# notebook only provides the challenge prompt, not separate code cells yet.
# Use `q2_vacuum_baseline.py` as the closest copy source when building the real
# tuning runs.
