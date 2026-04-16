# Source: notebooks/Lab3.2-st.ipynb code cell 32

# PART 3: DQN with Stable-Baselines3
### 3.1: Import Stable-Baselines3

print("\n" + "="*60)
print("PART 3: STABLE-BASELINES3 DQN ON FROZENLAKE-V1")
print("="*60)

print("\n3.1: Import Stable-Baselines3")
print("-" * 60)

try:
    from stable_baselines3 import DQN
    sb3_available = True
    print("✓ Stable-Baselines3 imported successfully")
except Exception as e:
    sb3_available = False
    print("⚠ SB3 import failed in this environment")
    print(f"  Error: {type(e).__name__}: {e}")
    print("  Please check package compatibility in env nt549-1-phatpt.")
