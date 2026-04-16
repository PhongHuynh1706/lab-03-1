# Local runner: execute the full Q4-Q6 flow in one Python process.

from pathlib import Path


def run_script(script_path: Path, shared_globals: dict) -> None:
    shared_globals["__file__"] = str(script_path)
    code = compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
    exec(code, shared_globals)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    scripts = [
        base_dir / "00_setup.py",
        base_dir / "01_replay_buffer.py",
        base_dir / "02_dqn_network.py",
        base_dir / "03_dqn_agent.py",
        base_dir / "04_load_balancing_env.py",
        base_dir / "05_q4_train_eval_plot.py",
        base_dir / "06_q5_metrics_csv.py",
        base_dir / "07_q6_reward_redesign.py",
    ]

    shared_globals = {"__name__": "__main__"}

    for script in scripts:
        print(f"[main] Running {script.name}")
        run_script(script, shared_globals)


if __name__ == "__main__":
    main()
