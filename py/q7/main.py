# Local runner: execute the full Q7 flow in one Python process.

from pathlib import Path


def run_script(script_path: Path, shared_globals: dict) -> None:
    shared_globals["__file__"] = str(script_path)
    code = compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
    exec(code, shared_globals)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    scripts = [
        base_dir / "00_setup.py",
        base_dir.parent / "q2_q3_vacuum" / "04_vacuum_env.py",
        base_dir / "02_sb3_setup.py",
        base_dir / "03_q7_train.py",
        base_dir / "04_q7_eval.py",
    ]

    shared_globals = {"__name__": "__main__"}

    for script in scripts:
        print(f"[main] Running {script.name}")
        run_script(script, shared_globals)


if __name__ == "__main__":
    main()
