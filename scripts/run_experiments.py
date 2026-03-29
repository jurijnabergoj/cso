"""
Submit a round of experiments to SLURM.
Experiment definitions are defined in cfg/experiments.py.
Change ACTIVE there to switch rounds then run this script.

Usage: python scripts/run_experiments.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfg.utils import generate_config
from cfg.experiments import GROUP, EXPERIMENTS

PROJECT_ROOT = Path("/d/hpc/home/jn16867/cso")
SLURM_SCRIPT = PROJECT_ROOT / "slurm" / "run_training.slurm"


def submit_job(name: str, config_path: Path) -> str:
    result = subprocess.run(
        [
            "sbatch",
            f"--export=ALL,EXP_NAME={name},CONFIG_PATH={config_path}",
            str(SLURM_SCRIPT),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {name}:\n{result.stderr}")
    job_id = result.stdout.strip().split()[-1]
    return job_id


if __name__ == "__main__":
    print(f"Submitting {len(EXPERIMENTS)} experiments (group: {GROUP})...\n")

    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"[{name}]")
        config_path = generate_config(exp)
        job_id = submit_job(name, config_path)
        print(f"  Submitted job {job_id}\n")
