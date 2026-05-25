from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ohlcv_current.csv"
MODEL = ROOT / "models" / "momentum_nn_current.pt"
CANDIDATES = ROOT / "outputs" / "candidates_current.csv"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def is_fresh() -> bool:
    if not CANDIDATES.exists():
        return False
    candidate_time = CANDIDATES.stat().st_mtime
    inputs = [path for path in [CACHE, MODEL] if path.exists()]
    return bool(inputs) and all(candidate_time >= path.stat().st_mtime for path in inputs)


def run_screen() -> None:
    command = [
        str(PYTHON),
        "-m",
        "momentum_screener.cli",
        "screen",
        "--cache-only",
        "--cache",
        str(CACHE),
        "--model-path",
        str(MODEL),
        "--output",
        str(CANDIDATES),
        "--gate-min-turnover-5d",
        "50000000",
        "--gate-min-ret-5d",
        "-0.01",
        "--gate-min-turnover-ratio-1d-20d",
        "1.05",
        "--gate-min-turnover-ratio-5d-20d",
        "1.05",
        "--gate-min-close-ma25-ratio",
        "-0.01",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def update_data() -> None:
    command = [
        str(PYTHON),
        "-m",
        "momentum_screener.cli",
        "refresh-data",
        "--ticker-csv",
        str(ROOT / "config" / "listed_stocks.csv"),
        "--ticker-csv-code-column",
        "code",
        "--no-sample-tickers",
        "--cache",
        str(CACHE),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    update_requested = "--update-data" in sys.argv
    force_requested = "--force" in sys.argv

    if is_fresh() and not force_requested:
        print("Existing latest candidates CSV is up to date. Skipping inference.", flush=True)
    else:
        if update_requested:
            print("Updating OHLCV data without retraining the model.", flush=True)
            update_data()
        print("Candidates CSV is missing or stale. Running inference.", flush=True)
        run_screen()

    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "show_candidates.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
