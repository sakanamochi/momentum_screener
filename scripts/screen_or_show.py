from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from momentum_screener.settings import SCREEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ohlcv_current.csv"
MODEL = ROOT / "models" / "momentum_nn_production.pt"
CANDIDATES = ROOT / "outputs" / "candidates_current.csv"
RECENT_CANDIDATES = ROOT / "outputs" / "candidates_recent.csv"
RANK_HISTORY = ROOT / "outputs" / "candidates_rank_history.csv"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def is_fresh() -> bool:
    if not CANDIDATES.exists() or not RECENT_CANDIDATES.exists() or not RANK_HISTORY.exists():
        return False
    candidate_time = min(CANDIDATES.stat().st_mtime, RECENT_CANDIDATES.stat().st_mtime, RANK_HISTORY.stat().st_mtime)
    inputs = [path for path in [CACHE, MODEL] if path.exists()]
    return bool(inputs) and all(candidate_time >= path.stat().st_mtime for path in inputs)


def screen_command(output: Path, recent_days: int, rank_history_output: Path | None = None) -> list[str]:
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
        str(output),
        "--recent-days",
        str(recent_days),
        "--signal-count-days",
        str(SCREEN_SETTINGS["signal_count_days"]),
        "--signal-count-top-n",
        str(SCREEN_SETTINGS["signal_count_top_n"]),
        "--signal-count-min-score",
        str(SCREEN_SETTINGS["signal_count_min_score"]),
    ]
    if rank_history_output is not None:
        command.extend(["--rank-history-output", str(rank_history_output)])
    return command


def write_current_from_recent() -> None:
    import pandas as pd

    recent = pd.read_csv(RECENT_CANDIDATES, parse_dates=["date"], dtype={"code": str})
    latest_date = recent["date"].dropna().max()
    current = recent[recent["date"] == latest_date].copy()
    current.to_csv(CANDIDATES, index=False)


def run_screen() -> None:
    subprocess.run(screen_command(RECENT_CANDIDATES, recent_days=6, rank_history_output=RANK_HISTORY), cwd=ROOT, check=True)
    write_current_from_recent()


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

    if update_requested:
        print("Updating OHLCV data without retraining the model.", flush=True)
        update_data()
        force_requested = True

    if is_fresh() and not force_requested:
        print("Existing latest candidates CSV is up to date. Skipping inference.", flush=True)
    else:
        print("Candidates CSV is missing or stale. Running inference.", flush=True)
        run_screen()

    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "show_candidates.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
