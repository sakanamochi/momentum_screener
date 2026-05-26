from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from momentum_screener.cli import default_rolling_folds, parse_rolling_fold
from momentum_screener.data import load_or_download_ohlcv
from momentum_screener.features import (
    FEATURE_COLUMNS,
    add_features,
    add_initial_momentum_gate,
    add_labels,
    make_event_dataset,
)
from momentum_screener.model import save_artifacts, train_model
from momentum_screener.settings import GATE_SETTINGS, LABEL_SETTINGS


FEATURE_SETS: dict[str, list[str]] = {
    "all": FEATURE_COLUMNS,
    "simple_core": [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_5d_accel",
        "close_ma25_ratio",
        "close_ma75_ratio",
        "is_20d_high",
        "is_60d_high",
        "turnover_5d_avg",
        "turnover_ratio_1d_20d",
        "turnover_ratio_5d_20d",
        "log_turnover_5d_avg",
        "volatility_20d",
    ],
    "no_intraday_shape": [
        column
        for column in FEATURE_COLUMNS
        if column not in {"close_position_in_range", "intraday_range_ratio", "upper_shadow_ratio"}
    ],
    "no_share_turnover": [
        column for column in FEATURE_COLUMNS if column not in {"share_turnover_5d", "float_turnover_5d"}
    ],
    "no_long_trend": [
        column for column in FEATURE_COLUMNS if column not in {"ret_20d", "close_ma75_ratio", "is_60d_high"}
    ],
    "liquidity_momentum": [
        "ret_1d",
        "ret_5d",
        "ret_5d_accel",
        "close_ma25_ratio",
        "is_20d_high",
        "turnover_5d_avg",
        "turnover_ratio_1d_20d",
        "turnover_ratio_5d_20d",
        "log_turnover_5d_avg",
        "volatility_20d",
    ],
}


RATIO_PROFILES: dict[str, tuple[float, float]] = {
    "loose": (1.00, 1.00),
    "current": (1.05, 1.05),
    "one_day_strict": (1.15, 1.05),
    "five_day_strict": (1.05, 1.15),
    "strict": (1.15, 1.15),
}


BASE_CONFIG: dict[str, Any] = {
    "feature_set": "all",
    "gate_min_turnover_5d": GATE_SETTINGS["min_turnover_5d"],
    "gate_min_ret_5d": GATE_SETTINGS["min_ret_5d"],
    "gate_ratio_profile": "current",
    "gate_min_turnover_ratio_1d_20d": GATE_SETTINGS["min_turnover_ratio_1d_20d"],
    "gate_min_turnover_ratio_5d_20d": GATE_SETTINGS["min_turnover_ratio_5d_20d"],
    "gate_min_close_ma25_ratio": GATE_SETTINGS["min_close_ma25_ratio"],
    "cooldown_days": GATE_SETTINGS["cooldown_days"],
    "sample_weight_mode": "future_max_ret",
}


def with_ratio_profile(config: dict[str, Any], ratio_name: str) -> dict[str, Any]:
    ratio_1d, ratio_5d = RATIO_PROFILES[ratio_name]
    updated = dict(config)
    updated["gate_ratio_profile"] = ratio_name
    updated["gate_min_turnover_ratio_1d_20d"] = ratio_1d
    updated["gate_min_turnover_ratio_5d_20d"] = ratio_5d
    return updated


def coordinate_values(size: str) -> dict[str, list[Any]]:
    if size == "smoke":
        return {
            "feature_set": ["all", "simple_core"],
            "sample_weight_mode": ["future_max_ret", "target_future_max_ret"],
        }
    if size == "medium":
        return {
            "feature_set": list(FEATURE_SETS),
            "gate_min_turnover_5d": [30_000_000, 50_000_000, 100_000_000],
            "gate_min_ret_5d": [-0.02, -0.01, 0.0],
            "gate_ratio_profile": list(RATIO_PROFILES),
            "gate_min_close_ma25_ratio": [-0.03, -0.01, 0.0],
            "cooldown_days": [10, 20, 30],
            "sample_weight_mode": ["future_max_ret", "target_future_max_ret", "uniform"],
        }
    if size == "focused":
        return {
            "feature_set": ["all"],
            "gate_min_turnover_5d": [100_000_000, 200_000_000],
            "gate_min_ret_5d": [-0.02, -0.01, 0.0],
            "gate_ratio_profile": ["current", "one_day_strict"],
            "gate_min_close_ma25_ratio": [-0.03, -0.01, 0.0],
            "cooldown_days": [10, 20, 30],
            "sample_weight_mode": ["future_max_ret"],
        }
    if size == "focused_ma_cooldown":
        return {
            "gate_min_close_ma25_ratio": [-0.03, -0.01, 0.0],
            "cooldown_days": [10, 20, 30],
        }
    if size == "full":
        return {
            "feature_set": list(FEATURE_SETS),
            "gate_min_turnover_5d": [20_000_000, 30_000_000, 50_000_000, 100_000_000, 200_000_000],
            "gate_min_ret_5d": [-0.03, -0.02, -0.01, 0.0, 0.02],
            "gate_ratio_profile": list(RATIO_PROFILES),
            "gate_min_close_ma25_ratio": [-0.05, -0.03, -0.01, 0.0, 0.02],
            "cooldown_days": [5, 10, 20, 30],
            "sample_weight_mode": ["future_max_ret", "target_future_max_ret", "uniform"],
        }
    return {
        "feature_set": ["all", "simple_core", "no_intraday_shape", "no_share_turnover", "liquidity_momentum"],
        "gate_min_turnover_5d": [50_000_000, 100_000_000],
        "gate_min_ret_5d": [-0.02, -0.01, 0.0],
        "gate_ratio_profile": ["loose", "current", "strict"],
        "gate_min_close_ma25_ratio": [-0.01, 0.0],
        "cooldown_days": [10, 20],
        "sample_weight_mode": ["future_max_ret", "target_future_max_ret", "uniform"],
    }


def coordinate_total_trials(size: str, passes: int) -> int:
    return sum(len(values) for values in coordinate_values(size).values()) * max(passes, 1)


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_search_space(size: str) -> list[dict[str, Any]]:
    if size == "smoke":
        feature_sets = ["all", "simple_core"]
        turnovers = [50_000_000]
        ret_5d = [-0.01]
        ratio_profiles = ["current"]
        ma25 = [-0.01]
        cooldowns = [20]
        weight_modes = ["future_max_ret", "target_future_max_ret"]
    elif size == "medium":
        feature_sets = list(FEATURE_SETS)
        turnovers = [30_000_000, 50_000_000, 100_000_000]
        ret_5d = [-0.02, -0.01, 0.0]
        ratio_profiles = list(RATIO_PROFILES)
        ma25 = [-0.03, -0.01, 0.0]
        cooldowns = [10, 20, 30]
        weight_modes = ["future_max_ret", "target_future_max_ret", "uniform"]
    elif size == "full":
        feature_sets = list(FEATURE_SETS)
        turnovers = [20_000_000, 30_000_000, 50_000_000, 100_000_000, 200_000_000]
        ret_5d = [-0.03, -0.02, -0.01, 0.0, 0.02]
        ratio_profiles = list(RATIO_PROFILES)
        ma25 = [-0.05, -0.03, -0.01, 0.0, 0.02]
        cooldowns = [5, 10, 20, 30]
        weight_modes = ["future_max_ret", "target_future_max_ret", "uniform"]
    else:
        feature_sets = ["all", "simple_core", "no_intraday_shape", "no_share_turnover", "liquidity_momentum"]
        turnovers = [50_000_000, 100_000_000]
        ret_5d = [-0.02, -0.01, 0.0]
        ratio_profiles = ["loose", "current", "strict"]
        ma25 = [-0.01, 0.0]
        cooldowns = [10, 20]
        weight_modes = ["future_max_ret", "target_future_max_ret", "uniform"]

    trials = []
    for feature_set, turnover, ret, ratio_name, close_ma25, cooldown, weight_mode in itertools.product(
        feature_sets,
        turnovers,
        ret_5d,
        ratio_profiles,
        ma25,
        cooldowns,
        weight_modes,
    ):
        ratio_1d, ratio_5d = RATIO_PROFILES[ratio_name]
        trials.append(
            {
                "feature_set": feature_set,
                "gate_min_turnover_5d": turnover,
                "gate_min_ret_5d": ret,
                "gate_ratio_profile": ratio_name,
                "gate_min_turnover_ratio_1d_20d": ratio_1d,
                "gate_min_turnover_ratio_5d_20d": ratio_5d,
                "gate_min_close_ma25_ratio": close_ma25,
                "cooldown_days": cooldown,
                "sample_weight_mode": weight_mode,
            }
        )
    return trials


def score_trial(summary: dict[str, float]) -> float:
    valid_p20 = summary.get("valid_precision_at_20_mean", float("nan"))
    valid_p50 = summary.get("valid_precision_at_50_mean", float("nan"))
    stop20 = summary.get("valid_stop_first_rate_at_20_mean", float("nan"))
    stop_penalty = 0.0 if np.isnan(stop20) else 0.15 * stop20
    if np.isnan(valid_p20):
        return float("-inf")
    return float(valid_p20 + 0.5 * (0.0 if np.isnan(valid_p50) else valid_p50) - stop_penalty)


def summarize_trial(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(fold_rows)
    summary: dict[str, Any] = {}
    metric_columns = [
        column
        for column in frame.columns
        if column.endswith(("_at_20", "_at_50", "_loss"))
        or column.endswith("_events")
        or column in {"epochs_trained", "best_valid_loss"}
    ]
    for column in metric_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_min"] = float(frame[column].min())
    summary["folds"] = int(len(frame))
    summary["score"] = score_trial(summary)
    return summary


def write_progress(output_dir: Path, trial_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(trial_rows).to_csv(output_dir / "trials.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "folds.csv", index=False)
    best = max(trial_rows, key=lambda row: row.get("score", float("-inf"))) if trial_rows else {}
    (output_dir / "best_config.json").write_text(
        json.dumps(clean_for_json(best), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_status(
    output_dir: Path,
    *,
    status: str,
    completed_trials: int,
    total_trials: int,
    started_at: float,
    stop_file: Path,
    current_trial: dict[str, Any] | None = None,
    best_trial: dict[str, Any] | None = None,
    message: str | None = None,
) -> None:
    elapsed = time.monotonic() - started_at
    avg_seconds = elapsed / completed_trials if completed_trials else None
    remaining_trials = max(total_trials - completed_trials, 0)
    eta_seconds = avg_seconds * remaining_trials if avg_seconds is not None else None
    payload = {
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_trials": completed_trials,
        "total_trials": total_trials,
        "progress_pct": (completed_trials / total_trials * 100.0) if total_trials else 100.0,
        "elapsed_seconds": elapsed,
        "avg_seconds_per_trial": avg_seconds,
        "eta_seconds": eta_seconds,
        "stop_file": str(stop_file),
        "current_trial": current_trial,
        "best_trial_id": None if not best_trial else best_trial.get("trial_id"),
        "best_score": None if not best_trial else best_trial.get("score"),
        "best_feature_set": None if not best_trial else best_trial.get("feature_set"),
    }
    (output_dir / "progress.json").write_text(
        json.dumps(clean_for_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_trial(
    trial_id: int,
    config: dict[str, Any],
    features: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    folds: list[tuple[str, str, str]],
    stop_file: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_columns = FEATURE_SETS[config["feature_set"]]
    gated = add_initial_momentum_gate(
        features,
        cooldown_days=config["cooldown_days"],
        min_turnover_5d=config["gate_min_turnover_5d"],
        min_ret_5d=config["gate_min_ret_5d"],
        min_turnover_ratio_1d_20d=config["gate_min_turnover_ratio_1d_20d"],
        min_turnover_ratio_5d_20d=config["gate_min_turnover_ratio_5d_20d"],
        min_close_ma25_ratio=config["gate_min_close_ma25_ratio"],
    )
    labelled = add_labels(
        gated,
        horizon=args.horizon,
        threshold=args.target_threshold,
        label_mode=args.label_mode,
        profit_barrier=args.profit_barrier,
        stop_barrier=args.stop_barrier,
        sample_weight_mode=config["sample_weight_mode"],
        sample_weight_cap=args.sample_weight_cap,
        sample_weight_scale=args.sample_weight_scale,
    )
    events = make_event_dataset(labelled, require_label=True, feature_columns=feature_columns)
    if len(events) < args.min_events:
        summary = {
            "trial_id": trial_id,
            **config,
            "status": "skipped",
            "reason": f"events<{args.min_events}",
            "events": int(len(events)),
            "score": float("-inf"),
        }
        return summary, []

    fold_rows = []
    for fold_name, train_end, valid_end in folds:
        if stop_file.exists():
            break
        result, metrics = train_model(
            events=events,
            feature_columns=feature_columns,
            train_end=train_end,
            valid_end=valid_end,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )
        if args.save_models:
            model_path = output_dir / "models" / f"trial_{trial_id:04d}_{fold_name}.pt"
            save_artifacts(result, model_path)
        row = {
            "trial_id": trial_id,
            "fold": fold_name,
            "train_end": train_end,
            "valid_end": valid_end,
            "events": int(len(events)),
            "feature_count": len(feature_columns),
            **config,
        }
        row.update(metrics)
        fold_rows.append(row)

    status = "stopped_partial" if len(fold_rows) < len(folds) else "ok"
    summary = {
        "trial_id": trial_id,
        **config,
        "status": status,
        "events": int(len(events)),
        "feature_count": len(feature_columns),
        "features": "|".join(feature_columns),
    }
    if fold_rows:
        summary.update(summarize_trial(fold_rows))
    else:
        summary["score"] = float("-inf")
    return summary, fold_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated parameter and feature-set optimization experiments.")
    parser.add_argument("--cache", default="data/ohlcv_current.csv")
    parser.add_argument("--output-dir", default="experiments/outputs/optimization")
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--algorithm",
        choices=["random", "coordinate"],
        default="coordinate",
        help="random samples combinations; coordinate changes one parameter at a time and keeps improvements.",
    )
    parser.add_argument(
        "--search-size",
        choices=["smoke", "small", "focused", "focused_ma_cooldown", "medium", "full"],
        default="small",
    )
    parser.add_argument("--max-trials", type=int, default=60)
    parser.add_argument("--coordinate-passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-events", type=int, default=50)
    parser.add_argument("--fold", action="append", type=parse_rolling_fold)
    parser.add_argument("--label-mode", choices=["barrier", "max_ret"], default=LABEL_SETTINGS["mode"])
    parser.add_argument("--profit-barrier", type=float, default=LABEL_SETTINGS["profit_barrier"])
    parser.add_argument("--stop-barrier", type=float, default=LABEL_SETTINGS["stop_barrier"])
    parser.add_argument("--target-threshold", type=float, default=LABEL_SETTINGS["target_threshold"])
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--sample-weight-cap", type=float, default=0.30)
    parser.add_argument("--sample-weight-scale", type=float, default=10.0)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Create this file to stop gracefully after the current fold. Defaults to <study>/STOP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_name = args.study_name or datetime.now().strftime("study_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file) if args.stop_file else output_dir / "STOP"

    folds = args.fold or default_rolling_folds()
    search_space = build_search_space(args.search_size)
    if args.algorithm == "random":
        rng = random.Random(args.seed)
        rng.shuffle(search_space)
        trials = search_space[: max(args.max_trials, 1)]
        total_trials = len(trials)
        search_space_count = len(search_space)
    else:
        trials = []
        total_trials = min(coordinate_total_trials(args.search_size, args.coordinate_passes), max(args.max_trials, 1))
        search_space_count = coordinate_total_trials(args.search_size, args.coordinate_passes)

    (output_dir / "config.json").write_text(
        json.dumps(
            clean_for_json(
                {
                    "args": vars(args),
                    "folds": folds,
                    "trial_count": total_trials,
                    "search_space_count": search_space_count,
                    "stop_file": str(stop_file),
                    "base_config": BASE_CONFIG,
                    "feature_sets": FEATURE_SETS,
                    "ratio_profiles": RATIO_PROFILES,
                    "coordinate_values": coordinate_values(args.search_size),
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"study={study_name} output_dir={output_dir}")
    print(f"progress={output_dir / 'progress.json'}")
    print(f"stop_file={stop_file}")
    print(f"loading cache={args.cache}")
    started_at = time.monotonic()
    write_status(
        output_dir,
        status="loading",
        completed_trials=0,
        total_trials=total_trials,
        started_at=started_at,
        stop_file=stop_file,
        message="Loading cache and building features.",
    )
    ohlcv = load_or_download_ohlcv(
        symbols=["CACHE_ONLY"],
        start="2020-01-01",
        end=None,
        cache_path=args.cache,
        refresh=False,
        batch_size=100,
    )
    print(f"building features rows={len(ohlcv)} symbols={ohlcv['code'].nunique()}")
    features = add_features(ohlcv)

    trial_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    if args.algorithm == "random":
        run_random_search(trials, features, args, output_dir, folds, stop_file, started_at, total_trials, trial_rows, all_fold_rows)
    else:
        run_coordinate_search(features, args, output_dir, folds, stop_file, started_at, total_trials, trial_rows, all_fold_rows)

    if not trial_rows:
        write_status(
            output_dir,
            status="stopped",
            completed_trials=0,
            total_trials=total_trials,
            started_at=started_at,
            stop_file=stop_file,
            message="Stopped before any trial completed.",
        )
        print("No trials completed.")
        return

    best = max(trial_rows, key=lambda row: row.get("score", float("-inf")))
    final_status = "stopped" if stop_file.exists() else "finished"
    write_status(
        output_dir,
        status=final_status,
        completed_trials=len(trial_rows),
        total_trials=total_trials,
        started_at=started_at,
        stop_file=stop_file,
        best_trial=best,
    )
    print()
    print("best_config")
    print(json.dumps(clean_for_json(best), indent=2, ensure_ascii=False))
    print(f"summary={output_dir / 'trials.csv'}")
    print(f"folds={output_dir / 'folds.csv'}")


def run_random_search(
    trials: list[dict[str, Any]],
    features: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    folds: list[tuple[str, str, str]],
    stop_file: Path,
    started_at: float,
    total_trials: int,
    trial_rows: list[dict[str, Any]],
    all_fold_rows: list[dict[str, Any]],
) -> None:
    for trial_id, config in enumerate(trials, start=1):
        if stop_file.exists():
            print(f"stop file found before trial {trial_id}: {stop_file}", flush=True)
            break
        best_trial = max(trial_rows, key=lambda row: row.get("score", float("-inf"))) if trial_rows else None
        write_status(
            output_dir,
            status="running",
            completed_trials=len(trial_rows),
            total_trials=total_trials,
            started_at=started_at,
            stop_file=stop_file,
            current_trial={"trial_id": trial_id, **config},
            best_trial=best_trial,
        )
        print(
            f"[{trial_id}/{total_trials}] feature_set={config['feature_set']} "
            f"turnover={config['gate_min_turnover_5d']:.0f} ret5={config['gate_min_ret_5d']} "
            f"ratio={config['gate_ratio_profile']} ma25={config['gate_min_close_ma25_ratio']} "
            f"cooldown={config['cooldown_days']} weight={config['sample_weight_mode']}",
            flush=True,
        )
        try:
            summary, fold_rows = run_trial(trial_id, config, features, args, output_dir, folds, stop_file)
        except Exception as exc:  # noqa: BLE001 - keep long experiments running.
            summary = {"trial_id": trial_id, **config, "status": "error", "reason": repr(exc), "score": float("-inf")}
            fold_rows = []
            print(f"  error={exc!r}", flush=True)
        trial_rows.append(summary)
        all_fold_rows.extend(fold_rows)
        write_progress(output_dir, trial_rows, all_fold_rows)
        best_trial = max(trial_rows, key=lambda row: row.get("score", float("-inf")))
        write_status(
            output_dir,
            status="stopping" if stop_file.exists() else "running",
            completed_trials=len(trial_rows),
            total_trials=total_trials,
            started_at=started_at,
            stop_file=stop_file,
            current_trial={"trial_id": trial_id, **config},
            best_trial=best_trial,
            message="Stop file found; finishing after current saved progress." if stop_file.exists() else None,
        )
        print(
            f"  status={summary.get('status')} score={summary.get('score')} "
            f"valid_p20={summary.get('valid_precision_at_20_mean')} "
            f"test_p50={summary.get('test_precision_at_50_mean')}",
            flush=True,
        )
        if stop_file.exists():
            print(f"stop file found after trial {trial_id}: {stop_file}", flush=True)
            break


def run_coordinate_search(
    features: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    folds: list[tuple[str, str, str]],
    stop_file: Path,
    started_at: float,
    total_trials: int,
    trial_rows: list[dict[str, Any]],
    all_fold_rows: list[dict[str, Any]],
) -> None:
    values_by_axis = coordinate_values(args.search_size)
    current_config = dict(BASE_CONFIG)
    best_overall: dict[str, Any] | None = None
    trial_id = 0

    for pass_no in range(1, max(args.coordinate_passes, 1) + 1):
        improved_in_pass = False
        for axis, values in values_by_axis.items():
            if stop_file.exists() or trial_id >= total_trials:
                return

            axis_best_summary: dict[str, Any] | None = None
            axis_best_config = dict(current_config)
            for value in values:
                if stop_file.exists() or trial_id >= total_trials:
                    return
                trial_id += 1
                config = dict(current_config)
                if axis == "gate_ratio_profile":
                    config = with_ratio_profile(config, value)
                else:
                    config[axis] = value
                config["coordinate_pass"] = pass_no
                config["coordinate_axis"] = axis
                config["coordinate_value"] = value

                best_trial = max(trial_rows, key=lambda row: row.get("score", float("-inf"))) if trial_rows else None
                write_status(
                    output_dir,
                    status="running",
                    completed_trials=len(trial_rows),
                    total_trials=total_trials,
                    started_at=started_at,
                    stop_file=stop_file,
                    current_trial={"trial_id": trial_id, **config},
                    best_trial=best_trial,
                )
                print(
                    f"[{trial_id}/{total_trials}] pass={pass_no} axis={axis} value={value} "
                    f"feature_set={config['feature_set']} turnover={config['gate_min_turnover_5d']:.0f} "
                    f"ret5={config['gate_min_ret_5d']} ratio={config['gate_ratio_profile']} "
                    f"ma25={config['gate_min_close_ma25_ratio']} cooldown={config['cooldown_days']} "
                    f"weight={config['sample_weight_mode']}",
                    flush=True,
                )
                try:
                    summary, fold_rows = run_trial(trial_id, config, features, args, output_dir, folds, stop_file)
                except Exception as exc:  # noqa: BLE001 - keep long experiments running.
                    summary = {"trial_id": trial_id, **config, "status": "error", "reason": repr(exc), "score": float("-inf")}
                    fold_rows = []
                    print(f"  error={exc!r}", flush=True)

                trial_rows.append(summary)
                all_fold_rows.extend(fold_rows)
                if axis_best_summary is None or summary.get("score", float("-inf")) > axis_best_summary.get("score", float("-inf")):
                    axis_best_summary = summary
                    axis_best_config = dict(config)
                if best_overall is None or summary.get("score", float("-inf")) > best_overall.get("score", float("-inf")):
                    best_overall = summary

                write_progress(output_dir, trial_rows, all_fold_rows)
                best_trial = max(trial_rows, key=lambda row: row.get("score", float("-inf")))
                write_status(
                    output_dir,
                    status="stopping" if stop_file.exists() else "running",
                    completed_trials=len(trial_rows),
                    total_trials=total_trials,
                    started_at=started_at,
                    stop_file=stop_file,
                    current_trial={"trial_id": trial_id, **config},
                    best_trial=best_trial,
                    message="Stop file found; finishing after current saved progress." if stop_file.exists() else None,
                )
                print(
                    f"  status={summary.get('status')} score={summary.get('score')} "
                    f"valid_p20={summary.get('valid_precision_at_20_mean')} "
                    f"test_p50={summary.get('test_precision_at_50_mean')}",
                    flush=True,
                )

            if axis_best_summary is not None:
                previous_value = current_config.get(axis)
                if axis == "gate_ratio_profile":
                    axis_best_config = with_ratio_profile(axis_best_config, axis_best_config["gate_ratio_profile"])
                current_config = {
                    key: value
                    for key, value in axis_best_config.items()
                    if not key.startswith("coordinate_") and key != "trial_id" and key != "status"
                }
                new_value = current_config.get(axis)
                improved_in_pass = improved_in_pass or previous_value != new_value
                print(
                    f"  selected axis={axis} value={new_value} score={axis_best_summary.get('score')}",
                    flush=True,
                )
        if not improved_in_pass:
            print(f"no coordinate changes in pass {pass_no}; stopping early", flush=True)
            break


if __name__ == "__main__":
    main()
