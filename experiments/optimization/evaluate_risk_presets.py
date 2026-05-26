from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from momentum_screener.cli import default_rolling_folds, parse_rolling_fold
from momentum_screener.data import load_or_download_ohlcv
from momentum_screener.features import FEATURE_COLUMNS, add_features, add_initial_momentum_gate, add_labels, make_event_dataset
from momentum_screener.model import DEFAULT_RISK_ADJUSTMENT, RISK_ADJUSTMENT_PRESETS, evaluate_splits, train_model
from momentum_screener.settings import GATE_SETTINGS, LABEL_SETTINGS


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


def split_by_date(events: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)
    train = events[events["date"] <= train_end_ts].copy()
    valid = events[(events["date"] > train_end_ts) & (events["date"] <= valid_end_ts)].copy()
    test = events[events["date"] > valid_end_ts].copy()
    return train, valid, test


def score_row(row: dict[str, Any]) -> float:
    valid_p20 = row.get("valid_precision_at_20", float("nan"))
    valid_p50 = row.get("valid_precision_at_50", float("nan"))
    stop50 = row.get("valid_stop_first_rate_at_50", float("nan"))
    max50 = row.get("valid_avg_future_max_ret_at_50", float("nan"))
    min50 = row.get("valid_avg_future_min_ret_at_50", float("nan"))
    if np.isnan(valid_p20):
        return float("-inf")
    return float(
        valid_p20
        + 0.5 * (0.0 if np.isnan(valid_p50) else valid_p50)
        - 0.35 * (0.0 if np.isnan(stop50) else stop50)
        + 0.35 * (0.0 if np.isnan(max50) else max50)
        + 0.35 * (0.0 if np.isnan(min50) else min50)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate risk-adjustment presets without retraining per preset.")
    parser.add_argument("--cache", default="data/ohlcv_current.csv")
    parser.add_argument("--output-dir", default="experiments/outputs/risk_presets")
    parser.add_argument("--fold", action="append", type=parse_rolling_fold)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-events", type=int, default=50)
    parser.add_argument("--label-mode", choices=["barrier", "max_ret"], default=LABEL_SETTINGS["mode"])
    parser.add_argument("--profit-barrier", type=float, default=LABEL_SETTINGS["profit_barrier"])
    parser.add_argument("--stop-barrier", type=float, default=LABEL_SETTINGS["stop_barrier"])
    parser.add_argument("--target-threshold", type=float, default=LABEL_SETTINGS["target_threshold"])
    parser.add_argument("--horizon", type=int, default=LABEL_SETTINGS["horizon"])
    parser.add_argument("--sample-weight-mode", choices=["future_max_ret", "target_future_max_ret", "uniform"], default="future_max_ret")
    parser.add_argument("--sample-weight-cap", type=float, default=0.30)
    parser.add_argument("--sample-weight-scale", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = args.fold or default_rolling_folds()

    ohlcv = load_or_download_ohlcv(
        symbols=["CACHE_ONLY"],
        start="2020-01-01",
        end=None,
        cache_path=args.cache,
        refresh=False,
        batch_size=100,
    )
    features = add_features(ohlcv)
    gated = add_initial_momentum_gate(
        features,
        cooldown_days=GATE_SETTINGS["cooldown_days"],
        min_turnover_5d=GATE_SETTINGS["min_turnover_5d"],
        min_ret_5d=GATE_SETTINGS["min_ret_5d"],
        min_turnover_ratio_1d_20d=GATE_SETTINGS["min_turnover_ratio_1d_20d"],
        min_turnover_ratio_5d_20d=GATE_SETTINGS["min_turnover_ratio_5d_20d"],
        min_close_ma25_ratio=GATE_SETTINGS["min_close_ma25_ratio"],
    )
    labelled = add_labels(
        gated,
        horizon=args.horizon,
        threshold=args.target_threshold,
        label_mode=args.label_mode,
        profit_barrier=args.profit_barrier,
        stop_barrier=args.stop_barrier,
        sample_weight_mode=args.sample_weight_mode,
        sample_weight_cap=args.sample_weight_cap,
        sample_weight_scale=args.sample_weight_scale,
    )
    events = make_event_dataset(labelled, require_label=True, feature_columns=FEATURE_COLUMNS)
    if len(events) < args.min_events:
        raise ValueError(f"Only {len(events)} labelled events were created.")

    rows: list[dict[str, Any]] = []
    for fold_name, train_end, valid_end in folds:
        result, training_metrics = train_model(
            events=events,
            feature_columns=FEATURE_COLUMNS,
            train_end=train_end,
            valid_end=valid_end,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            risk_adjustment=DEFAULT_RISK_ADJUSTMENT,
        )
        train, valid, test = split_by_date(events, train_end, valid_end)
        for preset in RISK_ADJUSTMENT_PRESETS:
            metrics = evaluate_splits(
                result.model,
                result.scaler,
                result.feature_columns,
                {"train": train, "valid": valid, "test": test},
                risk_adjustment=preset,
            )
            row = {
                "fold": fold_name,
                "train_end": train_end,
                "valid_end": valid_end,
                "risk_adjustment": preset,
                "all_events": float(len(events)),
                "best_valid_loss": training_metrics["best_valid_loss"],
                "best_selection_score": training_metrics.get("best_selection_score"),
                "epochs_trained": training_metrics["epochs_trained"],
            }
            row.update(metrics)
            row["score"] = score_row(row)
            rows.append(row)
            print(
                f"fold={fold_name} risk={preset} score={row['score']:.4f} "
                f"valid_p50={row.get('valid_precision_at_50')} "
                f"valid_stop50={row.get('valid_stop_first_rate_at_50')}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("risk_adjustment", as_index=False)
        .mean(numeric_only=True)
        .sort_values("score", ascending=False)
    )
    frame.to_csv(output_dir / "folds.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(clean_for_json(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"folds={output_dir / 'folds.csv'}")
    print(f"summary={output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
