from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_screener.data import load_or_download_ohlcv, read_tickers, refresh_ohlcv_cache
from momentum_screener.features import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    add_features,
    add_initial_momentum_gate,
    add_labels,
    make_event_dataset,
    make_reason,
    raw_gate_mask,
)
from momentum_screener.model import load_artifacts, predict_proba, save_artifacts, train_model
from momentum_screener.settings import GATE_SETTINGS, LABEL_SETTINGS, SCREEN_SETTINGS


def build_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.cache_only:
        symbols = ["CACHE_ONLY"]
        print(f"cache_only={args.cache}")
    else:
        tickers_file = None if args.no_sample_tickers else args.tickers_file
        symbols = read_tickers(
            tickers_file,
            args.ticker,
            ticker_universe=args.ticker_universe,
            code_start=args.code_start,
            code_end=args.code_end,
            max_tickers=args.max_tickers,
            ticker_csv=args.ticker_csv,
            ticker_csv_code_column=args.ticker_csv_code_column,
            ticker_csv_market_column=args.ticker_csv_market_column,
            ticker_csv_include_markets=args.ticker_csv_include_market,
            ticker_csv_product_column=args.ticker_csv_product_column,
            ticker_csv_include_products=args.ticker_csv_include_product,
            ticker_csv_exclude_products=args.ticker_csv_exclude_product,
        )
        if not symbols:
            raise ValueError("No tickers were supplied. Use --tickers-file, --ticker, --ticker-csv, or --cache-only.")
        print(f"ticker_count={len(symbols)}")
    ohlcv = load_or_download_ohlcv(
        symbols=symbols,
        start=args.start,
        end=args.end,
        cache_path=args.cache,
        refresh=args.refresh,
        batch_size=args.download_batch_size,
    )
    if ohlcv.empty:
        raise ValueError("No OHLCV rows were downloaded. Check ticker symbols and network access.")
    shares = pd.read_csv(args.shares_csv) if args.shares_csv else None
    features = add_features(ohlcv, shares=shares)
    gated = add_initial_momentum_gate(
        features,
        cooldown_days=args.cooldown_days,
        min_turnover_5d=args.gate_min_turnover_5d,
        min_ret_5d=args.gate_min_ret_5d,
        min_turnover_ratio_1d_20d=args.gate_min_turnover_ratio_1d_20d,
        min_turnover_ratio_5d_20d=args.gate_min_turnover_ratio_5d_20d,
        min_close_ma25_ratio=args.gate_min_close_ma25_ratio,
    )
    return add_labels(
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


def get_feature_columns(args: argparse.Namespace) -> list[str]:
    return FEATURE_COLUMNS


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    tickers_file = None if args.no_sample_tickers else args.tickers_file
    symbols = read_tickers(
        tickers_file,
        args.ticker,
        ticker_universe=args.ticker_universe,
        code_start=args.code_start,
        code_end=args.code_end,
        max_tickers=args.max_tickers,
        ticker_csv=args.ticker_csv,
        ticker_csv_code_column=args.ticker_csv_code_column,
        ticker_csv_market_column=args.ticker_csv_market_column,
        ticker_csv_include_markets=args.ticker_csv_include_market,
        ticker_csv_product_column=args.ticker_csv_product_column,
        ticker_csv_include_products=args.ticker_csv_include_product,
        ticker_csv_exclude_products=args.ticker_csv_exclude_product,
    )
    if not symbols:
        raise ValueError("No tickers were supplied. Use --tickers-file, --ticker, or --ticker-csv.")
    return symbols


def compute_gate_recall(args: argparse.Namespace, labelled: pd.DataFrame, threshold: float) -> float:
    labelled = labelled.dropna(subset=["future_max_ret_20d"]).copy()
    winners = labelled["future_max_ret_20d"] >= threshold
    if winners.sum() == 0:
        return float("nan")
    raw_gate = raw_gate_mask(
        labelled,
        min_turnover_5d=args.gate_min_turnover_5d,
        min_ret_5d=args.gate_min_ret_5d,
        min_turnover_ratio_1d_20d=args.gate_min_turnover_ratio_1d_20d,
        min_turnover_ratio_5d_20d=args.gate_min_turnover_ratio_5d_20d,
        min_close_ma25_ratio=args.gate_min_close_ma25_ratio,
    )
    return float((raw_gate & winners).sum() / winners.sum())


def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_training_config(args: argparse.Namespace, feature_columns: list[str]) -> dict:
    return {
        "data": {
            "cache": str(args.cache),
            "cache_only": bool(args.cache_only),
            "start": args.start,
            "end": args.end,
            "ticker_csv": args.ticker_csv,
            "ticker_csv_code_column": args.ticker_csv_code_column,
            "ticker_universe": args.ticker_universe,
            "code_start": args.code_start,
            "code_end": args.code_end,
            "max_tickers": args.max_tickers,
            "shares_csv": args.shares_csv,
        },
        "gate": {
            "cooldown_days": args.cooldown_days,
            "min_turnover_5d": args.gate_min_turnover_5d,
            "min_ret_5d": args.gate_min_ret_5d,
            "min_turnover_ratio_1d_20d": args.gate_min_turnover_ratio_1d_20d,
            "min_turnover_ratio_5d_20d": args.gate_min_turnover_ratio_5d_20d,
            "min_close_ma25_ratio": args.gate_min_close_ma25_ratio,
        },
        "label": {
            "mode": args.label_mode,
            "horizon": args.horizon,
            "target_threshold": args.target_threshold,
            "profit_barrier": args.profit_barrier,
            "stop_barrier": args.stop_barrier,
        },
        "sample_weight": {
            "mode": args.sample_weight_mode,
            "cap": args.sample_weight_cap,
            "scale": args.sample_weight_scale,
        },
        "model": {
            "feature_columns": feature_columns,
            "train_end": getattr(args, "train_end", None),
            "valid_end": getattr(args, "valid_end", None),
            "epochs": getattr(args, "epochs", None),
            "batch_size": getattr(args, "batch_size", None),
            "learning_rate": getattr(args, "learning_rate", None),
            "patience": getattr(args, "patience", None),
            "seed": getattr(args, "seed", None),
            "min_events": getattr(args, "min_events", None),
        },
    }


def write_metrics(metrics: dict, path: str | Path) -> None:
    clean = clean_for_json(metrics)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_caches_command(args: argparse.Namespace) -> None:
    frames = []
    for cache_path in args.input:
        frame = pd.read_csv(cache_path, parse_dates=["date"])
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last")
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"rows={len(merged)} symbols={merged['code'].nunique()} output={output}")


def refresh_data_command(args: argparse.Namespace) -> None:
    symbols = resolve_symbols(args)
    print(f"ticker_count={len(symbols)}")
    refreshed = refresh_ohlcv_cache(
        symbols=symbols,
        start=args.start,
        end=args.end,
        cache_path=args.cache,
        batch_size=args.download_batch_size,
    )
    print(
        f"rows={len(refreshed)} symbols={refreshed['code'].nunique()} "
        f"min_date={refreshed['date'].min().date()} max_date={refreshed['date'].max().date()} cache={args.cache}"
    )
def build_listed_stocks_command(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, dtype=str, encoding=args.encoding)
    required = ["銘柄コード", "国内外区分", "商品分類", "市場区分", "東証上場廃止日"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"JPX issues CSV is missing columns: {missing}")

    markets = args.market or ["プライム", "スタンダード", "グロース"]
    stocks = df[
        (df["国内外区分"].eq("国内"))
        & (df["商品分類"].eq("株式"))
        & (df["市場区分"].isin(markets))
        & (df["東証上場廃止日"].isna())
    ].copy()
    stocks["code"] = stocks["銘柄コード"].str[:4]

    columns = ["code", "銘柄名称", "市場区分", "業種", "売買単位", "東証上場日"]
    output = stocks[columns].drop_duplicates("code").sort_values("code")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"rows={len(output)} markets={output['市場区分'].value_counts().to_dict()} output={output_path}")


def train_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    feature_columns = get_feature_columns(args)
    events = make_event_dataset(labelled, require_label=True, feature_columns=feature_columns)
    if len(events) < args.min_events:
        raise ValueError(
            f"Only {len(events)} labelled events were created. "
            f"Add more tickers/history or lower --min-events for a smoke test."
        )

    result, metrics = train_model(
        events=events,
        feature_columns=feature_columns,
        train_end=args.train_end,
        valid_end=args.valid_end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    metrics["all_events"] = float(len(events))
    metrics["gate_recall"] = compute_gate_recall(args, labelled, args.target_threshold)
    metrics["raw_gate_rows"] = float(labelled["raw_initial_momentum"].sum())
    metrics["initial_momentum_events"] = float(labelled["initial_momentum"].sum())
    metrics["symbols_with_ohlcv"] = float(labelled["code"].nunique())
    metrics["label_mode"] = args.label_mode
    metrics["profit_barrier"] = args.profit_barrier
    metrics["stop_barrier"] = args.stop_barrier
    metrics["target_threshold"] = args.target_threshold
    metrics["sample_weight_mode"] = args.sample_weight_mode
    metrics["sample_weight_cap"] = args.sample_weight_cap
    metrics["sample_weight_scale"] = args.sample_weight_scale
    metrics["training_config"] = build_training_config(args, feature_columns)
    save_artifacts(result, args.model_path)
    write_metrics(metrics, args.metrics_path)
    print(f"trained_events={len(events)} model={args.model_path} metrics={args.metrics_path}")


def parse_rolling_fold(value: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 2:
        train_end, valid_end = parts
        name = f"train_to_{train_end}_valid_to_{valid_end}"
    elif len(parts) == 3:
        name, train_end, valid_end = parts
    else:
        raise argparse.ArgumentTypeError("Use --fold NAME,TRAIN_END,VALID_END or --fold TRAIN_END,VALID_END.")
    if not train_end or not valid_end:
        raise argparse.ArgumentTypeError("TRAIN_END and VALID_END are required.")
    return name, train_end, valid_end


def default_rolling_folds() -> list[tuple[str, str, str]]:
    return [
        ("valid_2023", "2022-12-31", "2023-12-31"),
        ("valid_2024", "2023-12-31", "2024-12-31"),
        ("valid_2025", "2024-12-31", "2025-12-31"),
    ]


def rolling_eval_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    feature_columns = get_feature_columns(args)
    events = make_event_dataset(labelled, require_label=True, feature_columns=feature_columns)
    if len(events) < args.min_events:
        raise ValueError(
            f"Only {len(events)} labelled events were created. "
            f"Add more tickers/history or lower --min-events for a smoke test."
        )

    folds = args.fold or default_rolling_folds()
    rows = []
    for fold_name, train_end, valid_end in folds:
        _, metrics = train_model(
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
        row = {
            "fold": fold_name,
            "train_end": train_end,
            "valid_end": valid_end,
            "all_events": float(len(events)),
            "gate_recall": compute_gate_recall(args, labelled, args.target_threshold),
            "raw_gate_rows": float(labelled["raw_initial_momentum"].sum()),
            "initial_momentum_events": float(labelled["initial_momentum"].sum()),
            "symbols_with_ohlcv": float(labelled["code"].nunique()),
            "label_mode": args.label_mode,
            "profit_barrier": args.profit_barrier,
            "stop_barrier": args.stop_barrier,
            "target_threshold": args.target_threshold,
            "sample_weight_mode": args.sample_weight_mode,
            "sample_weight_cap": args.sample_weight_cap,
            "sample_weight_scale": args.sample_weight_scale,
            "training_config": build_training_config(args, feature_columns),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"fold={fold_name} train_end={train_end} valid_end={valid_end} "
            f"valid_p20={row.get('valid_precision_at_20')} valid_p50={row.get('valid_precision_at_50')} "
            f"test_p20={row.get('test_precision_at_20')} test_p50={row.get('test_precision_at_50')}"
        )

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(clean_for_json(rows), indent=2, ensure_ascii=False), encoding="utf-8")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"folds={len(rows)} metrics={metrics_path} output={output_path}")


def inspect_gate_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    feature_columns = get_feature_columns(args)
    events = make_event_dataset(labelled, require_label=True, feature_columns=feature_columns)
    labelled_with_future = labelled.dropna(subset=["future_max_ret_20d"])
    winners = labelled_with_future["future_max_ret_20d"] >= args.target_threshold
    metrics = {
        "symbols_with_ohlcv": float(labelled["code"].nunique()),
        "rows_with_future": float(len(labelled_with_future)),
        "winner_rows": float(winners.sum()),
        "raw_gate_rows": float(labelled["raw_initial_momentum"].sum()),
        "initial_momentum_events": float(labelled["initial_momentum"].sum()),
        "labelled_events": float(len(events)),
        "event_success_rate": float(events["target_20d"].mean()) if not events.empty else float("nan"),
        "event_avg_future_max_ret_20d": float(events["future_max_ret_20d"].mean()) if not events.empty else float("nan"),
        "gate_recall": compute_gate_recall(args, labelled, args.target_threshold),
        "training_config": build_training_config(args, feature_columns),
    }
    write_metrics(metrics, args.metrics_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics={args.metrics_path}")


def screen_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    feature_columns = get_feature_columns(args)
    raw_candidates = labelled[labelled["raw_initial_momentum"]].copy()
    raw_candidates = raw_candidates.replace([np.inf, -np.inf], np.nan)
    raw_candidates = raw_candidates.dropna(subset=feature_columns).reset_index(drop=True)
    recent_days = max(int(args.recent_days), 1)
    signal_count_days = max(int(args.signal_count_days or recent_days), recent_days)
    latest_date = labelled["date"].dropna().max()
    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
        output_mask = raw_candidates["date"] == as_of
        count_date_source = raw_candidates.loc[raw_candidates["date"] <= as_of, "date"]
    else:
        latest_dates = raw_candidates["date"].dropna().drop_duplicates().sort_values().tail(recent_days)
        output_mask = raw_candidates["date"].isin(latest_dates)
        count_date_source = raw_candidates["date"]

    signal_count_dates = count_date_source.dropna().drop_duplicates().sort_values().tail(signal_count_days)
    candidates = raw_candidates[raw_candidates["date"].isin(signal_count_dates)].copy()

    model, scaler, feature_columns = load_artifacts(args.model_path)
    candidates["follow_through_prob"] = predict_proba(model, scaler, feature_columns, candidates)
    candidates["final_score"] = candidates["follow_through_prob"]
    latest_by_code = labelled.sort_values("date").groupby("code")["close"].last()
    candidates["latest_close"] = candidates["code"].map(latest_by_code)
    candidates["return_since_candidate"] = candidates["latest_close"] / candidates["close"] - 1.0
    candidates["recent_signal_count"] = candidates.groupby("code")["date"].transform("nunique")
    raw_recent = labelled[labelled["date"].isin(signal_count_dates) & labelled["raw_initial_momentum"]].copy()
    raw_recent_counts = raw_recent.groupby("code")["date"].nunique()
    latest_raw_codes = set(
        labelled.loc[
            (labelled["date"] == latest_date) & labelled["raw_initial_momentum"],
            "code",
        ]
    )
    raw_signal_counts = (
        labelled[labelled["raw_initial_momentum"]]
        .groupby("code")["date"]
        .apply(lambda dates: dates.sort_values().to_numpy())
    )
    candidates["raw_recent_signal_count"] = candidates["code"].map(raw_recent_counts).fillna(0).astype(int)
    candidates["raw_signal_count_since_candidate"] = candidates.apply(
        lambda row: int((raw_signal_counts.get(row["code"], np.array([], dtype="datetime64[ns]")) >= row["date"]).sum()),
        axis=1,
    )
    score_count_candidates = candidates[candidates["final_score"] >= args.signal_count_min_score]
    score_recent_counts = score_count_candidates.groupby("code")["date"].nunique()
    score_signal_counts = score_count_candidates.groupby("code")["date"].apply(lambda dates: dates.sort_values().to_numpy())
    first_score_signals = (
        score_count_candidates.sort_values(["code", "date"])
        .groupby("code")
        .first()[["date", "close"]]
        .rename(columns={"date": "first_score_signal_date", "close": "first_score_signal_close"})
        .reset_index()
    )
    candidates["score_recent_signal_count"] = candidates["code"].map(score_recent_counts).fillna(0).astype(int)
    candidates["score_signal_count_since_candidate"] = candidates.apply(
        lambda row: int((score_signal_counts.get(row["code"], np.array([], dtype="datetime64[ns]")) >= row["date"]).sum()),
        axis=1,
    )
    candidates = candidates.merge(first_score_signals, on="code", how="left")
    candidates["return_since_first_score_signal"] = (
        candidates["latest_close"] / candidates["first_score_signal_close"] - 1.0
    )
    candidates["signal_still_active"] = candidates["code"].isin(latest_raw_codes)
    candidates["reason"] = candidates.apply(make_reason, axis=1) if not candidates.empty else []
    output_codes_dates = raw_candidates.loc[output_mask, ["code", "date"]].drop_duplicates()
    candidates = candidates.merge(output_codes_dates, on=["code", "date"], how="inner")
    candidates = candidates.sort_values(["date", "final_score"], ascending=[False, False])

    output = candidates.reindex(columns=OUTPUT_COLUMNS)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"candidates={len(output)} output={output_path}")


def normalize_code_arg(code: str) -> str:
    text = code.strip()
    if text.endswith(".T"):
        return text
    return f"{text}.T"


def inspect_symbols_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    symbols = [normalize_code_arg(code) for code in args.code]
    rows = labelled[labelled["code"].isin(symbols)].copy()
    if rows.empty:
        raise ValueError(f"No rows found for: {symbols}")

    feature_columns = get_feature_columns(args)
    events = make_event_dataset(rows, require_label=False, feature_columns=feature_columns)
    model = scaler = feature_columns = None
    if args.model_path:
        model, scaler, feature_columns = load_artifacts(args.model_path)
        if not events.empty:
            events["follow_through_prob"] = predict_proba(model, scaler, feature_columns, events)

    summaries = []
    for symbol in symbols:
        symbol_rows = rows[rows["code"] == symbol]
        symbol_events = events[events["code"] == symbol].sort_values("date")
        if symbol_rows.empty:
            summaries.append({"code": symbol, "rows": 0, "initial_events": 0})
            continue
        latest = symbol_rows.sort_values("date").iloc[-1]
        summary = {
            "code": symbol,
            "rows": int(len(symbol_rows)),
            "first_date": str(symbol_rows["date"].min().date()),
            "last_date": str(symbol_rows["date"].max().date()),
            "latest_close": float(latest["close"]),
            "raw_gate_rows": int(symbol_rows["raw_initial_momentum"].sum()),
            "initial_events": int(symbol_rows["initial_momentum"].sum()),
        }
        if not symbol_events.empty:
            last_event = symbol_events.iloc[-1]
            best_event = (
                symbol_events.sort_values("follow_through_prob", ascending=False).iloc[0]
                if "follow_through_prob" in symbol_events.columns
                else symbol_events.sort_values("future_max_ret_20d", ascending=False).iloc[0]
            )
            summary.update(
                {
                    "last_event_date": str(last_event["date"].date()),
                    "last_event_future_max_ret_20d": (
                        None if pd.isna(last_event.get("future_max_ret_20d")) else float(last_event["future_max_ret_20d"])
                    ),
                    "best_event_date": str(best_event["date"].date()),
                    "best_event_future_max_ret_20d": (
                        None if pd.isna(best_event.get("future_max_ret_20d")) else float(best_event["future_max_ret_20d"])
                    ),
                }
            )
            if "follow_through_prob" in symbol_events.columns:
                summary["last_event_prob"] = float(last_event["follow_through_prob"])
                summary["best_event_prob"] = float(best_event["follow_through_prob"])
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"output={output_path}")


def run_command(args: argparse.Namespace) -> None:
    train_command(args)
    args.refresh = False
    screen_command(args)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tickers-file", default="config/tickers_sample.txt")
    parser.add_argument("--no-sample-tickers", action="store_true", help="Ignore the default sample ticker file.")
    parser.add_argument("--ticker", action="append", help="Additional yfinance ticker. Can be repeated.")
    parser.add_argument("--ticker-csv", default=None, help="Optional CSV containing listed symbols or codes.")
    parser.add_argument("--ticker-csv-code-column", default="code")
    parser.add_argument("--ticker-csv-market-column", default=None)
    parser.add_argument("--ticker-csv-include-market", action="append")
    parser.add_argument("--ticker-csv-product-column", default=None)
    parser.add_argument("--ticker-csv-include-product", action="append")
    parser.add_argument("--ticker-csv-exclude-product", action="append")
    parser.add_argument(
        "--ticker-universe",
        choices=["none", "tse-all"],
        default="none",
        help="Use generated ticker universe. tse-all tries 1300.T through 9999.T and keeps symbols yfinance returns.",
    )
    parser.add_argument("--code-start", type=int, default=1300)
    parser.add_argument("--code-end", type=int, default=9999)
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit ticker count for trial runs.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache", default="data/ohlcv.csv")
    parser.add_argument("--cache-only", action="store_true", help="Use an existing cache without requiring ticker inputs.")
    parser.add_argument("--download-batch-size", type=int, default=100)
    parser.add_argument("--shares-csv", default=None, help="Optional CSV with code, shares_outstanding, free_float_shares.")
    parser.add_argument("--refresh", action="store_true", help="Download data even when cache exists.")
    parser.add_argument("--cooldown-days", type=int, default=GATE_SETTINGS["cooldown_days"])
    parser.add_argument("--gate-min-turnover-5d", type=float, default=GATE_SETTINGS["min_turnover_5d"])
    parser.add_argument("--gate-min-ret-5d", type=float, default=GATE_SETTINGS["min_ret_5d"])
    parser.add_argument("--gate-min-turnover-ratio-1d-20d", type=float, default=GATE_SETTINGS["min_turnover_ratio_1d_20d"])
    parser.add_argument("--gate-min-turnover-ratio-5d-20d", type=float, default=GATE_SETTINGS["min_turnover_ratio_5d_20d"])
    parser.add_argument("--gate-min-close-ma25-ratio", type=float, default=GATE_SETTINGS["min_close_ma25_ratio"])
    parser.add_argument("--horizon", type=int, default=LABEL_SETTINGS["horizon"])
    parser.add_argument("--target-threshold", type=float, default=LABEL_SETTINGS["target_threshold"])
    parser.add_argument("--label-mode", choices=["max_ret", "barrier"], default=LABEL_SETTINGS["mode"])
    parser.add_argument("--profit-barrier", type=float, default=LABEL_SETTINGS["profit_barrier"])
    parser.add_argument("--stop-barrier", type=float, default=LABEL_SETTINGS["stop_barrier"])
    parser.add_argument(
        "--sample-weight-mode",
        choices=["future_max_ret", "target_future_max_ret", "uniform"],
        default="future_max_ret",
        help=(
            "Training sample weighting. future_max_ret is the legacy behavior; "
            "target_future_max_ret weights only successful labels by future max return."
        ),
    )
    parser.add_argument("--sample-weight-cap", type=float, default=0.30)
    parser.add_argument("--sample-weight-scale", type=float, default=10.0)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-events", type=int, default=50)
    parser.add_argument("--model-path", default="models/momentum_nn.pt")
    parser.add_argument("--metrics-path", default="outputs/metrics.json")


def add_screen_args(parser: argparse.ArgumentParser, include_model_path: bool = True) -> None:
    if include_model_path:
        parser.add_argument("--model-path", default="models/momentum_nn.pt")
    parser.add_argument("--output", default="outputs/candidates.csv")
    parser.add_argument("--as-of", default=None, help="Screen a specific date, e.g. 2026-05-25. Defaults to latest candidate date.")
    parser.add_argument("--recent-days", type=int, default=1, help="Keep candidates from the latest N candidate dates.")
    parser.add_argument(
        "--signal-count-days",
        type=int,
        default=SCREEN_SETTINGS["signal_count_days"],
        help="Use the latest N candidate dates for repeat-signal counts.",
    )
    parser.add_argument(
        "--signal-count-min-score",
        type=float,
        default=SCREEN_SETTINGS["signal_count_min_score"],
        help="Only count repeat signals whose final_score is at least this value.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initial momentum NN screener")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_caches = subparsers.add_parser("merge-caches", help="Merge multiple OHLCV cache CSV files")
    merge_caches.add_argument("input", nargs="+")
    merge_caches.add_argument("--output", required=True)
    merge_caches.set_defaults(func=merge_caches_command)

    build_listed_stocks = subparsers.add_parser("build-listed-stocks", help="Build a common-stock ticker CSV from a JPX issues CSV")
    build_listed_stocks.add_argument("input")
    build_listed_stocks.add_argument("--output", default="config/listed_stocks.csv")
    build_listed_stocks.add_argument("--encoding", default="cp932")
    build_listed_stocks.add_argument("--market", action="append", help="Market segment to include. Defaults to Prime/Standard/Growth in Japanese.")
    build_listed_stocks.set_defaults(func=build_listed_stocks_command)

    train = subparsers.add_parser("train", help="Download/build events and train the NN")
    add_data_args(train)
    add_train_args(train)
    train.set_defaults(func=train_command)

    rolling_eval = subparsers.add_parser("rolling-eval", help="Train/evaluate repeated time-based folds")
    add_data_args(rolling_eval)
    add_train_args(rolling_eval)
    rolling_eval.add_argument(
        "--fold",
        action="append",
        type=parse_rolling_fold,
        help="Evaluation fold as NAME,TRAIN_END,VALID_END. Defaults to valid_2023/2024/2025.",
    )
    rolling_eval.add_argument("--output", default="outputs/rolling_evaluation.csv")
    rolling_eval.set_defaults(func=rolling_eval_command)

    refresh_data = subparsers.add_parser("refresh-data", help="Refresh OHLCV cache without retraining")
    add_data_args(refresh_data)
    refresh_data.set_defaults(func=refresh_data_command)

    inspect_gate = subparsers.add_parser("inspect-gate", help="Evaluate gate breadth without training the NN")
    add_data_args(inspect_gate)
    inspect_gate.add_argument("--metrics-path", default="outputs/gate_metrics.json")
    inspect_gate.set_defaults(func=inspect_gate_command)

    screen = subparsers.add_parser("screen", help="Score latest initial momentum candidates")
    add_data_args(screen)
    add_screen_args(screen)
    screen.set_defaults(func=screen_command)

    inspect_symbols = subparsers.add_parser("inspect-symbols", help="Inspect gate/model history for specific symbols")
    add_data_args(inspect_symbols)
    inspect_symbols.add_argument("code", nargs="+", help="Code such as 186A, 6976, or 6996.T")
    inspect_symbols.add_argument("--model-path", default="models/momentum_nn.pt")
    inspect_symbols.add_argument("--output", default="outputs/symbol_inspection.csv")
    inspect_symbols.set_defaults(func=inspect_symbols_command)

    run = subparsers.add_parser("run", help="Train and then screen in one command")
    add_data_args(run)
    add_train_args(run)
    add_screen_args(run, include_model_path=False)
    run.set_defaults(func=run_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
