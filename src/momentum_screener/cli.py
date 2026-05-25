from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_screener.data import load_or_download_ohlcv, read_tickers
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
    return add_labels(gated, horizon=args.horizon, threshold=args.target_threshold)


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


def write_metrics(metrics: dict[str, float], path: str | Path) -> None:
    clean = {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in metrics.items()}
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
    events = make_event_dataset(labelled, require_label=True)
    if len(events) < args.min_events:
        raise ValueError(
            f"Only {len(events)} labelled events were created. "
            f"Add more tickers/history or lower --min-events for a smoke test."
        )

    result, metrics = train_model(
        events=events,
        feature_columns=FEATURE_COLUMNS,
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
    save_artifacts(result, args.model_path)
    write_metrics(metrics, args.metrics_path)
    print(f"trained_events={len(events)} model={args.model_path} metrics={args.metrics_path}")


def inspect_gate_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    events = make_event_dataset(labelled, require_label=True)
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
    }
    write_metrics(metrics, args.metrics_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics={args.metrics_path}")


def screen_command(args: argparse.Namespace) -> None:
    labelled = build_frame(args)
    candidates = make_event_dataset(labelled, require_label=False)
    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
        candidates = candidates[candidates["date"] == as_of].copy()
    else:
        latest_date = candidates["date"].max()
        candidates = candidates[candidates["date"] == latest_date].copy()

    model, scaler, feature_columns = load_artifacts(args.model_path)
    candidates["follow_through_prob"] = predict_proba(model, scaler, feature_columns, candidates)
    candidates["final_score"] = candidates["follow_through_prob"]
    candidates["reason"] = candidates.apply(make_reason, axis=1) if not candidates.empty else []
    candidates = candidates.sort_values("final_score", ascending=False)

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

    events = make_event_dataset(rows, require_label=False)
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
    parser.add_argument("--cooldown-days", type=int, default=20)
    parser.add_argument("--gate-min-turnover-5d", type=float, default=100_000_000)
    parser.add_argument("--gate-min-ret-5d", type=float, default=0.0)
    parser.add_argument("--gate-min-turnover-ratio-1d-20d", type=float, default=1.2)
    parser.add_argument("--gate-min-turnover-ratio-5d-20d", type=float, default=1.15)
    parser.add_argument("--gate-min-close-ma25-ratio", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--target-threshold", type=float, default=0.10)


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
