from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_screener.features import FEATURE_COLUMNS, add_features, add_initial_momentum_gate
from momentum_screener.model import DEFAULT_RISK_ADJUSTMENT, RISK_ADJUSTMENT_PRESETS, add_risk_adjusted_score, load_artifacts, predict_proba
from momentum_screener.settings import GATE_SETTINGS, LABEL_SETTINGS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "ohlcv_current.csv"
DEFAULT_MODEL = ROOT / "models" / "momentum_nn_production.pt"
DEFAULT_LISTED = ROOT / "config" / "listed_stocks.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "symbol_history"


def normalize_code(code: str) -> str:
    text = code.strip().upper()
    return text if text.endswith(".T") else f"{text}.T"


def display_code(code: str) -> str:
    return code[:-2] if code.endswith(".T") else code


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 127 else 1 for char in text)


def pad(text: object, width: int, align: str = "left") -> str:
    value = "" if pd.isna(text) else str(text)
    padding = max(width - display_width(value), 0)
    return (" " * padding + value) if align == "right" else (value + " " * padding)


def print_table(rows: list[dict[str, str]], headers: dict[str, str], align_right: set[str]) -> None:
    if not rows:
        return
    widths = {key: max(display_width(label), *(display_width(row[key]) for row in rows)) for key, label in headers.items()}
    print("  ".join(pad(label, widths[key], "right" if key in align_right else "left") for key, label in headers.items()))
    print("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        print("  ".join(pad(row[key], widths[key], "right" if key in align_right else "left") for key in headers))


def load_name(symbol: str, listed_path: Path) -> str:
    if not listed_path.exists():
        return ""
    listed = pd.read_csv(listed_path, dtype={"code": str}, encoding="utf-8-sig")
    listed["code"] = listed["code"].astype(str).str.strip()
    listed["yf_code"] = listed["code"].where(listed["code"].str.endswith(".T"), listed["code"] + ".T")
    matches = listed[listed["yf_code"] == symbol]
    if matches.empty:
        return ""
    name_columns = [column for column in listed.columns if column not in {"code", "yf_code"}]
    return str(matches.iloc[0][name_columns[0]]) if name_columns else ""


def format_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:+.1%}" if np.isfinite(number) else ""


def candidate_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    symbol = normalize_code(args.code)
    ohlcv = pd.read_csv(args.cache, parse_dates=["date"], dtype={"code": str})
    symbol_ohlcv = ohlcv[ohlcv["code"] == symbol].copy()
    if symbol_ohlcv.empty:
        raise ValueError(f"No OHLCV rows found for {symbol}.")

    features = add_features(symbol_ohlcv)
    gated = add_initial_momentum_gate(
        features,
        cooldown_days=args.cooldown_days,
        min_turnover_5d=args.gate_min_turnover_5d,
        min_ret_5d=args.gate_min_ret_5d,
        min_turnover_ratio_1d_20d=args.gate_min_turnover_ratio_1d_20d,
        min_turnover_ratio_5d_20d=args.gate_min_turnover_ratio_5d_20d,
        min_close_ma25_ratio=args.gate_min_close_ma25_ratio,
    ).sort_values("date").reset_index(drop=True)

    candidates = gated[gated["raw_initial_momentum"]].copy()
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    if candidates.empty:
        return pd.DataFrame(), symbol

    model, scaler, feature_columns = load_artifacts(args.model_path)
    candidates["follow_through_prob"] = predict_proba(model, scaler, feature_columns, candidates)
    candidates = add_risk_adjusted_score(candidates, preset=args.risk_adjustment)
    candidates = candidates[candidates["final_score"] >= args.score_threshold].copy()
    if candidates.empty:
        return pd.DataFrame(), symbol

    positions = pd.Series(gated.index.to_numpy(), index=gated["date"])
    latest = gated.iloc[-1]
    rows: list[dict[str, object]] = []
    for _, candidate in candidates.sort_values("date").iterrows():
        position = int(positions[candidate["date"]])
        entry_position = position + 1
        result_position = min(position + args.horizon, len(gated) - 1)
        result = gated.iloc[result_position]
        window = gated.iloc[entry_position : result_position + 1]

        entry_date = ""
        entry_open = np.nan
        entry_return = np.nan
        max_return = np.nan
        min_return = np.nan
        if entry_position < len(gated) and not window.empty:
            entry = gated.iloc[entry_position]
            entry_date = entry["date"].date().isoformat()
            entry_open = float(entry["open"])
            if np.isfinite(entry_open) and entry_open > 0:
                entry_return = float(result["close"] / entry_open - 1.0)
                max_return = float(window["high"].max() / entry_open - 1.0)
                min_return = float(window["low"].min() / entry_open - 1.0)

        rows.append(
            {
                "date": candidate["date"].date().isoformat(),
                "code": display_code(symbol),
                "candidate_close": float(candidate["close"]),
                "score": float(candidate["final_score"]),
                "ret_5d": float(candidate["ret_5d"]),
                "turnover_5d_avg": float(candidate["turnover_5d_avg"]),
                "entry_date": entry_date,
                "entry_open": entry_open,
                "result_date": result["date"].date().isoformat(),
                "elapsed_trading_days": result_position - position,
                "status": "20d" if result_position - position >= args.horizon else "partial",
                "result_close": float(result["close"]),
                "return_from_candidate_close": float(result["close"] / candidate["close"] - 1.0),
                "return_from_entry_open": entry_return,
                "max_return_from_entry_open": max_return,
                "min_return_from_entry_open": min_return,
                "latest_date": latest["date"].date().isoformat(),
                "latest_close": float(latest["close"]),
            }
        )
    return pd.DataFrame(rows), symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show historical candidate dates for one symbol.")
    parser.add_argument("code", help="Code such as 285A, 6976, or 6996.T")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--listed-path", type=Path, default=DEFAULT_LISTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--risk-adjustment", choices=list(RISK_ADJUSTMENT_PRESETS), default=DEFAULT_RISK_ADJUSTMENT)
    parser.add_argument("--horizon", type=int, default=LABEL_SETTINGS["horizon"])
    parser.add_argument("--cooldown-days", type=int, default=GATE_SETTINGS["cooldown_days"])
    parser.add_argument("--gate-min-turnover-5d", type=float, default=GATE_SETTINGS["min_turnover_5d"])
    parser.add_argument("--gate-min-ret-5d", type=float, default=GATE_SETTINGS["min_ret_5d"])
    parser.add_argument("--gate-min-turnover-ratio-1d-20d", type=float, default=GATE_SETTINGS["min_turnover_ratio_1d_20d"])
    parser.add_argument("--gate-min-turnover-ratio-5d-20d", type=float, default=GATE_SETTINGS["min_turnover_ratio_5d_20d"])
    parser.add_argument("--gate-min-close-ma25-ratio", type=float, default=GATE_SETTINGS["min_close_ma25_ratio"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history, symbol = candidate_rows(args)
    name = load_name(symbol, args.listed_path)
    title = f"{display_code(symbol)} {name}".strip()

    print(f"銘柄: {title}")
    print(f"条件: raw初動ゲート + final_score >= {args.score_threshold:.2f}")
    print(f"成績: 候補日の翌営業日始値から{args.horizon}営業日後の終値まで。未到達は直近日まで。")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{display_code(symbol)}_history.csv"
    history.to_csv(output_path, index=False, encoding="utf-8-sig")

    if history.empty:
        print("該当する過去候補日はありません。")
        print(f"output={output_path}")
        return

    rows = []
    for _, row in history.sort_values("date", ascending=False).iterrows():
        rows.append(
            {
                "date": row["date"],
                "score": f"{row['score']:.3f}",
                "entry_date": row["entry_date"],
                "entry_open": f"{row['entry_open']:,.1f}" if np.isfinite(row["entry_open"]) else "",
                "result_date": row["result_date"],
                "days": str(int(row["elapsed_trading_days"])),
                "status": row["status"],
                "return": format_pct(row["return_from_entry_open"]),
                "close_return": format_pct(row["return_from_candidate_close"]),
                "max": format_pct(row["max_return_from_entry_open"]),
                "min": format_pct(row["min_return_from_entry_open"]),
            }
        )

    print_table(
        rows,
        {
            "date": "候補日",
            "score": "スコア",
            "entry_date": "翌営業日",
            "entry_open": "翌始値",
            "result_date": "判定日",
            "days": "経過",
            "status": "区分",
            "return": "騰落",
            "close_return": "候補終値比",
            "max": "最大",
            "min": "最小",
        },
        align_right={"score", "entry_open", "days", "return", "close_return", "max", "min"},
    )
    print()
    print(f"signals={len(history)} output={output_path}")


if __name__ == "__main__":
    main()
