from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_screener.features import FEATURE_COLUMNS, add_features, add_initial_momentum_gate
from momentum_screener.model import load_artifacts, predict_proba
from momentum_screener.settings import GATE_SETTINGS, LABEL_SETTINGS, SCREEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "ohlcv_current.csv"
DEFAULT_MODEL = ROOT / "models" / "momentum_nn_production.pt"
DEFAULT_LISTED = ROOT / "config" / "listed_stocks.csv"
DEFAULT_RANK_HISTORY = ROOT / "outputs" / "candidates_rank_history.csv"
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
    if align == "right":
        return " " * padding + value
    return value + " " * padding


def print_table(rows: list[dict[str, str]], headers: dict[str, str], align_right: set[str]) -> None:
    widths = {
        key: max(display_width(label), *(display_width(row[key]) for row in rows))
        for key, label in headers.items()
    }
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
    name_candidates = [column for column in listed.columns if column not in {"code", "yf_code"}]
    return str(matches.iloc[0][name_candidates[0]]) if name_candidates else ""


def format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:+.1%}"


def load_rank_history(args: argparse.Namespace, symbol: str) -> pd.DataFrame | None:
    if args.no_rank_cache or not args.rank_history.exists():
        return None
    history = pd.read_csv(args.rank_history, parse_dates=["date"], dtype={"code": str})
    required = {"date", "code", "close", "final_score", "score_rank_in_date"}
    if not required.issubset(history.columns):
        return None
    return history[(history["code"] == symbol) & (history["score_rank_in_date"] <= args.top_n)].copy()


def build_history(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    symbol = normalize_code(args.code)
    ohlcv = pd.read_csv(args.cache, parse_dates=["date"], dtype={"code": str})
    if ohlcv[ohlcv["code"] == symbol].empty:
        raise ValueError(f"No OHLCV rows found for {symbol}. Check data cache or code.")

    candidates = load_rank_history(args, symbol)
    if candidates is None:
        features = add_features(ohlcv)
        gated = add_initial_momentum_gate(
            features,
            cooldown_days=args.cooldown_days,
            min_turnover_5d=args.gate_min_turnover_5d,
            min_ret_5d=args.gate_min_ret_5d,
            min_turnover_ratio_1d_20d=args.gate_min_turnover_ratio_1d_20d,
            min_turnover_ratio_5d_20d=args.gate_min_turnover_ratio_5d_20d,
            min_close_ma25_ratio=args.gate_min_close_ma25_ratio,
        ).sort_values(["code", "date"]).reset_index(drop=True)

        candidates = gated[gated["raw_initial_momentum"]].copy()
        candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
        if candidates.empty:
            return pd.DataFrame(), symbol

        model, scaler, feature_columns = load_artifacts(args.model_path)
        candidates["final_score"] = predict_proba(model, scaler, feature_columns, candidates)
        candidates["score_rank_in_date"] = (
            candidates.groupby("date")["final_score"].rank(method="first", ascending=False).astype(int)
        )
        candidates = candidates[
            (candidates["code"] == symbol) & (candidates["score_rank_in_date"] <= args.top_n)
        ].copy()
    if candidates.empty:
        return pd.DataFrame(), symbol

    symbol_gated = add_features(ohlcv[ohlcv["code"] == symbol].copy()).sort_values("date").reset_index(drop=True)
    positions = pd.Series(symbol_gated.index.to_numpy(), index=symbol_gated["date"])
    rows: list[dict[str, object]] = []
    latest = symbol_gated.iloc[-1]
    for _, candidate in candidates.sort_values("date").iterrows():
        position = int(positions[candidate["date"]])
        result_position = min(position + args.horizon, len(symbol_gated) - 1)
        entry_position = position + 1
        result = symbol_gated.iloc[result_position]
        elapsed_days = result_position - position
        complete = elapsed_days >= args.horizon

        entry_date = ""
        entry_price = np.nan
        entry_to_result = np.nan
        future_max_ret = np.nan
        future_min_ret = np.nan
        if entry_position < len(symbol_gated):
            entry_row = symbol_gated.iloc[entry_position]
            entry_date = entry_row["date"].date().isoformat()
            entry_price = float(entry_row["open"])
            future_window = symbol_gated.iloc[entry_position : result_position + 1]
            if np.isfinite(entry_price) and entry_price > 0 and not future_window.empty:
                entry_to_result = float(result["close"] / entry_price - 1.0)
                future_max_ret = float(future_window["high"].max() / entry_price - 1.0)
                future_min_ret = float(future_window["low"].min() / entry_price - 1.0)

        rows.append(
            {
                "date": candidate["date"].date().isoformat(),
                "code": display_code(symbol),
                "candidate_close": float(candidate["close"]),
                "score": float(candidate["final_score"]),
                "rank": int(candidate["score_rank_in_date"]),
                "ret_5d": float(candidate["ret_5d"]),
                "turnover_5d_avg": float(candidate["turnover_5d_avg"]),
                "result_date": result["date"].date().isoformat(),
                "elapsed_trading_days": elapsed_days,
                "status": "20d" if complete else "partial",
                "result_close": float(result["close"]),
                "return_from_candidate_close": float(result["close"] / candidate["close"] - 1.0),
                "entry_date": entry_date,
                "entry_open": entry_price,
                "return_from_entry_open": entry_to_result,
                "max_return_from_entry_open": future_max_ret,
                "min_return_from_entry_open": future_min_ret,
                "latest_date": latest["date"].date().isoformat(),
                "latest_close": float(latest["close"]),
            }
        )

    return pd.DataFrame(rows), symbol


def main() -> None:
    parser = argparse.ArgumentParser(description="Show historical high-score candidate dates for one symbol.")
    parser.add_argument("code", help="Code such as 285A, 6976, or 6996.T")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--listed-path", type=Path, default=DEFAULT_LISTED)
    parser.add_argument("--rank-history", type=Path, default=DEFAULT_RANK_HISTORY)
    parser.add_argument("--no-rank-cache", action="store_true", help="Ignore cached rank history and recompute ranks from OHLCV/model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=SCREEN_SETTINGS["signal_count_top_n"])
    parser.add_argument("--horizon", type=int, default=LABEL_SETTINGS["horizon"])
    parser.add_argument("--cooldown-days", type=int, default=GATE_SETTINGS["cooldown_days"])
    parser.add_argument("--gate-min-turnover-5d", type=float, default=GATE_SETTINGS["min_turnover_5d"])
    parser.add_argument("--gate-min-ret-5d", type=float, default=GATE_SETTINGS["min_ret_5d"])
    parser.add_argument("--gate-min-turnover-ratio-1d-20d", type=float, default=GATE_SETTINGS["min_turnover_ratio_1d_20d"])
    parser.add_argument("--gate-min-turnover-ratio-5d-20d", type=float, default=GATE_SETTINGS["min_turnover_ratio_5d_20d"])
    parser.add_argument("--gate-min-close-ma25-ratio", type=float, default=GATE_SETTINGS["min_close_ma25_ratio"])
    args = parser.parse_args()

    history, symbol = build_history(args)
    name = load_name(symbol, args.listed_path)
    title = f"{display_code(symbol)} {name}".strip()
    print(f"銘柄: {title}")
    print(f"条件: raw初動ゲート + 候補日別final_score上位{args.top_n}位以内")
    print(f"成績: 候補日の翌営業日始値から{args.horizon}営業日後の終値まで。未到達は直近日まで。")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{display_code(symbol)}_history.csv"
    history.to_csv(output_path, index=False, encoding="utf-8-sig")

    if history.empty:
        print("該当する過去候補日はありません。")
        print(f"output={output_path}")
        return

    display = history.sort_values("date", ascending=False).copy()
    rows = []
    for _, row in display.iterrows():
        rows.append(
            {
                "date": row["date"],
                "rank": f"{int(row['rank'])}",
                "score": f"{row['score']:.3f}",
                "entry_date": row["entry_date"],
                "entry_open": f"{row['entry_open']:,.1f}" if np.isfinite(row["entry_open"]) else "",
                "result_date": row["result_date"],
                "days": f"{int(row['elapsed_trading_days'])}",
                "status": row["status"],
                "return": format_pct(row["return_from_entry_open"]),
                "close_return": format_pct(row["return_from_candidate_close"]),
                "max": format_pct(row["max_return_from_entry_open"]),
                "min": format_pct(row["min_return_from_entry_open"]),
            }
        )
    headers = {
        "date": "候補日",
        "rank": "順位",
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
    }
    print_table(rows, headers, align_right={"rank", "score", "entry_open", "days", "return", "close_return", "max", "min"})
    print()
    print(f"signals={len(history)} output={output_path}")


if __name__ == "__main__":
    main()
