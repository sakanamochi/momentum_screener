from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "outputs" / "candidates_current.csv"
RECENT_CANDIDATES_PATH = ROOT / "outputs" / "candidates_recent.csv"
LISTED_STOCKS_PATH = ROOT / "config" / "listed_stocks.csv"


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 127 else 1 for char in text)


def pad(text: object, width: int, align: str = "left") -> str:
    value = "" if pd.isna(text) else str(text)
    padding = max(width - display_width(value), 0)
    if align == "right":
        return " " * padding + value
    return value + " " * padding


def print_table(rows: list[dict[str, str]], headers: dict[str, str], align_right: set[str] | None = None) -> None:
    align_right = align_right or set()
    widths = {
        key: max(display_width(label), *(display_width(row[key]) for row in rows))
        for key, label in headers.items()
    }
    print("  ".join(pad(label, widths[key], "right" if key in align_right else "left") for key, label in headers.items()))
    print("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        print(
            "  ".join(
                pad(row[key], widths[key], "right" if key in align_right else "left")
                for key in headers
            )
        )


def load_listed() -> pd.DataFrame:
    listed = pd.read_csv(LISTED_STOCKS_PATH, dtype={"code": str}, encoding="utf-8-sig")
    listed["code"] = listed["code"].astype(str).str.strip()
    listed["yf_code"] = listed["code"].where(listed["code"].str.endswith(".T"), listed["code"] + ".T")
    name_column = "銘柄名称" if "銘柄名称" in listed.columns else listed.columns[1]
    return listed[["yf_code", name_column]].rename(columns={name_column: "name"})


def with_names(frame: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(listed, left_on="code", right_on="yf_code", how="left")
    merged["name"] = merged["name"].fillna("")
    return merged


def load_recent_summary() -> pd.DataFrame:
    if not RECENT_CANDIDATES_PATH.exists():
        return pd.DataFrame()
    recent = pd.read_csv(RECENT_CANDIDATES_PATH, dtype={"code": str})
    if recent.empty:
        return pd.DataFrame()

    if "score_recent_signal_count" in recent.columns:
        count_column = "score_recent_signal_count"
    elif "raw_recent_signal_count" in recent.columns:
        count_column = "raw_recent_signal_count"
    else:
        count_column = "recent_signal_count"

    aggregations = {count_column: "max"}
    if "first_score_signal_date" in recent.columns:
        aggregations["first_score_signal_date"] = "min"
    for column in ["first_score_signal_close", "return_since_first_score_signal"]:
        if column in recent.columns:
            aggregations[column] = "first"

    summary = recent.sort_values(["code", "date"]).groupby("code", as_index=False).agg(aggregations)
    summary = summary.rename(columns={count_column: "recent_signal_count"})
    return summary


def print_latest(candidates: pd.DataFrame, listed: pd.DataFrame, recent_summary: pd.DataFrame) -> None:
    merged = with_names(candidates, listed)
    if not recent_summary.empty:
        merged = merged.merge(recent_summary, on="code", how="left", suffixes=("", "_recent"))
        if "recent_signal_count_recent" in merged.columns:
            merged["recent_signal_count"] = merged["recent_signal_count_recent"]
        for column in ["first_score_signal_date", "first_score_signal_close", "return_since_first_score_signal"]:
            recent_column = f"{column}_recent"
            if recent_column in merged.columns:
                merged[column] = merged[recent_column]
        merged["recent_signal_count"] = merged["recent_signal_count"].fillna(0).astype(int)
    elif "score_recent_signal_count" in merged.columns:
        merged["recent_signal_count"] = merged["score_recent_signal_count"].fillna(0).astype(int)
    elif "raw_recent_signal_count" in merged.columns:
        merged["recent_signal_count"] = merged["raw_recent_signal_count"].fillna(1).astype(int)
    else:
        merged["recent_signal_count"] = 1
    for column in ["first_score_signal_date", "return_since_first_score_signal"]:
        if column not in merged.columns:
            merged[column] = pd.NA
    output = merged[
        [
            "code",
            "name",
            "close",
            "recent_signal_count",
            "first_score_signal_date",
            "return_since_first_score_signal",
            "final_score",
        ]
    ].copy()
    output["close"] = output["close"].map(lambda value: f"{float(value):,.1f}")
    output["recent_signal_count"] = output["recent_signal_count"].map(lambda value: f"{int(value)}")
    output["first_score_signal_date"] = output.apply(
        lambda row: "" if int(row["recent_signal_count"]) <= 1 or pd.isna(row["first_score_signal_date"]) else str(row["first_score_signal_date"])[:10],
        axis=1,
    )
    output["return_since_first_score_signal"] = output.apply(
        lambda row: ""
        if int(row["recent_signal_count"]) <= 1 or pd.isna(row["return_since_first_score_signal"])
        else f"{float(row['return_since_first_score_signal']):+.1%}",
        axis=1,
    )
    output["final_score"] = output["final_score"].map(lambda value: f"{float(value):.4f}")

    rows = output.head(30).to_dict("records")
    headers = {
        "code": "銘柄コード",
        "name": "銘柄名",
        "close": "現在株価",
        "recent_signal_count": "回数",
        "first_score_signal_date": "初回日",
        "return_since_first_score_signal": "初回後騰落",
        "final_score": "最終スコア",
    }

    print(f"候補日: {merged['date'].iloc[0]}  表示: {len(rows)} / 全{len(output)}件")
    print_table(
        rows,
        headers,
        align_right={"close", "recent_signal_count", "return_since_first_score_signal", "final_score"},
    )


def print_recent(listed: pd.DataFrame) -> None:
    if not RECENT_CANDIDATES_PATH.exists():
        return
    recent = pd.read_csv(RECENT_CANDIDATES_PATH, dtype={"code": str})
    if recent.empty or "return_since_candidate" not in recent.columns:
        return
    latest_date = recent["date"].max()
    recent = recent[recent["date"] < latest_date].copy()
    if recent.empty:
        return
    if "score_recent_signal_count" in recent.columns:
        recent["recent_signal_count"] = recent["score_recent_signal_count"].fillna(
            recent["recent_signal_count"]
        )
    elif "score_signal_count_since_candidate" in recent.columns:
        recent["recent_signal_count"] = recent["score_signal_count_since_candidate"].fillna(
            recent["recent_signal_count"]
        )
    elif "raw_signal_count_since_candidate" in recent.columns:
        recent["recent_signal_count"] = recent["raw_signal_count_since_candidate"].fillna(
            recent["recent_signal_count"]
        )
    merged = with_names(recent, listed)
    if "signal_still_active" in merged.columns:
        merged = merged.sort_values(
            ["date", "signal_still_active", "recent_signal_count", "final_score"],
            ascending=[False, False, False, False],
        )
    else:
        merged = merged.sort_values(["date", "final_score"], ascending=[False, False])
    merged = merged.groupby("date", group_keys=False).head(8)

    output = merged[
        ["date", "code", "name", "close", "latest_close", "return_since_candidate", "recent_signal_count", "final_score"]
    ].copy()
    output["date"] = output["date"].astype(str)
    output["close"] = output["close"].map(lambda value: f"{float(value):,.1f}")
    output["latest_close"] = output["latest_close"].map(lambda value: f"{float(value):,.1f}")
    output["return_since_candidate"] = output["return_since_candidate"].map(lambda value: f"{float(value):+.1%}")
    output["recent_signal_count"] = output["recent_signal_count"].map(lambda value: f"{int(value)}")
    output["final_score"] = output["final_score"].map(lambda value: f"{float(value):.4f}")

    rows = output.head(40).to_dict("records")
    headers = {
        "date": "候補日",
        "code": "銘柄コード",
        "name": "銘柄名",
        "close": "候補日終値",
        "latest_close": "現在株価",
        "return_since_candidate": "候補後騰落",
        "recent_signal_count": "回数",
        "final_score": "スコア",
    }

    print()
    print(f"過去5候補日の履歴: 各日上位8件 / 全履歴{len(recent)}件")
    print_table(rows, headers, align_right={"close", "latest_close", "return_since_candidate", "recent_signal_count", "final_score"})


def main() -> None:
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"{CANDIDATES_PATH} was not found.")

    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"code": str})
    if candidates.empty:
        print("No candidates found.")
        return

    listed = load_listed()
    recent_summary = load_recent_summary()
    print_latest(candidates, listed, recent_summary)
    print_recent(listed)

    print()
    print(f"CSV: {CANDIDATES_PATH}")
    if RECENT_CANDIDATES_PATH.exists():
        print(f"履歴CSV: {RECENT_CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
