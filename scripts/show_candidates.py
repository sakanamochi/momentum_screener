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
    return (" " * padding + value) if align == "right" else (value + " " * padding)


def print_table(rows: list[dict[str, str]], headers: dict[str, str], align_right: set[str]) -> None:
    if not rows:
        return
    widths = {key: max(display_width(label), *(display_width(row[key]) for row in rows)) for key, label in headers.items()}
    print("  ".join(pad(label, widths[key], "right" if key in align_right else "left") for key, label in headers.items()))
    print("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        print("  ".join(pad(row[key], widths[key], "right" if key in align_right else "left") for key in headers))


def load_listed() -> pd.DataFrame:
    listed = pd.read_csv(LISTED_STOCKS_PATH, dtype={"code": str}, encoding="utf-8-sig")
    listed["code"] = listed["code"].astype(str).str.strip()
    listed["yf_code"] = listed["code"].where(listed["code"].str.endswith(".T"), listed["code"] + ".T")
    name_columns = [column for column in listed.columns if column not in {"code", "yf_code"}]
    name_column = name_columns[0] if name_columns else "code"
    return listed[["yf_code", name_column]].rename(columns={name_column: "name"})


def add_names(frame: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(listed, left_on="code", right_on="yf_code", how="left")
    merged["name"] = merged["name"].fillna("")
    return merged


def top30_count_column(frame: pd.DataFrame) -> str:
    for column in ["top30_recent_signal_count", "score_recent_signal_count", "raw_recent_signal_count", "recent_signal_count"]:
        if column in frame.columns:
            return column
    return "recent_signal_count"


def load_recent_summary() -> pd.DataFrame:
    if not RECENT_CANDIDATES_PATH.exists():
        return pd.DataFrame()
    recent = pd.read_csv(RECENT_CANDIDATES_PATH, dtype={"code": str})
    if recent.empty:
        return pd.DataFrame()

    count_column = top30_count_column(recent)
    aggregations = {count_column: "max"}
    for column in ["first_score_signal_date", "first_score_signal_close", "return_since_first_score_signal"]:
        if column in recent.columns:
            aggregations[column] = "first" if column != "first_score_signal_date" else "min"
    return (
        recent.sort_values(["code", "date"])
        .groupby("code", as_index=False)
        .agg(aggregations)
        .rename(columns={count_column: "recent_signal_count"})
    )


def print_latest(candidates: pd.DataFrame, listed: pd.DataFrame, recent_summary: pd.DataFrame) -> None:
    merged = add_names(candidates, listed)
    if not recent_summary.empty:
        merged = merged.merge(recent_summary, on="code", how="left", suffixes=("", "_recent"))
        if "recent_signal_count_recent" in merged.columns:
            merged["recent_signal_count"] = merged["recent_signal_count_recent"]
        for column in ["first_score_signal_date", "first_score_signal_close", "return_since_first_score_signal"]:
            recent_column = f"{column}_recent"
            if recent_column in merged.columns:
                merged[column] = merged[recent_column]

    count_column = "recent_signal_count" if "recent_signal_count" in merged.columns else top30_count_column(merged)
    merged["recent_signal_count"] = merged[count_column].fillna(0).astype(int).clip(lower=1)
    for column in ["first_score_signal_date", "return_since_first_score_signal"]:
        if column not in merged.columns:
            merged[column] = pd.NA

    output = merged.sort_values("final_score", ascending=False).head(30).copy()
    rows = []
    for _, row in output.iterrows():
        repeated = int(row["recent_signal_count"]) > 1
        rows.append(
            {
                "code": str(row["code"]).replace(".T", ""),
                "name": row["name"],
                "close": f"{float(row['close']):,.1f}",
                "count": str(int(row["recent_signal_count"])),
                "first_date": "" if not repeated or pd.isna(row["first_score_signal_date"]) else str(row["first_score_signal_date"])[:10],
                "first_return": "" if not repeated or pd.isna(row["return_since_first_score_signal"]) else f"{float(row['return_since_first_score_signal']):+.1%}",
                "score": f"{float(row['final_score']):.4f}",
            }
        )

    print(f"候補日: {merged['date'].iloc[0]}  表示: {len(rows)} / 全{len(output)}件")
    print_table(
        rows,
        {
            "code": "コード",
            "name": "銘柄名",
            "close": "現在値",
            "count": "Top30回数",
            "first_date": "初回日",
            "first_return": "初回比",
            "score": "スコア",
        },
        align_right={"close", "count", "first_return", "score"},
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

    count_column = top30_count_column(recent)
    recent["recent_signal_count"] = recent[count_column].fillna(recent.get("recent_signal_count", 0)).astype(int)
    merged = add_names(recent, listed).sort_values(["date", "final_score"], ascending=[False, False])
    output = merged.groupby("date", group_keys=False).head(8).head(40)

    rows = []
    for _, row in output.iterrows():
        rows.append(
            {
                "date": str(row["date"])[:10],
                "code": str(row["code"]).replace(".T", ""),
                "name": row["name"],
                "close": f"{float(row['close']):,.1f}",
                "latest_close": f"{float(row['latest_close']):,.1f}",
                "return": f"{float(row['return_since_candidate']):+.1%}",
                "count": str(int(row["recent_signal_count"])),
                "score": f"{float(row['final_score']):.4f}",
            }
        )

    print()
    print(f"直近候補履歴: 各日上位8件 / 全履歴{len(recent)}件")
    print_table(
        rows,
        {
            "date": "候補日",
            "code": "コード",
            "name": "銘柄名",
            "close": "候補日終値",
            "latest_close": "現在値",
            "return": "候補後騰落",
            "count": "Top30回数",
            "score": "スコア",
        },
        align_right={"close", "latest_close", "return", "count", "score"},
    )


def main() -> None:
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"{CANDIDATES_PATH} was not found.")

    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"code": str})
    if candidates.empty:
        print("No candidates found.")
        return

    listed = load_listed()
    print_latest(candidates, listed, load_recent_summary())
    print_recent(listed)
    print()
    print(f"CSV: {CANDIDATES_PATH}")
    if RECENT_CANDIDATES_PATH.exists():
        print(f"履歴CSV: {RECENT_CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
