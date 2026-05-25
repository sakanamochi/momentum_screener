from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "outputs" / "candidates_current.csv"
LISTED_STOCKS_PATH = ROOT / "config" / "listed_stocks.csv"


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 127 else 1 for char in text)


def pad(text: object, width: int, align: str = "left") -> str:
    value = "" if pd.isna(text) else str(text)
    padding = max(width - display_width(value), 0)
    if align == "right":
        return " " * padding + value
    return value + " " * padding


def main() -> None:
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"{CANDIDATES_PATH} was not found.")

    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"code": str})
    if candidates.empty:
        print("No candidates found.")
        return

    listed = pd.read_csv(LISTED_STOCKS_PATH, dtype={"code": str}, encoding="utf-8-sig")
    listed["code"] = listed["code"].astype(str).str.strip()
    listed["yf_code"] = listed["code"].where(listed["code"].str.endswith(".T"), listed["code"] + ".T")

    name_column = "銘柄名称" if "銘柄名称" in listed.columns else listed.columns[1]
    merged = candidates.merge(listed[["yf_code", name_column]], left_on="code", right_on="yf_code", how="left")
    merged = merged.rename(columns={name_column: "name"})

    output = merged[["code", "name", "close", "final_score"]].copy()
    output["name"] = output["name"].fillna("")
    output["close"] = output["close"].map(lambda value: f"{float(value):,.1f}")
    output["final_score"] = output["final_score"].map(lambda value: f"{float(value):.4f}")

    rows = output.head(30).to_dict("records")
    headers = {"code": "銘柄コード", "name": "銘柄名", "close": "現在株価", "final_score": "最終スコア"}
    widths = {
        key: max(display_width(label), *(display_width(row[key]) for row in rows))
        for key, label in headers.items()
    }

    print(f"候補日: {merged['date'].iloc[0]}  表示: {len(rows)} / 全{len(output)}件")
    print(
        "  ".join(
            [
                pad(headers["code"], widths["code"]),
                pad(headers["name"], widths["name"]),
                pad(headers["close"], widths["close"], "right"),
                pad(headers["final_score"], widths["final_score"], "right"),
            ]
        )
    )
    print(
        "  ".join(
            [
                "-" * widths["code"],
                "-" * widths["name"],
                "-" * widths["close"],
                "-" * widths["final_score"],
            ]
        )
    )
    for row in rows:
        print(
            "  ".join(
                [
                    pad(row["code"], widths["code"]),
                    pad(row["name"], widths["name"]),
                    pad(row["close"], widths["close"], "right"),
                    pad(row["final_score"], widths["final_score"], "right"),
                ]
            )
        )

    print()
    print(f"CSV: {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()

