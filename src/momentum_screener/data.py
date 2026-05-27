from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume"]
MIN_LATEST_DATE_COVERAGE = 0.8
ACCEPT_TODAY_AFTER_HOUR_JST = 16


def _normalize_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if text.endswith(".T"):
        return text
    if len(text) == 4 and text.isalnum():
        return f"{text}.T"
    return text


def read_ticker_csv(
    path: str | Path,
    code_column: str = "code",
    market_column: str | None = None,
    include_markets: list[str] | None = None,
    product_column: str | None = None,
    include_products: list[str] | None = None,
    exclude_products: list[str] | None = None,
) -> list[str]:
    df = pd.read_csv(path, dtype=str)
    if code_column not in df.columns:
        raise ValueError(f"Ticker CSV is missing code column: {code_column}")

    filtered = df.copy()
    if market_column and include_markets:
        if market_column not in filtered.columns:
            raise ValueError(f"Ticker CSV is missing market column: {market_column}")
        pattern = "|".join(include_markets)
        filtered = filtered[filtered[market_column].fillna("").str.contains(pattern, case=False, regex=True)]

    if product_column:
        if product_column not in filtered.columns:
            raise ValueError(f"Ticker CSV is missing product column: {product_column}")
        product = filtered[product_column].fillna("")
        if include_products:
            filtered = filtered[product.str.contains("|".join(include_products), case=False, regex=True)]
            product = filtered[product_column].fillna("")
        if exclude_products:
            filtered = filtered[~product.str.contains("|".join(exclude_products), case=False, regex=True)]

    symbols = [_normalize_symbol(value) for value in filtered[code_column]]
    return [symbol for symbol in symbols if symbol]


def read_tickers(
    path: str | Path | None,
    tickers: list[str] | None,
    ticker_csv: str | Path | None = None,
    ticker_csv_code_column: str = "code",
    ticker_csv_market_column: str | None = None,
    ticker_csv_include_markets: list[str] | None = None,
    ticker_csv_product_column: str | None = None,
    ticker_csv_include_products: list[str] | None = None,
    ticker_csv_exclude_products: list[str] | None = None,
) -> list[str]:
    symbols: list[str] = []
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                symbols.append(item)
    if tickers:
        symbols.extend(tickers)
    if ticker_csv:
        symbols.extend(
            read_ticker_csv(
                ticker_csv,
                code_column=ticker_csv_code_column,
                market_column=ticker_csv_market_column,
                include_markets=ticker_csv_include_markets,
                product_column=ticker_csv_product_column,
                include_products=ticker_csv_include_products,
                exclude_products=ticker_csv_exclude_products,
            )
        )
    return sorted(dict.fromkeys(symbols))


def normalize_yfinance_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["code"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[REQUIRED_COLUMNS].dropna(subset=["open", "high", "low", "close"])


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start : batch_start + batch_size]
        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            symbols_in_raw = raw.columns.get_level_values(0).unique()
            for symbol in batch:
                if symbol in symbols_in_raw:
                    frames.append(normalize_yfinance_frame(raw[symbol], symbol))
        elif len(batch) == 1:
            frames.append(normalize_yfinance_frame(raw, batch[0]))

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def drop_intraday_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if now.hour >= ACCEPT_TODAY_AFTER_HOUR_JST:
        return df
    today = pd.Timestamp(now.date())
    return df[df["date"] < today].copy()


def load_or_download_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None,
    cache_path: str | Path,
    refresh: bool = False,
    batch_size: int = 100,
) -> pd.DataFrame:
    path = Path(cache_path)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Cached OHLCV is missing columns: {sorted(missing)}")
        df = drop_intraday_today(df[REQUIRED_COLUMNS])
        return df.sort_values(["code", "date"]).reset_index(drop=True)

    df = download_ohlcv(symbols, start=start, end=end, batch_size=batch_size)
    df = drop_intraday_today(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def refresh_ohlcv_cache(
    symbols: list[str],
    start: str,
    end: str | None,
    cache_path: str | Path,
    batch_size: int = 100,
) -> pd.DataFrame:
    path = Path(cache_path)
    existing = pd.DataFrame(columns=REQUIRED_COLUMNS)
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"])
        missing = set(REQUIRED_COLUMNS) - set(existing.columns)
        if missing:
            raise ValueError(f"Cached OHLCV is missing columns: {sorted(missing)}")
        existing = existing[REQUIRED_COLUMNS]
        if not existing.empty:
            max_date = existing["date"].max()
            start = max(max_date - pd.Timedelta(days=10), pd.Timestamp(start)).strftime("%Y-%m-%d")

    latest = download_ohlcv(symbols, start=start, end=end, batch_size=batch_size)
    if latest.empty:
        existing = drop_intraday_today(existing)
        if existing.empty:
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_csv(path, index=False)
        return existing

    merged = pd.concat([existing, latest], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last")
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    merged = drop_intraday_today(merged)
    expected_symbols = len(set(symbols))
    if expected_symbols > 0:
        while not merged.empty:
            latest_date = merged["date"].max()
            latest_count = int((merged["date"] == latest_date).sum())
            if latest_count >= expected_symbols * MIN_LATEST_DATE_COVERAGE:
                break
            merged = merged[merged["date"] < latest_date].copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged
