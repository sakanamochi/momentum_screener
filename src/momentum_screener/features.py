from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_5d_accel",
    "close_ma25_ratio",
    "close_ma75_ratio",
    "is_20d_high",
    "is_60d_high",
    "volume_ratio_1d_20d",
    "turnover_5d_avg",
    "turnover_ratio_1d_20d",
    "turnover_ratio_5d_20d",
    "log_turnover_5d_avg",
    "close_position_in_range",
    "intraday_range_ratio",
    "upper_shadow_ratio",
    "volatility_20d",
    "share_turnover_5d",
    "float_turnover_5d",
]

OUTPUT_COLUMNS = [
    "date",
    "code",
    "close",
    "ret_5d",
    "ret_20d",
    "ret_5d_accel",
    "turnover_5d_avg",
    "turnover_ratio_1d_20d",
    "turnover_ratio_5d_20d",
    "close_ma25_ratio",
    "is_20d_high",
    "close_position_in_range",
    "share_turnover_5d",
    "float_turnover_5d",
    "follow_through_prob",
    "final_score",
    "reason",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_features(
    ohlcv: pd.DataFrame,
    shares: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    if shares is not None and not shares.empty:
        df = df.merge(shares, on="code", how="left")
    if "shares_outstanding" not in df.columns:
        df["shares_outstanding"] = np.nan
    if "free_float_shares" not in df.columns:
        df["free_float_shares"] = np.nan

    grouped = df.groupby("code", group_keys=False)
    df["ret_1d"] = grouped["close"].pct_change(1)
    df["ret_5d"] = grouped["close"].pct_change(5)
    df["ret_20d"] = grouped["close"].pct_change(20)
    df["ret_5d_accel"] = df["ret_5d"] - df["ret_20d"] / 4.0

    df["ma25"] = grouped["close"].transform(lambda s: s.rolling(25, min_periods=25).mean())
    df["ma75"] = grouped["close"].transform(lambda s: s.rolling(75, min_periods=75).mean())
    df["close_ma25_ratio"] = df["close"] / df["ma25"] - 1.0
    df["close_ma75_ratio"] = df["close"] / df["ma75"] - 1.0

    high_20 = grouped["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    high_60 = grouped["high"].transform(lambda s: s.rolling(60, min_periods=60).max())
    df["is_20d_high"] = (df["high"] >= high_20).astype(float)
    df["is_60d_high"] = (df["high"] >= high_60).astype(float)

    volume_20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["volume_ratio_1d_20d"] = _safe_div(df["volume"], volume_20)
    df["turnover_1d"] = df["close"] * df["volume"]
    df["turnover_5d_avg"] = grouped["turnover_1d"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df["turnover_20d_avg"] = grouped["turnover_1d"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["turnover_ratio_1d_20d"] = _safe_div(df["turnover_1d"], df["turnover_20d_avg"])
    df["turnover_ratio_5d_20d"] = _safe_div(df["turnover_5d_avg"], df["turnover_20d_avg"])
    df["log_turnover_5d_avg"] = np.log1p(df["turnover_5d_avg"])

    day_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_position_in_range"] = ((df["close"] - df["low"]) / day_range).clip(0, 1)
    df["intraday_range_ratio"] = _safe_div(df["high"] - df["low"], df["close"])
    body_top = df[["open", "close"]].max(axis=1)
    df["upper_shadow_ratio"] = ((df["high"] - body_top) / day_range).clip(lower=0)

    df["volatility_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    volume_5d_sum = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    df["share_turnover_5d"] = _safe_div(volume_5d_sum, df["shares_outstanding"])
    df["float_turnover_5d"] = _safe_div(volume_5d_sum, df["free_float_shares"])

    df[["share_turnover_5d", "float_turnover_5d"]] = df[
        ["share_turnover_5d", "float_turnover_5d"]
    ].fillna(0.0)
    return df


def raw_gate_mask(
    df: pd.DataFrame,
    min_turnover_5d: float = 100_000_000,
    min_ret_5d: float = 0.0,
    min_turnover_ratio_1d_20d: float = 1.2,
    min_turnover_ratio_5d_20d: float = 1.15,
    min_close_ma25_ratio: float = 0.0,
) -> pd.Series:
    return (
        (df["turnover_5d_avg"] >= min_turnover_5d)
        & (df["close_ma25_ratio"] >= min_close_ma25_ratio)
        & (df["ret_5d"] > min_ret_5d)
        & (
            (df["turnover_ratio_1d_20d"] >= min_turnover_ratio_1d_20d)
            | (df["turnover_ratio_5d_20d"] >= min_turnover_ratio_5d_20d)
        )
    )


def add_initial_momentum_gate(
    df: pd.DataFrame,
    cooldown_days: int = 20,
    min_turnover_5d: float = 100_000_000,
    min_ret_5d: float = 0.0,
    min_turnover_ratio_1d_20d: float = 1.2,
    min_turnover_ratio_5d_20d: float = 1.15,
    min_close_ma25_ratio: float = 0.0,
) -> pd.DataFrame:
    gated = df.copy()
    gated["raw_initial_momentum"] = raw_gate_mask(
        gated,
        min_turnover_5d=min_turnover_5d,
        min_ret_5d=min_ret_5d,
        min_turnover_ratio_1d_20d=min_turnover_ratio_1d_20d,
        min_turnover_ratio_5d_20d=min_turnover_ratio_5d_20d,
        min_close_ma25_ratio=min_close_ma25_ratio,
    )
    gated["initial_momentum"] = False

    for _, group in gated.groupby("code", sort=False):
        last_position = -10**9
        for position, idx in enumerate(group.index):
            if bool(gated.at[idx, "raw_initial_momentum"]) and position - last_position > cooldown_days:
                gated.at[idx, "initial_momentum"] = True
                last_position = position
    return gated


def add_labels(df: pd.DataFrame, horizon: int = 20, threshold: float = 0.10) -> pd.DataFrame:
    labelled = df.copy()

    def future_max(high: pd.Series) -> pd.Series:
        return high.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1].shift(-1)

    labelled["future_high_20d"] = labelled.groupby("code", group_keys=False)["high"].transform(future_max)
    labelled["future_max_ret_20d"] = labelled["future_high_20d"] / labelled["close"] - 1.0
    labelled["target_20d"] = (labelled["future_max_ret_20d"] >= threshold).astype(float)
    labelled.loc[labelled["future_max_ret_20d"].isna(), "target_20d"] = np.nan
    labelled["sample_weight"] = 1.0 + labelled["future_max_ret_20d"].clip(lower=0, upper=0.30) * 10.0
    return labelled


def make_event_dataset(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    events = df[df["initial_momentum"]].copy()
    if require_label:
        events = events.dropna(subset=["target_20d", "sample_weight"])
    events = events.replace([np.inf, -np.inf], np.nan)
    events = events.dropna(subset=FEATURE_COLUMNS)
    return events.reset_index(drop=True)


def make_reason(row: pd.Series) -> str:
    parts = [
        f"5日{row['ret_5d']:+.1%}",
        f"売買代金5日平均{row['turnover_ratio_5d_20d']:.1f}倍",
        f"25日線{row['close_ma25_ratio']:+.1%}",
    ]
    if row.get("is_20d_high", 0) >= 1:
        parts.append("20日高値更新")
    parts.append(f"高値引け度{row['close_position_in_range']:.2f}")
    return " / ".join(parts)
