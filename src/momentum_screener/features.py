from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
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

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS

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
    "latest_close",
    "return_since_candidate",
    "recent_signal_count",
    "raw_recent_signal_count",
    "raw_signal_count_since_candidate",
    "score_recent_signal_count",
    "score_signal_count_since_candidate",
    "first_score_signal_date",
    "first_score_signal_close",
    "return_since_first_score_signal",
    "signal_still_active",
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


def add_sample_weights(
    labelled: pd.DataFrame,
    mode: str = "future_max_ret",
    cap: float = 0.30,
    scale: float = 10.0,
) -> pd.DataFrame:
    weighted = labelled.copy()
    if mode == "uniform":
        weighted["sample_weight"] = 1.0
        return weighted

    clipped_return = weighted["future_max_ret_20d"].clip(lower=0, upper=cap).fillna(0.0)
    if mode == "future_max_ret":
        weighted["sample_weight"] = 1.0 + clipped_return * scale
        return weighted
    if mode == "target_future_max_ret":
        target = weighted["target_20d"].fillna(0.0).clip(lower=0.0, upper=1.0)
        weighted["sample_weight"] = 1.0 + clipped_return * scale * target
        return weighted
    raise ValueError(f"Unknown sample_weight_mode: {mode}")


def _add_max_return_labels(
    df: pd.DataFrame,
    horizon: int,
    threshold: float,
    sample_weight_mode: str,
    sample_weight_cap: float,
    sample_weight_scale: float,
) -> pd.DataFrame:
    labelled = df.copy()

    def future_max(high: pd.Series) -> pd.Series:
        return high.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1].shift(-1)

    labelled["future_high_20d"] = labelled.groupby("code", group_keys=False)["high"].transform(future_max)
    labelled["entry_price"] = labelled["close"]
    labelled["future_max_ret_20d"] = labelled["future_high_20d"] / labelled["close"] - 1.0
    labelled["future_min_ret_20d"] = np.nan
    labelled["hit_profit_day"] = np.nan
    labelled["hit_stop_day"] = np.nan
    labelled["target_20d"] = (labelled["future_max_ret_20d"] >= threshold).astype(float)
    labelled.loc[labelled["future_max_ret_20d"].isna(), "target_20d"] = np.nan
    return add_sample_weights(
        labelled,
        mode=sample_weight_mode,
        cap=sample_weight_cap,
        scale=sample_weight_scale,
    )


def _add_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    profit_barrier: float,
    stop_barrier: float,
    sample_weight_mode: str,
    sample_weight_cap: float,
    sample_weight_scale: float,
) -> pd.DataFrame:
    labelled = df.copy()
    labelled["entry_price"] = np.nan
    labelled["future_high_20d"] = np.nan
    labelled["future_low_20d"] = np.nan
    labelled["future_max_ret_20d"] = np.nan
    labelled["future_min_ret_20d"] = np.nan
    labelled["hit_profit_day"] = np.nan
    labelled["hit_stop_day"] = np.nan
    labelled["target_20d"] = np.nan

    for _, group in labelled.groupby("code", sort=False):
        idx = group.index.to_numpy()
        opens = group["open"].to_numpy(dtype=float)
        highs = group["high"].to_numpy(dtype=float)
        lows = group["low"].to_numpy(dtype=float)
        if "raw_initial_momentum" in group.columns:
            candidate_positions = np.flatnonzero(group["raw_initial_momentum"].to_numpy(dtype=bool))
        else:
            candidate_positions = np.arange(len(group))
        n = len(group)

        for position in candidate_positions:
            entry_position = position + 1
            end_position = min(position + horizon, n - 1)
            if entry_position > end_position:
                continue

            entry = opens[entry_position]
            if not np.isfinite(entry) or entry <= 0:
                continue

            future_highs = highs[entry_position : end_position + 1]
            future_lows = lows[entry_position : end_position + 1]
            if len(future_highs) < horizon:
                continue

            max_ret = np.nanmax(future_highs) / entry - 1.0
            min_ret = np.nanmin(future_lows) / entry - 1.0
            profit_hit = future_highs / entry - 1.0 >= profit_barrier
            stop_hit = future_lows / entry - 1.0 <= stop_barrier

            hit_profit_day = np.nan
            hit_stop_day = np.nan
            if np.any(profit_hit):
                hit_profit_day = float(np.argmax(profit_hit) + 1)
            if np.any(stop_hit):
                hit_stop_day = float(np.argmax(stop_hit) + 1)

            target = 0.0
            if np.isfinite(hit_profit_day) and (not np.isfinite(hit_stop_day) or hit_profit_day < hit_stop_day):
                target = 1.0

            row_idx = idx[position]
            labelled.at[row_idx, "entry_price"] = entry
            labelled.at[row_idx, "future_high_20d"] = np.nanmax(future_highs)
            labelled.at[row_idx, "future_low_20d"] = np.nanmin(future_lows)
            labelled.at[row_idx, "future_max_ret_20d"] = max_ret
            labelled.at[row_idx, "future_min_ret_20d"] = min_ret
            labelled.at[row_idx, "hit_profit_day"] = hit_profit_day
            labelled.at[row_idx, "hit_stop_day"] = hit_stop_day
            labelled.at[row_idx, "target_20d"] = target

    return add_sample_weights(
        labelled,
        mode=sample_weight_mode,
        cap=sample_weight_cap,
        scale=sample_weight_scale,
    )


def add_labels(
    df: pd.DataFrame,
    horizon: int = 20,
    threshold: float = 0.10,
    label_mode: str = "max_ret",
    profit_barrier: float = 0.10,
    stop_barrier: float = -0.07,
    sample_weight_mode: str = "future_max_ret",
    sample_weight_cap: float = 0.30,
    sample_weight_scale: float = 10.0,
) -> pd.DataFrame:
    if label_mode == "max_ret":
        return _add_max_return_labels(
            df,
            horizon=horizon,
            threshold=threshold,
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
            sample_weight_scale=sample_weight_scale,
        )
    if label_mode == "barrier":
        return _add_barrier_labels(
            df,
            horizon=horizon,
            profit_barrier=profit_barrier,
            stop_barrier=stop_barrier,
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
            sample_weight_scale=sample_weight_scale,
        )
    raise ValueError(f"Unknown label_mode: {label_mode}")


def make_event_dataset(
    df: pd.DataFrame,
    require_label: bool = True,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    events = df[df["initial_momentum"]].copy()
    if require_label:
        events = events.dropna(subset=["target_20d", "sample_weight"])
    events = events.replace([np.inf, -np.inf], np.nan)
    events = events.dropna(subset=feature_columns)
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
