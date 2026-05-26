from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RISK_ADJUSTMENT_PRESETS: dict[str, dict[str, float]] = {
    "none": {
        "volatility": 0.0,
        "intraday_range": 0.0,
        "upper_shadow": 0.0,
        "ret20_overheat": 0.0,
        "ret5_overheat": 0.0,
        "weak_close": 0.0,
        "risk_cap": 0.0,
        "ret5_accel": 0.0,
        "turnover": 0.0,
        "high_breakout": 0.0,
        "upside_cap": 0.0,
    },
    "light": {
        "volatility": 0.08,
        "intraday_range": 0.06,
        "upper_shadow": 0.05,
        "ret20_overheat": 0.06,
        "ret5_overheat": 0.04,
        "weak_close": 0.04,
        "risk_cap": 0.30,
        "ret5_accel": 0.06,
        "turnover": 0.04,
        "high_breakout": 0.03,
        "upside_cap": 0.13,
    },
    "current": {
        "volatility": 0.14,
        "intraday_range": 0.10,
        "upper_shadow": 0.09,
        "ret20_overheat": 0.10,
        "ret5_overheat": 0.07,
        "weak_close": 0.06,
        "risk_cap": 0.45,
        "ret5_accel": 0.08,
        "turnover": 0.06,
        "high_breakout": 0.04,
        "upside_cap": 0.18,
    },
    "strict": {
        "volatility": 0.20,
        "intraday_range": 0.15,
        "upper_shadow": 0.13,
        "ret20_overheat": 0.15,
        "ret5_overheat": 0.10,
        "weak_close": 0.08,
        "risk_cap": 0.65,
        "ret5_accel": 0.08,
        "turnover": 0.06,
        "high_breakout": 0.04,
        "upside_cap": 0.18,
    },
    "volatility_only": {
        "volatility": 0.18,
        "intraday_range": 0.13,
        "upper_shadow": 0.0,
        "ret20_overheat": 0.0,
        "ret5_overheat": 0.0,
        "weak_close": 0.0,
        "risk_cap": 0.35,
        "ret5_accel": 0.08,
        "turnover": 0.06,
        "high_breakout": 0.04,
        "upside_cap": 0.18,
    },
    "overheat_only": {
        "volatility": 0.0,
        "intraday_range": 0.0,
        "upper_shadow": 0.08,
        "ret20_overheat": 0.16,
        "ret5_overheat": 0.12,
        "weak_close": 0.06,
        "risk_cap": 0.35,
        "ret5_accel": 0.08,
        "turnover": 0.06,
        "high_breakout": 0.04,
        "upside_cap": 0.18,
    },
}


class MomentumNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


@dataclass
class SimpleScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "SimpleScaler":
        values = frame.astype(float).to_numpy()
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(mean_=mean, scale_=scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.astype(float).to_numpy()
        return (values - self.mean_) / self.scale_


@dataclass
class TrainResult:
    model: MomentumNet
    scaler: SimpleScaler
    feature_columns: list[str]
    best_valid_loss: float
    epochs_trained: int


def _split_by_date(events: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)
    train = events[events["date"] <= train_end_ts].copy()
    valid = events[(events["date"] > train_end_ts) & (events["date"] <= valid_end_ts)].copy()
    test = events[events["date"] > valid_end_ts].copy()
    return train, valid, test


def topn_selection_score(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    n: int = 50,
    stop_penalty: float = 0.5,
) -> float:
    if frame.empty or len(probabilities) == 0:
        return float("-inf")
    scored = frame.copy()
    scored["follow_through_prob"] = probabilities
    top = scored.sort_values("follow_through_prob", ascending=False).head(n)
    if top.empty or top["target_20d"].isna().all():
        return float("-inf")

    precision = float(top["target_20d"].mean())
    if "hit_stop_day" not in top.columns or "hit_profit_day" not in top.columns:
        return precision

    hit_stop = top["hit_stop_day"].notna()
    hit_profit = top["hit_profit_day"].notna()
    stop_first = hit_stop & (~hit_profit | (top["hit_stop_day"] <= top["hit_profit_day"]))
    return precision - stop_penalty * float(stop_first.mean())


DEFAULT_RISK_ADJUSTMENT = "volatility_only"


def train_model(
    events: pd.DataFrame,
    feature_columns: list[str],
    train_end: str,
    valid_end: str,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 12,
    seed: int = 42,
    risk_adjustment: str = DEFAULT_RISK_ADJUSTMENT,
) -> tuple[TrainResult, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, valid, test = _split_by_date(events, train_end, valid_end)
    if train.empty:
        raise ValueError("Training split is empty. Use an earlier start date or later --train-end.")
    if valid.empty:
        valid = train.sample(min(len(train), max(1, len(train) // 5)), random_state=seed)

    scaler = SimpleScaler.fit(train[feature_columns])
    x_train = scaler.transform(train[feature_columns])
    y_train = train["target_20d"].astype(float).to_numpy()
    w_train = train["sample_weight"].astype(float).to_numpy()

    x_valid = scaler.transform(valid[feature_columns].astype(float))
    y_valid = valid["target_20d"].astype(float).to_numpy()
    w_valid = valid["sample_weight"].astype(float).to_numpy()

    model = MomentumNet(input_dim=len(feature_columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_valid_loss = float("inf")
    best_selection_score = float("-inf")
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            losses = loss_fn(model(xb), yb)
            loss = (losses * wb).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_valid, dtype=torch.float32)
            yv = torch.tensor(y_valid, dtype=torch.float32)
            wv = torch.tensor(w_valid, dtype=torch.float32)
            valid_logits = model(xv)
            valid_loss = (loss_fn(valid_logits, yv) * wv).mean().item()
            valid_prob = torch.sigmoid(valid_logits).cpu().numpy()

        selection_score = topn_selection_score(valid, valid_prob, n=50, stop_penalty=0.5)
        if selection_score > best_selection_score:
            best_selection_score = selection_score

        epochs_trained = epoch
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate_splits(
        model,
        scaler,
        feature_columns,
        {"train": train, "valid": valid, "test": test},
        risk_adjustment=risk_adjustment,
    )
    metrics["best_valid_loss"] = best_valid_loss
    metrics["best_selection_score"] = best_selection_score
    metrics["epochs_trained"] = float(epochs_trained)
    return TrainResult(model, scaler, feature_columns, best_valid_loss, epochs_trained), metrics


def predict_proba(model: MomentumNet, scaler: SimpleScaler, feature_columns: list[str], frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([])
    x = scaler.transform(frame[feature_columns].astype(float))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def add_risk_adjusted_score(frame: pd.DataFrame, preset: str = DEFAULT_RISK_ADJUSTMENT) -> pd.DataFrame:
    if preset not in RISK_ADJUSTMENT_PRESETS:
        raise ValueError(f"Unknown risk_adjustment preset: {preset}")
    weights = RISK_ADJUSTMENT_PRESETS[preset]
    scored = frame.copy()
    risk_penalty = (
        scored["volatility_20d"].fillna(0.0).clip(lower=0.02, upper=0.12).sub(0.02).div(0.10).mul(weights["volatility"])
        + scored["intraday_range_ratio"].fillna(0.0).clip(lower=0.04, upper=0.20).sub(0.04).div(0.16).mul(weights["intraday_range"])
        + scored["upper_shadow_ratio"].fillna(0.0).clip(lower=0.20, upper=0.80).sub(0.20).div(0.60).mul(weights["upper_shadow"])
        + scored["ret_20d"].fillna(0.0).clip(lower=0.20, upper=1.00).sub(0.20).div(0.80).mul(weights["ret20_overheat"])
        + scored["ret_5d"].fillna(0.0).clip(lower=0.15, upper=0.60).sub(0.15).div(0.45).mul(weights["ret5_overheat"])
        + (1.0 - scored["close_position_in_range"].fillna(0.5).clip(lower=0.0, upper=1.0)).mul(weights["weak_close"])
    )
    upside_bonus = (
        scored["ret_5d_accel"].fillna(0.0).clip(lower=0.00, upper=0.20).div(0.20).mul(weights["ret5_accel"])
        + scored["turnover_ratio_5d_20d"].fillna(1.0).clip(lower=1.00, upper=2.50).sub(1.00).div(1.50).mul(weights["turnover"])
        + scored["is_20d_high"].fillna(0.0).clip(lower=0.0, upper=1.0).mul(weights["high_breakout"])
    )
    scored["risk_penalty"] = risk_penalty.clip(lower=0.0, upper=weights["risk_cap"])
    scored["upside_bonus"] = upside_bonus.clip(lower=0.0, upper=weights["upside_cap"])
    scored["final_score"] = (scored["follow_through_prob"] + scored["upside_bonus"] - scored["risk_penalty"]).clip(0.0, 1.0)
    return scored


def _score_column(frame: pd.DataFrame) -> str:
    return "final_score" if "final_score" in frame.columns else "follow_through_prob"


def precision_at_n(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return float("nan")
    top = frame.sort_values(_score_column(frame), ascending=False).head(n)
    if top.empty or top["target_20d"].isna().all():
        return float("nan")
    return float(top["target_20d"].mean())


def stop_first_rate_at_n(frame: pd.DataFrame, n: int) -> float:
    if frame.empty or "hit_stop_day" not in frame.columns or "hit_profit_day" not in frame.columns:
        return float("nan")
    top = frame.sort_values(_score_column(frame), ascending=False).head(n)
    if top.empty:
        return float("nan")
    hit_stop = top["hit_stop_day"].notna()
    hit_profit = top["hit_profit_day"].notna()
    stop_first = hit_stop & (~hit_profit | (top["hit_stop_day"] <= top["hit_profit_day"]))
    return float(stop_first.mean())


def avg_days_to_profit_at_n(frame: pd.DataFrame, n: int) -> float:
    if frame.empty or "hit_profit_day" not in frame.columns:
        return float("nan")
    top = frame.sort_values(_score_column(frame), ascending=False).head(n)
    profit_days = top.loc[top["hit_profit_day"].notna(), "hit_profit_day"]
    if profit_days.empty:
        return float("nan")
    return float(profit_days.mean())


def evaluate_splits(
    model: MomentumNet,
    scaler: SimpleScaler,
    feature_columns: list[str],
    splits: dict[str, pd.DataFrame],
    risk_adjustment: str = DEFAULT_RISK_ADJUSTMENT,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, split in splits.items():
        if split.empty:
            metrics[f"{name}_events"] = 0.0
            continue
        scored = split.copy()
        scored["follow_through_prob"] = predict_proba(model, scaler, feature_columns, scored)
        scored = add_risk_adjusted_score(scored, preset=risk_adjustment)
        metrics[f"{name}_events"] = float(len(scored))
        metrics[f"{name}_precision_at_20"] = precision_at_n(scored, 20)
        metrics[f"{name}_precision_at_50"] = precision_at_n(scored, 50)
        metrics[f"{name}_stop_first_rate_at_20"] = stop_first_rate_at_n(scored, 20)
        metrics[f"{name}_stop_first_rate_at_50"] = stop_first_rate_at_n(scored, 50)
        metrics[f"{name}_avg_days_to_profit_at_20"] = avg_days_to_profit_at_n(scored, 20)
        metrics[f"{name}_avg_days_to_profit_at_50"] = avg_days_to_profit_at_n(scored, 50)
        ranked = scored.sort_values(_score_column(scored), ascending=False)
        top20 = ranked.head(20)
        top50 = ranked.head(50)
        metrics[f"{name}_avg_future_max_ret_at_20"] = float(top20["future_max_ret_20d"].mean()) if not top20.empty else float("nan")
        metrics[f"{name}_avg_future_min_ret_at_20"] = float(top20["future_min_ret_20d"].mean()) if not top20.empty and "future_min_ret_20d" in top20.columns else float("nan")
        metrics[f"{name}_avg_future_max_ret_at_50"] = float(top50["future_max_ret_20d"].mean()) if not top50.empty else float("nan")
        metrics[f"{name}_avg_future_min_ret_at_50"] = float(top50["future_min_ret_20d"].mean()) if not top50.empty and "future_min_ret_20d" in top50.columns else float("nan")
    return metrics


def save_artifacts(result: TrainResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "input_dim": len(result.feature_columns),
            "feature_columns": result.feature_columns,
            "scaler_mean": result.scaler.mean_.tolist(),
            "scaler_scale": result.scaler.scale_.tolist(),
        },
        path,
    )


def load_artifacts(path: str | Path) -> tuple[MomentumNet, SimpleScaler, list[str]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MomentumNet(input_dim=int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    scaler = SimpleScaler(
        mean_=np.asarray(payload["scaler_mean"], dtype=float),
        scale_=np.asarray(payload["scaler_scale"], dtype=float),
    )
    return model, scaler, list(payload["feature_columns"])
