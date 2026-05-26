from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "screening_settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "gate": {
        "cooldown_days": 20,
        "min_turnover_5d": 50_000_000,
        "min_ret_5d": -0.01,
        "min_turnover_ratio_1d_20d": 1.05,
        "min_turnover_ratio_5d_20d": 1.05,
        "min_close_ma25_ratio": -0.01,
    },
    "label": {
        "mode": "barrier",
        "horizon": 20,
        "target_threshold": 0.10,
        "profit_barrier": 0.15,
        "stop_barrier": -0.10,
    },
    "screen": {
        "signal_count_days": 6,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    settings_path = Path(path) if path is not None else SETTINGS_PATH
    if not settings_path.exists():
        return DEFAULT_SETTINGS
    loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    return _deep_merge(DEFAULT_SETTINGS, loaded)


SETTINGS = load_settings()
GATE_SETTINGS = SETTINGS["gate"]
LABEL_SETTINGS = SETTINGS["label"]
SCREEN_SETTINGS = SETTINGS["screen"]
