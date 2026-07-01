"""Shared TemporalBPR configuration utilities.

Centralizes defaults and JSON loading so all runners use the same config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_TBPR_CONFIG: Dict[str, Any] = {
    "factors": 64,
    "learning_rate": 0.012,
    "reg": 1e-4,
    "epochs": 16,
    "alpha": 0.9,
    "beta": 1.1,
    "random_state": 42,
    "recency_boost": 1.5,
    "max_recent_boost": 180,
    "temporal_activation_weight": 0.6,
    "temporal_activation_decay": 0.5,
    "hard_negative_ratio": 0.3,
    "hard_negative_pool": 30,
    "target_repeat_ratio": 0.35,
    "calibration_strength": 1.0,
    "calibration_iterations": 16,
    "verbose": False,
}

_INT_KEYS = {
    "factors",
    "epochs",
    "random_state",
    "max_recent_boost",
    "hard_negative_pool",
    "calibration_iterations",
}

_FLOAT_KEYS = {
    "learning_rate",
    "reg",
    "alpha",
    "beta",
    "recency_boost",
    "temporal_activation_weight",
    "temporal_activation_decay",
    "hard_negative_ratio",
    "target_repeat_ratio",
    "calibration_strength",
}


def normalize_tbpr_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for key in _INT_KEYS:
        if key in out:
            out[key] = int(out[key])
    for key in _FLOAT_KEYS:
        if key in out and out[key] is not None:
            out[key] = float(out[key])
    if "verbose" in out:
        out["verbose"] = bool(out["verbose"])
    return out


def load_tbpr_config(config_path: str | Path | None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_TBPR_CONFIG)
    if not config_path:
        return normalize_tbpr_config(cfg)

    path = Path(config_path)
    if not path.exists():
        return normalize_tbpr_config(cfg)

    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    for key in cfg.keys():
        if key in loaded:
            cfg[key] = loaded[key]

    return normalize_tbpr_config(cfg)
