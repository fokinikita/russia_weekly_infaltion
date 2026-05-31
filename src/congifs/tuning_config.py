from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TuningConfig(BaseSettings):
    """Settings for the Optuna CatBoost tuner (headline ``common_cpi`` only).

    Loaded via pydantic-settings from env vars / ``.env`` with the ``TUNE_``
    prefix, e.g. ``TUNE_N_TRIALS=100`` or ``TUNE_VALID_START=2025-12-01``.

    Two things are configurable here, as requested:

    * the min/max hyperparameter search grid (``*_min`` / ``*_max`` pairs), and
    * the date boundaries of the train / validation / test split.
    """

    model_config = SettingsConfigDict(env_prefix="TUNE_", env_file=".env", extra="ignore")

    # ── Target ───────────────────────────────────────────────────────────────
    # Tuning is headline-CPI only. `horizon` sets the naive baseline column
    # ({target}__lag{horizon}) the metrics are scored against.
    target: str = "common_cpi"
    horizon: int = 1

    # ── Date split ───────────────────────────────────────────────────────────
    # train  = date <  valid_start
    # valid  = valid_start <= date < valid_end   (objective is scored here)
    # test   = test_start  <= date < test_end    (held-out, reported only)
    valid_start: date = date(2025, 12, 1)
    valid_end: date = date(2026, 3, 1)
    test_start: date | None = date(2026, 3, 1)
    test_end: date | None = None

    # ── Optuna ───────────────────────────────────────────────────────────────
    n_trials: int = 50
    seed: int = 42
    loss_function: str = "RMSE"
    # Objective metric to minimise on the validation slice ("rmse" or "mae").
    metric: str = "rmse"

    # ── Hyperparameter search grid (inclusive bounds) ────────────────────────
    iterations_min: int = 100
    iterations_max: int = 2000
    depth_min: int = 3
    depth_max: int = 10
    # l2_leaf_reg and learning_rate are searched on a log scale.
    l2_leaf_reg_min: float = 1.0
    l2_leaf_reg_max: float = 30.0
    learning_rate_min: float = 0.005
    learning_rate_max: float = 0.3

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = Path("data/tuning")
