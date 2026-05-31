"""Optuna hyperparameter tuning for the headline ``common_cpi`` CatBoost model.

Searches four hyperparameters — ``iterations``, ``depth``, ``l2_leaf_reg`` and
``learning_rate`` — over the min/max grid in `TuningConfig`, minimising the
validation metric on the date slice the config defines. ``iterations`` is tuned
directly (no early stopping), so each trial trains a fixed-size model.

`tune_common_cpi` returns the best params and the best trial's metrics;
`save_results` persists both as Parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import optuna
import polars as pl

from congifs.tuning_config import TuningConfig
from services.feature_service import FeatureSet
from services.modeling_service import (
    DataSplit,
    evaluate,
    split_by_date,
    train_catboost,
)


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a study: chosen params, best-model metrics, and the study."""

    best_params: dict
    best_metrics: dict
    study: optuna.study.Study


def _suggest_params(trial: optuna.Trial, config: TuningConfig) -> dict:
    """Sample one hyperparameter set from the configured min/max grid."""
    return {
        "iterations": trial.suggest_int(
            "iterations", config.iterations_min, config.iterations_max
        ),
        "depth": trial.suggest_int("depth", config.depth_min, config.depth_max),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", config.l2_leaf_reg_min, config.l2_leaf_reg_max, log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.learning_rate_min,
            config.learning_rate_max,
            log=True,
        ),
    }


def tune_common_cpi(fs: FeatureSet, config: TuningConfig) -> TuningResult:
    """Run an Optuna study over the CatBoost grid for ``config.target``.

    Each trial trains on the train slice and is scored by ``config.metric`` on
    the validation slice. The best params are then refit (on train for the
    reported validation metrics; on train+valid for held-out test metrics) so
    the returned metrics reflect a clean evaluation rather than the noisy
    in-loop trial value.
    """
    target = config.target
    if target not in fs.targets:
        raise ValueError(
            f"target {target!r} not in FeatureSet.targets; "
            "build features with weights so common_cpi is present"
        )
    if config.metric not in ("rmse", "mae"):
        raise ValueError(f"config.metric must be 'rmse' or 'mae', got {config.metric!r}")

    feature_names = fs.feature_names
    cat = fs.categorical
    naive_col = f"{target}__lag{config.horizon}"

    split: DataSplit = split_by_date(
        fs.df, config.valid_start, config.valid_end, config.test_start, config.test_end
    )
    if split.train.height == 0 or split.valid.height == 0:
        raise ValueError(
            "empty train or validation slice — check TUNE_VALID_START / "
            f"TUNE_VALID_END against the data range (train={split.train.height}, "
            f"valid={split.valid.height} rows)"
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, config)
        model = train_catboost(
            split.train, feature_names, cat, target, params,
            loss_function=config.loss_function,
        )
        m = evaluate(model, split.valid, feature_names, cat, target, naive_col)
        trial.set_user_attr("valid_rmse", m["rmse"])
        trial.set_user_attr("valid_mae", m["mae"])
        return m[config.metric]

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=config.n_trials)

    best_params = dict(study.best_params)

    # Refit cleanly for the reported metrics.
    model_v = train_catboost(
        split.train, feature_names, cat, target, best_params,
        loss_function=config.loss_function,
    )
    best_metrics = {
        f"valid_{k}": v
        for k, v in evaluate(model_v, split.valid, feature_names, cat, target, naive_col).items()
    }

    if split.test is not None and split.test.height > 0:
        train_valid = pl.concat([split.train, split.valid])
        model_t = train_catboost(
            train_valid, feature_names, cat, target, best_params,
            loss_function=config.loss_function,
        )
        best_metrics.update(
            {
                f"test_{k}": v
                for k, v in evaluate(model_t, split.test, feature_names, cat, target, naive_col).items()
            }
        )

    return TuningResult(best_params=best_params, best_metrics=best_metrics, study=study)


def save_results(
    result: TuningResult,
    config: TuningConfig,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write best params and best metrics to Parquet (one row each).

    Returns ``{"params": <path>, "metrics": <path>}``.
    """
    out = output_dir or config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    meta = {
        "target": config.target,
        "horizon": config.horizon,
        "metric": config.metric,
        "n_trials": config.n_trials,
        "best_value": float(result.study.best_value),
        "created_at": created_at,
    }
    params_df = pl.DataFrame([{**meta, **result.best_params}])
    metrics_df = pl.DataFrame([{**meta, **result.best_metrics}])

    params_path = out / "best_params.parquet"
    metrics_path = out / "best_metrics.parquet"
    params_df.write_parquet(params_path)
    metrics_df.write_parquet(metrics_path)
    return {"params": params_path, "metrics": metrics_path}
