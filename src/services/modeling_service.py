"""CatBoost modeling for weekly CPI forecasting.

The single place CatBoost is called. Turns a feature table (from
`feature_service.build_features`) into date-based train/valid/test splits,
builds `Pool`s, fits a `CatBoostRegressor`, predicts, and scores predictions
against the realised target and the naive last-known baseline.

Pure library: every function takes data + config values and returns plain
objects, so both the notebook and `tuning_service` can drive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool


@dataclass(frozen=True)
class DataSplit:
    """Date-ordered split of a feature table. `test` is None when unconfigured."""

    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame | None


def split_by_date(
    df: pl.DataFrame,
    valid_start: date,
    valid_end: date,
    test_start: date | None = None,
    test_end: date | None = None,
) -> DataSplit:
    """Chronological split: train < valid_start ≤ valid < valid_end ≤ test < test_end.

    No row leaks across boundaries; `test` is built only when `test_start` is
    given (left-closed, right-open, optionally bounded by `test_end`).
    """
    train = df.filter(pl.col("date") < valid_start)
    valid = df.filter((pl.col("date") >= valid_start) & (pl.col("date") < valid_end))
    test: pl.DataFrame | None = None
    if test_start is not None:
        cond = pl.col("date") >= test_start
        if test_end is not None:
            cond = cond & (pl.col("date") < test_end)
        test = df.filter(cond)
    return DataSplit(train=train, valid=valid, test=test)


def build_pool(
    df: pl.DataFrame,
    feature_names: list[str],
    cat_features: list[str],
    target: str | None = None,
) -> Pool:
    """Build a CatBoost `Pool`. With `target`, rows whose label is null are
    dropped (early lag/roll rows); feature NaNs are kept — CatBoost handles them.
    """
    if target is not None:
        df = df.filter(pl.col(target).is_not_null())
        return Pool(
            data=df.select(feature_names),
            label=df.select(target),
            cat_features=cat_features,
        )
    return Pool(data=df.select(feature_names), cat_features=cat_features)


def train_catboost(
    train_df: pl.DataFrame,
    feature_names: list[str],
    cat_features: list[str],
    target: str,
    params: dict,
    *,
    valid_df: pl.DataFrame | None = None,
    loss_function: str = "RMSE",
    early_stopping_rounds: int | None = None,
    verbose: bool | int = False,
) -> CatBoostRegressor:
    """Fit a `CatBoostRegressor` with `params` (iterations, depth, l2_leaf_reg,
    learning_rate, …). When `valid_df` is given it is passed as `eval_set` so
    `early_stopping_rounds` can take effect; the tuner leaves it off because it
    optimises `iterations` directly.
    """
    train_pool = build_pool(train_df, feature_names, cat_features, target)
    eval_set = None
    if valid_df is not None and valid_df.height > 0:
        eval_set = build_pool(valid_df, feature_names, cat_features, target)

    model = CatBoostRegressor(loss_function=loss_function, **params)
    model.fit(
        train_pool,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    return model


def predict(
    model: CatBoostRegressor,
    df: pl.DataFrame,
    feature_names: list[str],
    cat_features: list[str],
) -> np.ndarray:
    return model.predict(build_pool(df, feature_names, cat_features))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def evaluate(
    model: CatBoostRegressor,
    df: pl.DataFrame,
    feature_names: list[str],
    cat_features: list[str],
    target: str,
    naive_col: str | None = None,
) -> dict[str, float]:
    """Score `model` on `df`: rmse/mae plus skill ratios vs the naive baseline.

    `naive_col` is the last-known value (e.g. ``common_cpi__lag1``). Skill
    ratios are model_error / naive_error, so < 1 means the model beats naive.
    Rows with a null target are dropped before scoring.
    """
    df = df.filter(pl.col(target).is_not_null())
    metrics: dict[str, float] = {"n": float(df.height)}
    if df.height == 0:
        metrics["rmse"] = float("nan")
        metrics["mae"] = float("nan")
        return metrics

    y_true = df[target].to_numpy()
    y_pred = predict(model, df, feature_names, cat_features)
    metrics.update(regression_metrics(y_true, y_pred))

    if naive_col is not None and naive_col in df.columns:
        naive = regression_metrics(y_true, df[naive_col].to_numpy())
        metrics["rmse_naive"] = naive["rmse"]
        metrics["mae_naive"] = naive["mae"]
        metrics["rmse_skill"] = metrics["rmse"] / naive["rmse"] if naive["rmse"] else float("nan")
        metrics["mae_skill"] = metrics["mae"] / naive["mae"] if naive["mae"] else float("nan")
    return metrics
