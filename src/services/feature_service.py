"""Feature engineering for weekly CPI inflation forecasting.

Builds a single Polars DataFrame (one row per price publication date) for
CatBoost / TFT training, with strict horizon-aware data-leak prevention on
price-based features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import polars as pl

_DEFAULT_ROLLING_WINDOWS: list[int] = [4, 8, 13]

# Seasonal columns CatBoost should treat as categorical (small integer domains).
_CATEGORICAL_COLS: list[str] = ["year", "month", "week_of_year"]


@dataclass(frozen=True)
class FeatureSet:
    """Result of build_features, ready to feed CatBoost.

    df
        Full table: `date`, all model features, the raw per-category price
        columns (candidate targets), and `common_cpi` when weights are given.
    categorical / continuous
        Column-name lists of the model features by dtype. Pass `categorical`
        as CatBoost's `cat_features`.
    targets
        Candidate target columns present in `df` but excluded from the feature
        lists: every raw per-category price column plus `common_cpi` (if built).
        Pick one as `y`; the rest can be dropped.
    """

    df: pl.DataFrame
    categorical: list[str]
    continuous: list[str]
    targets: list[str]

    @property
    def feature_names(self) -> list[str]:
        """All model-input columns (categorical + continuous), excluding targets."""
        return self.categorical + self.continuous


def _pivot_prices(prices: pl.DataFrame) -> pl.DataFrame:
    return prices.pivot(values="price_index", index="date", on="category").sort("date")


def build_common_cpi(prices: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-category weekly indices into the headline ("common") CPI.

    Each observation date is weighted with its own year's basket weights, so the
    aggregation tracks Rosstat's annual basket re-weighting. The common index is
    the weight-normalised average of the per-category weekly price indices:

        common_cpi(t) = Σ_c weight(year(t), c) · price_index(t, c)
                        ─────────────────────────────────────────
                              Σ_c weight(year(t), c)

    Only categories present in both `prices` and that year's `weights` contribute
    (inner join), and the denominator re-normalises over exactly those, so the
    result is well-defined even when the basket and the weekly survey don't
    cover identical category sets.

    Parameters
    ----------
    prices:
        Long-format CPI DataFrame (columns: date, category, price_index).
    weights:
        Per-year basket weights (columns: year, category, weight).

    Returns
    -------
    DataFrame with columns: date, common_cpi (one row per date, sorted).
    """
    priced = prices.with_columns(
        pl.col("date").dt.year().cast(pl.Int32).alias("year")
    )
    joined = priced.join(
        weights.select("year", "category", "weight"),
        on=["year", "category"],
        how="inner",
    )
    return (
        joined.group_by("date")
        .agg(
            (
                (pl.col("price_index") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("common_cpi")
        )
        .sort("date")
    )


def _rebase_to_cumulative(df: pl.DataFrame, cat_cols: list[str]) -> pl.DataFrame:
    """Chain % → cumulative level index; first observation becomes the base."""
    return df.with_columns(
        [((pl.col(c) / 100).cum_prod() * 100).alias(c) for c in cat_cols]
    )


def _miacr_period_means(
    price_dates: pl.DataFrame,
    miacr: pl.DataFrame,
    miacr_val_cols: list[str],
) -> pl.DataFrame:
    """Mean of each MIACR variable over each price publication period.

    Each daily MIACR observation is assigned to the next price publication date
    via a forward asof-join, then averaged within that period. MIACR data
    before the first price date is excluded so the first period's mean reflects
    only the data published on that date.
    """
    first_date = price_dates["date"].min()
    miacr_trimmed = miacr.filter(pl.col("date") >= first_date).sort("date")

    # Map each MIACR day → its next price publication date ("period_date")
    mapped = miacr_trimmed.join_asof(
        price_dates.with_columns(pl.col("date").alias("period_date")),
        left_on="date",
        right_on="date",
        strategy="forward",
    ).drop_nulls("period_date")

    return (
        mapped
        .group_by("period_date")
        .agg([pl.col(c).mean().alias(f"{c}_period_mean") for c in miacr_val_cols])
        .rename({"period_date": "date"})
    )


def build_features(
    prices: pl.DataFrame,
    miacr: pl.DataFrame,
    weights: pl.DataFrame | None = None,
    horizon: int = 1,
    n_lags: int = 4,
    rolling_windows: list[int] | None = None,
    index_mode: Literal["chain", "base"] = "chain",
    lag_common_cpi: bool = True,
) -> FeatureSet:
    """Build a horizon-aware FeatureSet for weekly CPI inflation forecasting.

    Returns a FeatureSet whose `df` has one row per price publication date.
    The lag set is fixed to the `n_lags` periods at and beyond the horizon,
    i.e. ``[horizon, horizon + 1, …, horizon + n_lags - 1]`` (the smallest
    leak-free lags). Every price feature uses only that set, so nothing newer
    than `horizon` periods ago leaks into a row.

    Columns produced:

    Raw per-category prices (NOT features)
        The wide per-category price columns are kept in `df` so any one can be
        chosen as the target `y`. They are listed in `targets`, never in the
        feature lists.

    days_since_last_pub
        Calendar days since the previous publication. Captures irregular New
        Year gaps (e.g. 17 days between 28 Dec → 14 Jan instead of the usual
        7), which accumulate extra price change a model would otherwise
        misread as a single-week spike. (continuous)

    Seasonal
        year, month, week_of_year (categorical); sin/cos encodings of
        week-of-year and month (continuous) for smooth cyclical representation.

    Lag price features
        ``price[t − k]`` per category for each k in the lag set. (continuous)
        Named ``{category}__lag{k}``.

    Rolling mean price features
        ``rolling_mean(price.shift(k), window=w)`` per category, for each window
        w in `rolling_windows` and each shift k in the same lag set. Shifting
        before the window keeps every value leak-free. (continuous)
        Named ``{category}__roll{w}_lag{k}``.

    MIACR spot
        Most-recent daily rate on or before the price date (asof backward).
        No leakage: rates are published in real time. (continuous)

    MIACR period mean
        Arithmetic mean of each MIACR rate over (prev_price_date, price_date].
        (continuous)

    common_cpi
        Year-weighted headline weekly index, attached when `weights` is given.
        Listed in `targets`. When `lag_common_cpi` is set, it additionally
        gets its own lag and rolling-mean features (``common_cpi__lag{k}`` and
        ``common_cpi__roll{w}_lag{k}``) built from the same leak-free lag set
        as the per-category prices, so the raw column stays a target while its
        lagged history feeds the feature lists.

    Parameters
    ----------
    prices:
        Long-format CPI DataFrame from fetch_prices()
        (columns: date, category, price_index).
    miacr:
        Daily MIACR rates from fetch_mkr() (columns: date, miacr_rub_1d, …).
    weights:
        Optional per-year basket weights from fetch_weights()
        (columns: year, category, weight). When provided, a `common_cpi`
        target column is attached per date via build_common_cpi().
    horizon:
        Prediction horizon in publication periods (weeks). Sets the smallest
        leak-free lag; the lag set starts here.
    n_lags:
        Number of consecutive lags/shifts to generate, starting at `horizon`.
        Default 4 → lags [horizon, horizon+1, horizon+2, horizon+3].
    rolling_windows:
        Rolling-mean window sizes in periods. Default: [4, 8, 13].
    index_mode:
        "chain" — keep raw per-period % change (original Rosstat values).
        "base"  — convert to cumulative level; first observation = base (~100).
    lag_common_cpi:
        When True (default) and `weights` are given, treat `common_cpi` like
        any other price column for feature construction: generate its lag and
        rolling-mean features (and rebase it in "base" mode) alongside the
        per-category columns. The raw `common_cpi` column remains a target.
        Has no effect when `weights` is None.
    """
    if rolling_windows is None:
        rolling_windows = _DEFAULT_ROLLING_WINDOWS

    lags = [horizon + i for i in range(n_lags)]
    miacr_val_cols = [c for c in miacr.columns if c != "date"]

    # ── 1. Wide price matrix (raw per-category columns kept as candidate targets)
    df = _pivot_prices(prices)
    price_cols = [c for c in df.columns if c != "date"]

    # ── 1b. Headline common CPI, joined early so it can be lagged like a price
    has_common = weights is not None
    lag_common = has_common and lag_common_cpi
    if has_common:
        df = df.join(build_common_cpi(prices, weights), on="date", how="left")

    # Columns that get lag / rolling features. common_cpi joins this set only
    # when lag_common_cpi is on; its raw column always stays a target.
    lag_source_cols = price_cols + (["common_cpi"] if lag_common else [])

    # ── 2. Days since last publication ───────────────────────────────────────
    df = df.with_columns(
        pl.col("date").diff().dt.total_days().alias("days_since_last_pub")
    )

    # ── 3. Index mode ────────────────────────────────────────────────────────
    if index_mode == "base":
        df = _rebase_to_cumulative(df, lag_source_cols)

    # ── 4. Seasonal features ─────────────────────────────────────────────────
    tau = 2.0 * math.pi
    df = df.with_columns(
        [
            pl.col("date").dt.year().cast(pl.Int32).alias("year"),
            pl.col("date").dt.month().cast(pl.Int32).alias("month"),
            pl.col("date").dt.week().cast(pl.Int32).alias("week_of_year"),
            (tau / 52 * pl.col("date").dt.week()).sin().alias("sin_week"),
            (tau / 52 * pl.col("date").dt.week()).cos().alias("cos_week"),
            (tau / 12 * pl.col("date").dt.month()).sin().alias("sin_month"),
            (tau / 12 * pl.col("date").dt.month()).cos().alias("cos_month"),
        ]
    )

    # ── 5. Lag price features (lags = horizon … horizon + n_lags - 1) ─────────
    lag_cols = [f"{c}__lag{k}" for c in lag_source_cols for k in lags]
    if lag_cols:
        df = df.with_columns(
            [
                pl.col(c).shift(k).alias(f"{c}__lag{k}")
                for c in lag_source_cols
                for k in lags
            ]
        )

    # ── 6. Rolling mean features (each window, shifted by each lag) ───────────
    roll_cols = [
        f"{c}__roll{w}_lag{k}"
        for c in lag_source_cols
        for w in rolling_windows
        for k in lags
    ]
    if roll_cols:
        df = df.with_columns(
            [
                pl.col(c).shift(k).rolling_mean(window_size=w).alias(f"{c}__roll{w}_lag{k}")
                for c in lag_source_cols
                for w in rolling_windows
                for k in lags
            ]
        )

    # ── 7. MIACR spot (asof backward: most-recent rate at or before price date)
    df = df.join_asof(miacr.sort("date"), on="date", strategy="backward")

    # ── 8. MIACR period means ────────────────────────────────────────────────
    period_means = _miacr_period_means(
        df.select("date").sort("date"), miacr, miacr_val_cols
    )
    df = df.join(period_means, on="date", how="left")
    period_cols = [f"{c}_period_mean" for c in miacr_val_cols]

    # ── 9. Target: year-weighted headline ("common") CPI ─────────────────────
    # common_cpi was joined early (step 1b); the raw column stays a target.
    targets = list(price_cols)
    if has_common:
        targets.append("common_cpi")

    continuous = (
        ["days_since_last_pub", "sin_week", "cos_week", "sin_month", "cos_month"]
        + lag_cols
        + roll_cols
        + miacr_val_cols
        + period_cols
    )
    return FeatureSet(
        df=df,
        categorical=list(_CATEGORICAL_COLS),
        continuous=continuous,
        targets=targets,
    )
