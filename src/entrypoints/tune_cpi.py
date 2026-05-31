"""Entrypoint: Optuna-tune the headline common_cpi CatBoost model.

Wires the collectors → feature_service → tuning_service, runs the study, and
writes best_params.parquet / best_metrics.parquet under the tuning output dir.

Run from the project root:
    $env:PYTHONPATH = "src"; python src/entrypoints/tune_cpi.py
    $env:PYTHONPATH = "src"; python src/entrypoints/tune_cpi.py --n-trials 100
    $env:PYTHONPATH = "src"; python src/entrypoints/tune_cpi.py --output-dir data/tuning

Search grid and validation dates come from TuningConfig (TUNE_* env vars / .env);
feature construction (horizon, n_lags, …) is fixed below to match the notebook.
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure Cyrillic category names print correctly on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import polars as pl

from congifs.config_data import MkrConfig
from congifs.cpi_config import CpiConfig
from congifs.tuning_config import TuningConfig
from services.data_collectors.cpi_collector import fetch_prices, fetch_weights
from services.data_collectors.mkr_collector import fetch_mkr
from services.data_collectors.mkr_collector import save_parquet as save_mkr_parquet
from services.feature_service import build_features
from services.tuning_service import save_results, tune_common_cpi

# Feature-construction settings (not tuned). Keep in sync with features.ipynb.
HORIZON = 1
N_LAGS = 2
ROLLING_WINDOWS = [1]
INDEX_MODE = "chain"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna-tune the common_cpi CatBoost model")
    p.add_argument("--n-trials", type=int, default=None, help="Override TuningConfig.n_trials")
    p.add_argument("--output-dir", type=Path, default=None, help="Override Parquet output dir")
    return p.parse_args()


def _load_miacr(mkr_cfg: MkrConfig) -> pl.DataFrame:
    """Read the newest cached MIACR parquet, else fetch from CBR and cache it."""
    mkr_dir = mkr_cfg.output_dir
    cached = sorted(mkr_dir.glob("*.parquet")) if mkr_dir.exists() else []
    if cached:
        print(f"MIACR: reading cached {cached[-1]}")
        return pl.read_parquet(cached[-1])
    print("MIACR: no cache — fetching from CBR ...")
    miacr = fetch_mkr(mkr_cfg.base_url, mkr_cfg.from_date)
    print(f"MIACR: cached → {save_mkr_parquet(miacr, mkr_dir)}")
    return miacr


def main() -> None:
    args = _parse_args()
    cpi_cfg = CpiConfig()
    mkr_cfg = MkrConfig()
    tune_cfg = TuningConfig()
    if args.n_trials is not None:
        tune_cfg = tune_cfg.model_copy(update={"n_trials": args.n_trials})

    print("Loading data ...")
    prices = fetch_prices(cpi_cfg.prices_file)
    weights = fetch_weights(cpi_cfg.weights_file)
    miacr = _load_miacr(mkr_cfg)

    print("Building features ...")
    fs = build_features(
        prices,
        miacr,
        weights=weights,
        horizon=HORIZON,
        n_lags=N_LAGS,
        rolling_windows=ROLLING_WINDOWS,
        index_mode=INDEX_MODE,
        lag_common_cpi=True,
    )
    print(
        f"  rows={fs.df.height}  features={len(fs.feature_names)}  "
        f"target={tune_cfg.target!r}"
    )

    print(
        f"Tuning {tune_cfg.n_trials} trials  |  "
        f"valid {tune_cfg.valid_start}→{tune_cfg.valid_end}  "
        f"test {tune_cfg.test_start}→{tune_cfg.test_end}"
    )
    result = tune_common_cpi(fs, tune_cfg)

    print("\nBest params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print("Best metrics:")
    for k, v in result.best_metrics.items():
        print(f"  {k}: {v}")

    paths = save_results(result, tune_cfg, args.output_dir)
    print(f"\nSaved params  → {paths['params'].resolve()}")
    print(f"Saved metrics → {paths['metrics'].resolve()}")


if __name__ == "__main__":
    main()
