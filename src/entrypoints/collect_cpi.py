"""Entrypoint: parse Rosstat CPI prices and weights from local XLSX files.

Run from the project root:
    $env:PYTHONPATH = "src"; python src/entrypoints/collect_cpi.py
    $env:PYTHONPATH = "src"; python src/entrypoints/collect_cpi.py --save
    $env:PYTHONPATH = "src"; python src/entrypoints/collect_cpi.py --save --output-dir data/cpi
    $env:PYTHONPATH = "src"; python src/entrypoints/collect_cpi.py --weights-only
    $env:PYTHONPATH = "src"; python src/entrypoints/collect_cpi.py --prices-only

Override source file paths via env vars:
    $env:CPI_PRICES_FILE = "src/test_data/prices.xlsx"
    $env:CPI_WEIGHTS_FILE = "src/test_data/weights.xlsx"
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure Cyrillic characters print correctly on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from congifs.cpi_config import CpiConfig
from services.data_collectors.cpi_collector import fetch_prices, fetch_weights, save_parquet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse Rosstat CPI prices and basket weights from XLSX")
    p.add_argument("--save", action="store_true", help="Write results to Parquet on disk")
    p.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                   help="Parquet output directory (default: config.output_dir)")
    exclusive = p.add_mutually_exclusive_group()
    exclusive.add_argument("--prices-only", action="store_true", help="Only parse prices")
    exclusive.add_argument("--weights-only", action="store_true", help="Only parse weights")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = CpiConfig()
    out_dir = args.output_dir or config.output_dir

    do_prices = not args.weights_only
    do_weights = not args.prices_only

    if do_prices:
        print(f"Parsing prices from {config.prices_file} ...")
        prices = fetch_prices(config.prices_file)
        print(prices)
        print(f"\nPrices shape: {prices.shape}  |  date range: {prices['date'].min()} → {prices['date'].max()}")
        if args.save:
            path = save_parquet(prices, out_dir, "cpi_prices.parquet")
            print(f"Saved → {path.resolve()}")

    if do_weights:
        print(f"\nParsing weights from {config.weights_file} ...")
        weights = fetch_weights(config.weights_file)
        print(weights)
        print(f"\nWeights shape: {weights.shape}  |  years: {weights['year'].min()} → {weights['year'].max()}")
        if args.save:
            path = save_parquet(weights, out_dir, "cpi_weights.parquet")
            print(f"Saved → {path.resolve()}")


if __name__ == "__main__":
    main()
