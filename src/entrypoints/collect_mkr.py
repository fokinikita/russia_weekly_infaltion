"""Entrypoint: collect MIACR interbank rates from CBR.

Run from the project root:
    PYTHONPATH=src python src/entrypoints/collect_mkr.py
    PYTHONPATH=src python src/entrypoints/collect_mkr.py --save
    PYTHONPATH=src python src/entrypoints/collect_mkr.py --save --output-dir /path/to/dir

Override config via env vars:
    MKR_FROM_DATE=01.01.2023 PYTHONPATH=src python src/entrypoints/collect_mkr.py --save
"""

import argparse
import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from congifs.config_data import MkrConfig
from services.data_collectors.mkr_collector import fetch_mkr, save_parquet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect MIACR 1-day rates from CBR")
    p.add_argument(
        "--save",
        action="store_true",
        help="Write result to Parquet on disk",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Parquet output directory (default: config.output_dir)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = MkrConfig()

    print(f"Fetching MIACR 1d data  {config.from_date} -> today ...")
    df = fetch_mkr(config.base_url, config.from_date)

    print(df)
    print(f"\nShape: {df.shape}  |  date range: {df['date'].min()} → {df['date'].max()}")

    if args.save:
        out_dir = args.output_dir or config.output_dir
        path = save_parquet(df, out_dir)
        print(f"Saved → {path.resolve()}")


if __name__ == "__main__":
    main()
