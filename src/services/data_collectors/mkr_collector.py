"""Collect MIACR interbank rates from the CBR website.

Data source: https://www.cbr.ru/hd_base/mkr/mkr_base/
The page returns one HTML table per indicator/currency combination,
each preceded by an <h3> heading. We match headings by keyword.
"""

from datetime import date, datetime
from pathlib import Path

import polars as pl
import requests
from bs4 import BeautifulSoup, Tag

# (required_keywords, forbidden_keywords, output_column_name)
# Keywords match against the <h3> heading that precedes each table.
_TABLE_SPECS: list[tuple[list[str], list[str], str]] = [
    (["рублях"],      ["долларах", "высоким", "спекулятивным", "Обороты"], "miacr_rub_1d"),
    (["долларах"],    ["Обороты"],                                          "miacr_usd_1d"),
    (["высоким"],     ["Обороты"],                                          "miacr_ig_rub_1d"),
    (["спекулятивным"], ["Обороты"],                                        "miacr_b_rub_1d"),
    (["Обороты", "рублях"], ["долларах", "высоким", "спекулятивным"],       "miacr_rub_1d_volume"),
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; miacr-collector/1.0)"}


def _matches(heading: str, required: list[str], forbidden: list[str]) -> bool:
    return all(kw in heading for kw in required) and not any(kw in heading for kw in forbidden)


def _parse_value(raw: str) -> float | None:
    clean = raw.replace("\xa0", "").replace(" ", "").strip()
    if clean in ("—", "-", ""):
        return None
    return float(clean.replace(",", "."))


def _table_to_series(table: Tag, col: str) -> pl.DataFrame:
    records: list[dict] = []
    for row in table.find_all("tr")[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        date_str = cells[0].get_text(strip=True)
        if not date_str:
            continue
        try:
            d = datetime.strptime(date_str, "%d.%m.%Y").date()
        except ValueError:
            continue
        val = _parse_value(cells[1].get_text(strip=True))
        records.append({"date": d, col: val})
    return pl.DataFrame(records, schema={"date": pl.Date, col: pl.Float64})


def _build_params(from_date: str) -> list[tuple[str, str]]:
    return [
        ("UniDbQuery.Posted",  "True"),
        ("UniDbQuery.From",    from_date),
        ("UniDbQuery.To",      date.today().strftime("%d.%m.%Y")),
        ("UniDbQuery.st",      "SF"),
        ("UniDbQuery.st",      "HR"),
        ("UniDbQuery.st",      "MB"),
        ("UniDbQuery.ob",      "OB_MIACR_0"),
        ("UniDbQuery.ob",      "OB_MIACR_IG"),
        ("UniDbQuery.ob",      "OB_MIACR_B"),
        ("UniDbQuery.Currency", "-1"),
        ("UniDbQuery.sk",      "Dd1_"),   # 1-day maturity only
    ]


def fetch_mkr(base_url: str, from_date: str) -> pl.DataFrame:
    """Fetch MIACR 1-day rates and volumes from CBR and return a Polars DataFrame.

    Columns:
      date                – observation date
      miacr_rub_1d        – MIACR weighted rate, RUB, 1 day (% p.a.)
      miacr_usd_1d        – MIACR weighted rate, USD, 1 day (% p.a.)
      miacr_ig_rub_1d     – MIACR-IG rate (high credit rating banks), RUB, 1 day
      miacr_b_rub_1d      – MIACR-B rate (speculative rating banks), RUB, 1 day
      miacr_rub_1d_volume – daily turnover of RUB interbank loans, 1 day (bln RUB)
    """
    resp = requests.get(base_url, params=_build_params(from_date), headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")

    frames: list[pl.DataFrame] = []
    matched: set[str] = set()

    for table in tables:
        heading_tag = table.find_previous(["h2", "h3", "h4"])
        if not heading_tag:
            continue
        heading = heading_tag.get_text(strip=True)

        for required, forbidden, col in _TABLE_SPECS:
            if col in matched:
                continue
            if _matches(heading, required, forbidden):
                frames.append(_table_to_series(table, col))
                matched.add(col)
                break

    missing = {col for *_, col in _TABLE_SPECS} - matched
    if missing:
        raise RuntimeError(f"Could not find tables for: {missing}")

    df = frames[0]
    for other in frames[1:]:
        df = df.join(other, on="date", how="full", coalesce=True)

    return df.sort("date")


def save_parquet(df: pl.DataFrame, output_dir: Path, filename: str = "mkr_miacr.parquet") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.write_parquet(path)
    return path
