from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class CpiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CPI_", env_file=".env", extra="ignore")

    prices_file: Path = Path("src/test_data/prices.xlsx")
    weights_file: Path = Path("src/test_data/weights.xlsx")
    output_dir: Path = Path("data/cpi")
