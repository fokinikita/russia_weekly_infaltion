from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class MkrConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MKR_", env_file=".env", extra="ignore")

    # Start of historical series — override via MKR_FROM_DATE env var or .env
    from_date: str = "01.01.2022"
    base_url: str = "https://www.cbr.ru/hd_base/mkr/mkr_base/"
    output_dir: Path = Path("data/mkr")
