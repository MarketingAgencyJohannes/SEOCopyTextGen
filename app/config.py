from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Core
    anthropic_api_key: str = ""
    api_secret_key: str = ""
    log_level: str = "INFO"

    # Agent 1 — YouTube Channel Scraper
    youtube_api_key: str = ""

    # Agent 2 — SERP Analyzer
    serper_api_key: str = ""

    # Agent 3 — Transcriber
    whisper_enabled: bool = True
    whisper_model_size: str = "small"
    # Optional: Webshare.io residential proxy to bypass YouTube IP blocks
    # Free tier at webshare.io (10 proxies, 1 GB/month) is sufficient
    webshare_proxy_username: str = ""
    webshare_proxy_password: str = ""

    # Agent 4 — SEO Copy Generator
    seo_word_target: int = 850
    seo_min_keyword_density: float = 0.008   # 0.8%
    seo_max_keyword_density: float = 0.012   # 1.2%

    # Google Drive output
    google_service_account_json: str = ""
    google_drive_folder_id: str = ""

    # Database
    database_url: str = "sqlite:///./jobs.db"


settings = Settings()
