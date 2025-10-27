"""Configuration management using environment variables.

TODO:
- Use pydantic-settings for validation
- Add support for GCP Secret Manager in production
- Add environment-specific overrides (dev/staging/prod)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///./dev.db"

    # GCP
    gcp_project_id: str = "your-project-id"
    gcs_bucket: str = "your-artifacts-bucket"

    # LLM Provider
    llm_provider: Literal["openai", "vertex"] = "openai"
    openai_api_key: str = ""
    vertex_ai_location: str = "us-central1"

    # Application
    log_level: str = "INFO"
    environment: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
