"""Configuration management using environment variables.

Configuration for Wave 2: Multi-LLM support with OpenAI, Vertex AI, and Mistral.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///./dev.db"

    # GCP
    gcp_project_id: str = "your-project-id"
    gcs_bucket: str = "your-artifacts-bucket"

    # LLM Provider Selection
    llm_provider: Literal["openai", "vertex", "mistral"] = "openai"
    llm_model: str | None = None  # Optional: override default model

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-2024-04-09"  # GPT-4.1
    openai_rate_limit_tpm: int = 90000  # Tokens per minute
    openai_rate_limit_tph: int = 5000000  # Tokens per hour

    # Vertex AI (Google Gemini) Configuration
    vertex_ai_project_id: str | None = None
    vertex_ai_location: str = "us-central1"
    vertex_ai_model: str = "gemini-2.0-flash-exp"  # Gemini 2.5
    # Vertex has no strict rate limits by default

    # Mistral Configuration
    mistral_api_key: str = ""
    mistral_model: str = "mistral-medium-latest"  # Mistral medium-3
    mistral_rate_limit_tpm: int = 2000000  # Tokens per minute
    mistral_rate_limit_tph: int = 100000000  # Tokens per hour

    # Weather API (OpenWeatherMap)
    openweather_api_key: str = ""
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"
    openweather_timeout: int = 30  # seconds

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
