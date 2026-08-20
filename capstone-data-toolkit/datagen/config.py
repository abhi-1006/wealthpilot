"""Configuration for the capstone data generator.

Reads from environment variables or a local .env file. See .env.example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "openrouter", "ollama", "groq"]


class Settings(BaseSettings):
    """Runtime settings for corpus generation.

    The generator is provider-agnostic by design: the same prompts run against
    Gemini AI Studio (free tier), OpenRouter, or a local Ollama model. Teams
    that hit a rate limit on one provider can switch with a single env var.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DATAGEN_", extra="ignore"
    )

    provider: Provider = "gemini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Reproducibility. Every team that runs with the same seed gets a byte
    # identical corpus, which is what makes a common grading rubric possible.
    seed: int = 42

    # Corpus sizing. The defaults match the sealed-corpus convention used in
    # the internship assignments: small enough to index on a laptop, large
    # enough that retrieval quality actually varies with your chunking choices.
    corpus_docs: int = 50
    intake_records: int = 200
    eval_items: int = 20

    output_dir: Path = Path("./data")

    # Generation controls
    max_retries: int = 4
    request_timeout: int = 120
    temperature: float = 0.9
    concurrency: int = 4


settings = Settings()
