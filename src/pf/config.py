"""Конфигурация агента.

Все параметры читаются из переменных окружения (или файла ``.env``).
Значения по умолчанию подобраны так, чтобы агент запускался без настройки:
локальная модель через Ollama, локальный корпус Википедии.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки агента."""

    model_config = SettingsConfigDict(
        env_prefix="PF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Провайдер LLM ───────────────────────────────────────────────────────
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # ── Параметры генерации ─────────────────────────────────────────────────
    temperature: float = 0.2
    max_tokens: int = 2400
    timeout_seconds: int = 180

    # ── Источники ───────────────────────────────────────────────────────────
    wiki_dir: Path = Path("data/wiki")
    source_top_k: int = 6
    source_max_paragraphs: int = 6

    #: Предел расхода одной сессии в условных единицах. Ноль — без предела.
    max_cost_units: float = 0.0

    #: Искать в вебе, когда локальных оснований не хватает. По умолчанию
    #: выключено: демо должно работать без сети.
    enable_web: bool = False

    #: Сколько подтверждённых оснований считается достаточным, чтобы не
    #: идти в веб.
    web_min_citations: int = 2

    #: Сколько подтверждённых оснований считается достаточным. Меньше —
    #: план исследования расширяется хуком.
    min_confirmed_citations: int = 2

    # ── Пути состояния ──────────────────────────────────────────────────────
    state_dir: Path = Path("state")
    journal_dir: Path = Path("journal")
    prompts_dir: Path = Path("prompts")
    methods_dir: Path = Path("methods")
    hooks_dir: Path = Path("hooks")
    skills_dir: Path = Path("skills")
    mcp_config_path: Path = Path("mcp/servers.json")

    @property
    def effective_base_url(self) -> str:
        """Базовый URL API выбранного провайдера."""
        if self.llm_provider == "ollama":
            return self.ollama_base_url
        return self.llm_base_url

    @property
    def effective_model(self) -> str:
        """Имя модели выбранного провайдера."""
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.llm_model

    @property
    def is_local(self) -> bool:
        """Работает ли агент на локальной модели (без внешних ключей)."""
        return self.llm_provider == "ollama"


settings = Settings()
