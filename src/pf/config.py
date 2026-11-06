"""Конфигурация агента.

Все параметры читаются из переменных окружения (или файла ``.env``).
Значения по умолчанию подобраны так, чтобы агент запускался без настройки:
локальная модель через Ollama, локальный корпус Википедии.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Прочитать переменные из файла ``.env``.

    pydantic-settings читает ``.env`` только для объявленных полей. Но
    агенту нужны и «чужие» переменные — например, ключи к внешним
    MCP-серверам, которые в конфигурации сервера подставляются как
    ``${ПЕРЕМЕННАЯ}``. Поэтому файл читается здесь, целиком.

    Формат — простой ``KEY=VALUE``, как в ``.env.example``: строки с
    ``#`` и пустые строки пропускаются, кавычки снимаются.

    Args:
        path: Путь к файлу. По умолчанию — ``.env`` в рабочем каталоге.

    Returns:
        Словарь «переменная → значение».
    """
    target = path or Path(".env")
    if not target.is_file():
        return {}

    values: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def environment(extra_file: Path | None = None) -> dict[str, str]:
    """Собрать переменные окружения: процесс, затем файл ``.env``.

    Приоритет у процесса: то, что задано в окружении, сильнее значения
    из файла. Иначе правило «скопируйте .env.example в .env» нельзя было
    бы перекрыть одноразовой переменной в терминале.

    Args:
        extra_file: Дополнительный файл с переменными.

    Returns:
        Объединённый словарь «переменная → значение».
    """
    merged = dict(os.environ)
    merged.update(read_env_file(extra_file))
    return merged


class Settings(BaseSettings):
    """Настройки агента."""

    #: Путь к файлу ``.env``, из которого читались переменные.
    _env_file_path: Path = PrivateAttr(default=Path(".env"))

    model_config = SettingsConfigDict(
        env_prefix="PF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Провайдер LLM ───────────────────────────────────────────────────────
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    #: Размышление модели (thinking-режим). У reasoning-моделей
    #: (``qwen3``, ``deepseek-r1``) оно включается по умолчанию: модель
    #: тратит токены на рассуждение и может вернуть его вместо JSON.
    #: Агенту нужен строгий JSON, поэтому режим выключается явно.
    ollama_think: bool = False

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

    #: Домены, с которых разрешено брать страницы. Пустой список — брать
    #: можно отовсюду (кроме правил сайта в robots.txt). Домены пишутся
    #: без ``www.`` — сравнение идёт по нормализованному имени.
    allowed_domains: frozenset[str] = frozenset(
        {
            "ru.wikipedia.org",
            "eprussia.ru",
            "megafon.ru",
            "mts.ru",
            "beeline.ru",
            "tass.ru",
            "rbc.ru",
            "vedomosti.ru",
            "kommersant.ru",
            "rg.ru",
            "interfax.ru",
            "cfo.ru",
            "forbes.ru",
            "vc.ru",
            "habr.com",
        }
    )

    #: Заголовок запроса к внешним сайтам. В нём указан контакт — без
    #: этого Википедия и часть сайтов отвечают отказом.
    user_agent: str = "product-factory-strategist/0.1 (contact@example.org)"

    #: Сколько страниц разбирать за один выход в веб.
    web_fetch_limit: int = 3

    #: Сколько результатов поиска запрашивать у поисковика.
    web_search_limit: int = 5

    #: Сколько подтверждённых оснований считается достаточным. Меньше —
    #: план исследования расширяется хуком.
    min_confirmed_citations: int = 2

    # ── Пути состояния ──────────────────────────────────────────────────────
    state_dir: Path = Path("state")
    journal_dir: Path = Path("journal")

    #: Материалы проекта: свои документы, таблицы и данные. Это третий
    #: слой знаний — он относится к конкретной задаче и не попадает
    #: в инструкции ролей.
    project_dir: Path = Path("project")
    prompts_dir: Path = Path("prompts")
    methods_dir: Path = Path("methods")
    hooks_dir: Path = Path("hooks")
    skills_dir: Path = Path("skills")
    mcp_config_path: Path = Path("mcp/servers.json")

    #: Каталог с MCP-серверами. Серверы лежат в соседнем репозитории и
    #: тянут тяжёлые зависимости, поэтому живут в своём окружении.
    #: Путь задаётся переменной, чтобы конфигурация переносилась на
    #: другую машину: абсолютный путь здесь зашивать нельзя.
    mcp_dir: Path = Path("../mcp-servers")

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

    def environment(self) -> dict[str, str]:
        """Все переменные, доступные агенту: процесс и файл ``.env``.

        Файл берётся из настроек агента, а не из текущего каталога: иначе
        настройки, созданные с другим путём к ``.env``, читали бы не тот
        файл.

        Returns:
            Объединённый словарь «переменная → значение».
        """
        return environment(self._env_file_path)

    def __init__(self, **values: Any) -> None:
        """Создать настройки, запомнив путь к файлу ``.env``.

        pydantic-settings принимает ``_env_file`` в конструкторе, но не
        сохраняет его: в ``model_config`` лежит классовое значение «.env».
        Агенту нужен именно тот файл, который ему передали, — иначе
        настройки с другим путём читали бы не тот файл.

        Args:
            **values: Значения полей и служебные параметры pydantic-settings.
        """
        env_file = values.pop("_env_file", None)
        super().__init__(**values)
        self._env_file_path = Path(env_file) if env_file is not None else Path(".env")


settings = Settings()
