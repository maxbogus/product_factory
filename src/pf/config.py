"""Конфигурация агента.

Все параметры читаются из переменных окружения (или файла ``.env``).
Значения по умолчанию подобраны так, чтобы агент запускался без настройки:
локальная модель через Ollama, локальный корпус Википедии.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Лимит ответа для облачных (не локальных) моделей. JSON стратегии
#: длинный: на локальном лимите 2400-3200 токенов облачная модель
#: обрывает его, и разбор падает на «JSON не закрыт». Локальная модель
#: остаётся на значении из настроек — ей этого хватает.
CLOUD_MAX_TOKENS = 8192


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

    #: Окно контекста локальной модели. Ollama по умолчанию берёт 4096 и
    #: **обрезает** промпт: на сохранённых журналах 56 шагов упирались
    #: ровно в 4096 токенов, и начало контекста отбрасывалось молча.
    #: 12288 выбрано замером (`docs/decisions.md`, D-20): при 8192 промпты
    #: ``analyst`` (9433 токена) и ``strategy`` (9560) не влезали, при
    #: 12288 не упирается ни один шаг из девяти, а видеопамяти занято
    #: 6,7 ГБ из 8,1 ГБ. Цена — примерно +22 % времени прогона. На карте
    #: меньше 8 ГБ верните 8192: окно больше — KV-кэш растёт линейно и
    #: часть слоёв уходит на CPU.
    ollama_num_ctx: int = 12288

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # ── Параметры генерации ─────────────────────────────────────────────────
    temperature: float = 0.2
    max_tokens: int = 3200
    timeout_seconds: int = 180

    # ── Проверка оснований ──────────────────────────────────────────────────
    #: Абсолютный допуск для чисел от единицы: «77,3» подтверждает «77,26».
    #: Для малых величин он опасен (0,1 — половина от 0,2), поэтому там
    #: работает только относительный допуск.
    metric_abs_tolerance: float = 0.1

    #: Относительный допуск: покрывает точность, в которой числа
    #: показываются в блоке РАСЧЁТЫ (``%.4g``).
    metric_rel_tolerance: float = 5e-4

    #: Порог покрытия утверждения словами абзаца. Сообщается как признак;
    #: решением стал порог степени поддержки (`claim_support_threshold`).
    claim_min_overlap: float = 0.8

    #: Порог степени поддержки утверждения полями (`pf.canonical`): ниже него
    #: утверждение не подтверждается. Заморожен по замеру на 60 размеченных
    #: парах (`eval/pairs/out_of_place.jsonl`): верх полосы, которую дают
    #: повторные фолды, округлённый вверх. Подбирается `make tune-threshold`,
    #: стережёт `tests/test_canonical.py`.
    claim_support_threshold: float = 0.17

    #: Просить ли модель уложиться в лимит ответа. Лимит уже режет
    #: генерацию, но модель об этом не знает, и JSON приходит обрезанным.
    answer_limit_hint: bool = True

    #: Сколько символов ограничений задачи попадает в запрос. Длинный
    #: блок вытесняет схему ответа, и модель возвращает структуру входа.
    max_constraints_chars: int = 600

    # ── Лексикон и распознание незнакомых слов ──────────────────────────────
    #: Каталог словаря домена: субъекты, термины, шаблоны заголовков.
    #: Отдельная настройка, а не ``knowledge_dir``: домен — данные проекта по
    #: ссылке, а не накопленное хранилище. Тесты уводят ``knowledge_dir`` в
    #: ``tmp_path`` (`D-29`), и домен, лежащий там, исчез бы во всех проверках.
    domain_dir: Path = Path("knowledge/domain")

    #: Каталог словаря близости и очереди распознания.
    lexicon_dir: Path = Path("knowledge/lexicon")

    #: Включено ли накопление незнакомых слов. Выключение полезно для
    #: замеров: очередь — это след работы, а не часть проверки.
    lexicon_enabled: bool = True

    #: Минимальная длина слова для поиска опечатки.
    oov_min_length: int = 4

    #: Сколько примеров контекста хранить на слово в очереди.
    oov_max_contexts: int = 3

    #: Допустимое расстояние Дамерау—Левенштейна для «похоже на опечатку».
    typo_max_distance: int = 1

    #: Сколько незнакомых слов разбирать за один вызов модели.
    lexicon_enrich_limit: int = 50

    # ── Перевод внешних материалов ──────────────────────────────────────────
    #: Каталог кэша переводов. Перевод недетерминирован, поэтому кэш — условие
    #: воспроизводимости прогона: `make test` и `make replay` в модель не ходят.
    #: В git кэш не попадает: это след работы, а не часть поставки.
    translations_dir: Path = Path("knowledge/translations")

    #: Предел длины куска при переводе. Длинный кусок модель пересказывает и
    #: теряет числа — поэтому текст режется по абзацам и предложениям.
    translate_max_chars: int = 1200

    # ── Источники ───────────────────────────────────────────────────────────
    wiki_dir: Path = Path("data/wiki")
    source_top_k: int = 6
    source_max_paragraphs: int = 6

    #: Читать ли сырые выгрузки (``knowledge/raw/<дата>/``) как материалы.
    #: Выгрузка — открытые данные, скачанные когда-то и кем-то: цитата из неё
    #: проверяется дословно, но происхождение не проверено, поэтому основание
    #: не поднимается выше «требует проверки» (``docs/decisions.md``, D-29).
    include_knowledge_raw: bool = True

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
    #: Здесь — политика выборки: общие издания. Сайты брендов принадлежат
    #: домену проекта (`knowledge/domain/entities.yaml`) и складываются с этим
    #: списком в `pf.web.allowed_domains`.
    allowed_domains: frozenset[str] = frozenset(
        {
            "ru.wikipedia.org",
            "eprussia.ru",
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

    #: Хранилище знаний: сырые выгрузки открытых данных и манифест
    #: с provenance. Отдельно от материалов проекта: здесь лежит то,
    #: что агент скачал сам, с датой и хешем версии.
    knowledge_dir: Path = Path("knowledge")

    #: Отчёты по прогонам: markdown и JSON. Коммитятся: по ним видно,
    #: что именно измерялось.
    reports_dir: Path = Path("reports/sessions")
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

    @property
    def effective_max_tokens(self) -> int:
        """Лимит длины ответа для выбранного провайдера.

        У локальной модели — значение из настроек. У облачной — не меньше
        ``CLOUD_MAX_TOKENS``: там ответ длиннее, и на локальном лимите
        JSON стратегии приходит обрезанным. Если в настройках задано
        больше, берётся большее значение.
        """
        if self.is_local:
            return self.max_tokens
        return max(self.max_tokens, CLOUD_MAX_TOKENS)

    @property
    def effective_num_ctx(self) -> int:
        """Окно контекста для локальной модели; ``0`` — окно задаёт провайдер.

        Окно — это промпт **и** ответ вместе, поэтому оно не может быть
        меньше лимита ответа: иначе на промпт не остаётся места. Валидатор
        настроек такое сочетание отклоняет, а ``max`` здесь — страховка для
        настроек, собранных в обход валидации.

        Облачному провайдеру окно не передаётся: у него своё, и навязывать
        ему локальное число бессмысленно.
        """
        if not self.is_local:
            return 0
        return max(self.ollama_num_ctx, self.effective_max_tokens)

    @model_validator(mode="after")
    def _check_context_window(self) -> Settings:
        """Окно обязано оставлять место ответу.

        Тихая обрезка промпта — ровно та ошибка, которую трудно заметить:
        модель отвечает, а часть контекста уже отброшена. Поэтому
        несовместимые значения — ошибка настроек, видимая сразу.

        Returns:
            Сами настройки, если сочетание допустимо.

        Raises:
            ValueError: Лимит ответа не оставляет места в окне контекста.
        """
        if self.is_local and self.max_tokens >= self.ollama_num_ctx:
            raise ValueError(
                f"PF_MAX_TOKENS={self.max_tokens} не оставляет места в окне контекста "
                f"PF_OLLAMA_NUM_CTX={self.ollama_num_ctx}: окно — это промпт и ответ "
                "вместе. Поднимите PF_OLLAMA_NUM_CTX или уменьшите PF_MAX_TOKENS."
            )
        return self

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
