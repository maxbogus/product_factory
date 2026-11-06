"""Веб-источники: поиск, когда локальной базы не хватает.

Правило простое: подтверждённым основанием может быть только дословная
цитата из **локального** среза. Внешний источник проходит проверку
дословности, но не проходит проверку «лежит в отобранном срезе» — и
поэтому получает статус «требует проверки», а не «подтверждено».

Это то, что делает агент честным: он может выйти в веб, но не может
выдать добытое из веба за проверенный факт. Пользователь видит пометку.

Поиск идёт по Википедии через API (тот же путь, что у сборщика
корпуса) и по разрешённым доменам через MCP-сервер разбора. Википедия
отвечает 403 на запросы без указания контакта, поэтому заголовок
задаётся явно.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from pf.config import settings
from pf.ground import SupportStatus, verify_quote

#: Заголовок запроса к Википедии: без указания контакта — ответ 403.
USER_AGENT = "product-factory-strategist/0.1 (contact@example.org)"

#: Домены, разрешённые для сбора.
ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "ru.wikipedia.org",
        "eprussia.ru",
        "megafon.ru",
        "www.mts.ru",
        "beeline.ru",
    }
)

#: Разделы Википедии, из которых берутся статьи телеком-тематики.
TELECOM_CATEGORIES: tuple[str, ...] = (
    "Категория:Сотовая связь",
    "Категория:Мобильные ТелеСистемы",
    "Категория:Тарифная политика",
    "Категория:Операторы беспроводной связи",
)


@dataclass(frozen=True, slots=True)
class WebSource:
    """Внешний источник с его происхождением.

    Attributes:
        url: Адрес источника.
        title: Заголовок.
        text: Текст, из которого можно цитировать.
        retrieved_at: Когда получен.
    """

    url: str
    title: str
    text: str
    retrieved_at: str = ""

    @property
    def source_id(self) -> str:
        """Идентификатор источника для ответа агента.

        Returns:
            Идентификатор с пометкой внешнего источника.
        """
        return f"web/{self.title}"


@dataclass(frozen=True, slots=True)
class WebSupport:
    """Результат проверки цитаты во внешнем источнике.

    Attributes:
        status: Статус подтверждения.
        citation: Ссылка для ответа.
        detail: Пояснение для человека.
    """

    status: SupportStatus
    citation: str
    detail: str = ""

    @property
    def is_usable(self) -> bool:
        """Можно ли опираться на это основание.

        Внешний источник не бывает подтверждённым: он требует проверки
        человеком.

        Returns:
            Основание пригодно к использованию.
        """
        return self.status is SupportStatus.CONFIRMED


def is_allowed(url: str) -> bool:
    """Разрешён ли домен для сбора.

    Args:
        url: Адрес.

    Returns:
        Домен разрешён.
    """
    match = re.match(r"https?://([^/]+)", url)
    if match is None:
        return False
    return match.group(1).lower() in ALLOWED_DOMAINS


def search_wikipedia(query: str, limit: int = 5) -> list[str]:
    """Найти статьи Википедии по запросу.

    Args:
        query: Поисковый запрос.
        limit: Сколько заголовков вернуть.

    Returns:
        Список заголовков статей.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
    }
    try:
        response = httpx.get(
            "https://ru.wikipedia.org/w/api.php",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    titles = response.json().get("query", {}).get("search", [])
    return [str(item["title"]) for item in titles]


def fetch_wikipedia(title: str) -> WebSource | None:
    """Загрузить полный текст статьи Википедии.

    Запрос делается по одной статье: при пакетном запросе с
    ``explaintext`` Википедия отдаёт текст только первой статьи.

    Args:
        title: Заголовок статьи.

    Returns:
        Источник либо ``None``, если статьи нет.
    """
    from datetime import UTC, datetime

    params = {
        "action": "query",
        "titles": title,
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
    }
    try:
        response = httpx.get(
            "https://ru.wikipedia.org/w/api.php",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    pages = response.json().get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract") or ""
        if not text:
            continue
        return WebSource(
            url=f"https://ru.wikipedia.org/wiki/{title.replace(' ', '_')}",
            title=str(page.get("title", title)),
            text=text,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    return None


def verify_web_quote(source: WebSource, quote: str) -> WebSupport:
    """Проверить цитату во внешнем источнике.

    Внешний источник проходит ту же проверку дословности, что и
    локальный, но его статус ограничен: он никогда не бывает
    «подтверждено» — только «требует проверки».

    Args:
        source: Внешний источник.
        quote: Цитата из ответа модели.

    Returns:
        Результат проверки.
    """
    support = verify_quote(source.text, quote, source.source_id)
    if support.status is SupportStatus.CONFIRMED:
        return WebSupport(
            status=SupportStatus.NEEDS_REVIEW,
            citation=f"{source.source_id} ({source.url})",
            detail="внешний источник: цитата найдена дословно, требует проверки",
        )
    return WebSupport(
        status=support.status,
        citation=source.source_id,
        detail=support.detail,
    )


def collect_web(query: str, limit: int = 3) -> list[WebSource]:
    """Собрать внешние источники по запросу.

    Args:
        query: Поисковый запрос.
        limit: Сколько источников вернуть.

    Returns:
        Список источников.
    """
    sources: list[WebSource] = []
    seen: set[str] = set()
    for title in search_wikipedia(query, limit=limit * 2):
        if title in seen:
            continue
        source = fetch_wikipedia(title)
        if source is None:
            continue
        seen.add(title)
        sources.append(source)
        if len(sources) >= limit:
            break
    return sources


def as_context(sources: list[WebSource]) -> str:
    """Собрать контекст из внешних источников.

    Args:
        sources: Внешние источники.

    Returns:
        Текст для модели.
    """
    blocks = []
    for source in sources:
        blocks.append(f"[{source.source_id} | {source.url}] {source.text[:1200]}")
    return "\n\n".join(blocks)


def summarize(sources: list[WebSource]) -> list[dict[str, Any]]:
    """Описать источники для журнала.

    Args:
        sources: Внешние источники.

    Returns:
        Список описаний.
    """
    return [
        {
            "source_id": source.source_id,
            "url": source.url,
            "title": source.title,
            "retrieved_at": source.retrieved_at,
            "status": "требует проверки",
        }
        for source in sources
    ]
