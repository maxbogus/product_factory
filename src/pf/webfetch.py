"""Разбор найденной веб-страницы: текст, пригодный для цитирования.

Задача модуля — превратить HTML найденной страницы в **абзацы, из
которых можно процитировать дословно**. Требования те же, что у
локального корпуса (``pf.sources``):

- абзац должен быть достаточно длинным, иначе из него нечего цитировать;
- из текста убирается служебная разметка — скрипты, стили, меню;
- перед разбором проверяется ``robots.txt``: сайт сам говорит, какие
  его части нельзя обходить.

Разбор не доверяет ни одной инструкции со страницы: текст — это данные,
а не правила для агента.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from pf import domain
from pf.config import settings

#: Минимальная длина абзаца, пригодного для цитирования. То же значение,
#: что у локального корпуса.
MIN_PARAGRAPH_CHARS = 60

#: Максимальная длина абзаца в контексте.
MAX_PARAGRAPH_CHARS = 1200

#: Теги, содержимое которых не является текстом статьи.
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg")

#: Служебные классы: по ним верстальщики помечают меню и подвалы.
_BOILERPLATE_CLASS = re.compile(
    r"(nav|menu|sidebar|footer|header|breadcrumb|cookie|banner|comment|share|social|promo)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """Разобранная страница.

    Attributes:
        url: Адрес страницы.
        title: Заголовок страницы.
        paragraphs: Абзацы, пригодные для цитирования.
        retrieved_at: Когда страница получена.
        provider: Канал поиска, который привёл на страницу.
        status: Статус разбора: ``ok`` либо причина отказа.
    """

    url: str
    title: str
    paragraphs: list[str] = field(default_factory=list)
    retrieved_at: str = ""
    provider: str = "bing"
    status: str = "ok"

    @property
    def text(self) -> str:
        """Весь текст страницы, пригодный для цитирования."""
        return "\n".join(self.paragraphs)


def _now() -> str:
    """Метка времени в ISO-формате."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def extract_paragraphs(html: str) -> list[str]:
    """Достать из HTML абзацы, пригодные для цитирования.

    Служебные теги и блоки с «техническими» классами выбрасываются:
    в цитате не должно быть меню и подвала.

    Args:
        html: Текст страницы.

    Returns:
        Список абзацев.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    for tag in soup.find_all(class_=True):
        classes = " ".join(tag.get("class") or [])
        if _BOILERPLATE_CLASS.search(classes):
            tag.decompose()

    paragraphs: list[str] = []
    for tag in soup.find_all(["p", "li", "blockquote", "h1", "h2", "h3", "td"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if len(text) < MIN_PARAGRAPH_CHARS:
            continue
        paragraphs.append(text[:MAX_PARAGRAPH_CHARS])
    if not paragraphs:
        # Страница без абзацев: берём весь видимый текст.
        body = soup.find("body") or soup
        text = re.sub(r"\s+", " ", body.get_text(" ", strip=True)).strip()
        if len(text) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(text[:MAX_PARAGRAPH_CHARS])
    return paragraphs


def extract_title(html: str) -> str:
    """Достать заголовок страницы.

    Args:
        html: Текст страницы.

    Returns:
        Заголовок либо пустая строка.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            return title
    heading = soup.find("h1")
    if heading is not None:
        return re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()
    return ""


def is_robots_allowed(url: str, client: httpx.Client | None = None) -> bool:
    """Разрешает ли сайт обход этой страницы.

    Если ``robots.txt`` недоступен, страница не запрещена явно — но
    агент всё равно не полезет на неё без разрешения домена.

    Args:
        url: Адрес страницы.
        client: Клиент HTTP. По умолчанию — свой.

    Returns:
        Обход разрешён.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        if client is not None:
            parser.read()
        else:
            response = httpx.get(
                robots_url,
                headers={"User-Agent": settings.user_agent},
                timeout=settings.timeout_seconds,
                follow_redirects=True,
            )
            parser.parse(response.text.splitlines())
    except httpx.HTTPError:
        return False
    except Exception:  # noqa: BLE001 — разбор robots.txt не должен ронять сбор
        return False
    return parser.can_fetch(settings.user_agent, url)


def fetch_page(
    url: str,
    *,
    provider: str = "bing",
    client: httpx.Client | None = None,
) -> FetchedPage:
    """Скачать и разобрать страницу.

    Args:
        url: Адрес страницы.
        provider: Канал поиска, который привёл на страницу.
        client: Клиент HTTP. По умолчанию — свой.

    Returns:
        Разобранная страница; при отказе — страница со статусом.
    """
    owns_client = client is None
    http = client or httpx.Client(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.timeout_seconds,
        follow_redirects=True,
    )
    try:
        if not is_robots_allowed(url, client=http):
            return FetchedPage(
                url=url,
                title="",
                retrieved_at=_now(),
                provider=provider,
                status="запрещено robots.txt",
            )
        response = http.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return FetchedPage(
            url=url,
            title="",
            retrieved_at=_now(),
            provider=provider,
            status=f"страница недоступна: {exc.__class__.__name__}",
        )
    finally:
        if owns_client:
            http.close()

    html = response.text
    paragraphs = extract_paragraphs(html)
    return FetchedPage(
        url=str(response.url),
        title=extract_title(html),
        paragraphs=paragraphs,
        retrieved_at=_now(),
        provider=provider,
        status="ok" if paragraphs else "текст не найден",
    )


def fetch_pages(
    hits: list[Any],
    *,
    limit: int = 3,
    allowed_domains: frozenset[str] | None = None,
) -> list[FetchedPage]:
    """Скачать и разобрать несколько найденных страниц.

    Args:
        hits: Результаты поиска (``pf.web.SearchHit``).
        limit: Сколько страниц разобрать.
        allowed_domains: Разрешённые домены. По умолчанию — настройки.

    Returns:
        Список разобранных страниц без пустых.
    """
    domains = (
        allowed_domains
        if allowed_domains is not None
        else settings.allowed_domains | domain.web_domains()
    )
    pages: list[FetchedPage] = []
    for hit in hits:
        if len(pages) >= limit:
            break
        host = urlparse(hit.url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if domains and host not in domains:
            continue
        page = fetch_page(hit.url, provider=getattr(hit, "provider", "bing"))
        if page.paragraphs:
            pages.append(page)
    return pages


def as_context(pages: list[FetchedPage]) -> str:
    """Собрать контекст из разобранных страниц.

    Args:
        pages: Разобранные страницы.

    Returns:
        Текст для модели.
    """
    blocks = []
    for page in pages:
        blocks.append(f"[web/{page.title or page.url} | {page.url}] {page.text[:1200]}")
    return "\n\n".join(blocks)


def summarize(pages: list[FetchedPage]) -> list[dict[str, Any]]:
    """Описать страницы для журнала.

    Args:
        pages: Разобранные страницы.

    Returns:
        Список описаний.
    """
    return [
        {
            "source_id": f"web/{page.title or page.url}",
            "url": page.url,
            "title": page.title,
            "retrieved_at": page.retrieved_at,
            "provider": page.provider,
            "paragraphs": len(page.paragraphs),
            "status": "требует проверки",
        }
        for page in pages
    ]
