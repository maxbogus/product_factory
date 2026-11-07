"""Веб-источники: поиск в интернете и разбор найденных страниц.

Правило простое: подтверждённым основанием может быть только дословная
цитата из **локального** среза. Внешний источник проходит проверку
дословности, но не проходит проверку «лежит в отобранном срезе» — и
поэтому получает статус «требует проверки», а не «подтверждено».

Это то, что делает агент честным: он может выйти в веб, но не может
выдать добытое из веба за проверенный факт. Пользователь видит пометку.

Поиск идёт тремя каналами, каждый из которых умеет отвалиться молча:

- **Википедия** — через API (тот же путь, что у сборщика корпуса);
- **Bing web** — HTML-выдача, реальные ссылки достаются из
  base64-редиректов;
- **Bing News** — RSS, где ссылки и заголовки уже готовы.

Разбор найденной страницы — в ``pf.webfetch``: robots.txt проверяется,
HTML превращается в абзацы, пригодные для цитирования.

Википедия отвечает 403 на запросы без указания контакта, поэтому
заголовок задаётся явно.
"""

from __future__ import annotations

import base64
import html
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from pf.config import settings
from pf.corpus import DOMAIN_TERMS
from pf.ground import SupportStatus, mark_external, verify_quote

#: Заголовок запроса к Википедии: без указания контакта — ответ 403.
USER_AGENT = "product-factory-strategist/0.1 (contact@example.org)"


def allowed_domains() -> frozenset[str]:
    """Домены, с которых разрешено брать страницы.

    Returns:
        Список доменов без ``www.``.
    """
    return settings.allowed_domains


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
    host = match.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host in allowed_domains()


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


#: Блоки результатов в HTML-выдаче Bing.
_BING_RESULT = re.compile(r'<li class="b_algo".*?</li>', re.DOTALL)

#: Заголовок результата.
_BING_TITLE = re.compile(r"<h2>(.*?)</h2>", re.DOTALL)

#: Сниппет результата.
_BING_SNIPPET = re.compile(r'<div class="b_caption".*?<p[^>]*>(.*?)</p>', re.DOTALL)

#: Ссылка внутри заголовка.
_BING_HREF = re.compile(r'href="([^"]+)"')


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Результат поиска: ссылка, заголовок, сниппет.

    Attributes:
        url: Адрес найденной страницы.
        title: Заголовок результата.
        snippet: Краткое описание из выдачи.
        provider: Канал поиска, который нашёл результат.
    """

    url: str
    title: str
    snippet: str = ""
    provider: str = "bing"

    @property
    def domain(self) -> str:
        """Домен найденной страницы без ``www.``."""
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host


def _strip_tags(fragment: str) -> str:
    """Убрать HTML-разметку из фрагмента.

    Args:
        fragment: Фрагмент разметки.

    Returns:
        Текст без тегов.
    """
    without_tags = re.sub(r"<[^>]+>", " ", fragment)
    entities = (
        ("&amp;", "&"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
    )
    for entity, replacement in entities:
        without_tags = without_tags.replace(entity, replacement)
    return re.sub(r"\s+", " ", without_tags).strip()


def decode_bing_url(href: str) -> str:
    """Достать настоящий адрес из редиректа Bing.

    Bing оборачивает ссылку в ``/ck/a?...&u=a1<base64url>``. Без
    декодирования в выдаче лежит адрес самой выдачи, а не сайта.

    Args:
        href: Значение ``href`` из выдачи.

    Returns:
        Настоящий адрес страницы; если это не редирект — исходная строка.
    """
    href = html.unescape(href)
    if "/ck/a" not in href:
        return href
    raw = parse_qs(urlparse(href).query).get("u", [""])[0]
    if not raw.startswith("a1"):
        return href
    payload = raw[2:]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload).decode("utf-8", "replace")
    except (ValueError, UnicodeDecodeError):
        return href
    return decoded if decoded.startswith("http") else href


def _search_headers() -> dict[str, str]:
    """Заголовки запроса к поисковику.

    Без ``Accept-Language`` выдача приходит не на том языке и по другим
    правилам — приходит мусор. Поэтому язык задан явно.

    Returns:
        Заголовки запроса.
    """
    return {
        "User-Agent": _browser_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
    }


def _browser_agent() -> str:
    """Заголовок запроса для поисковика.

    Returns:
        User-Agent, который выдача принимает.
    """
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


def search_bing_web(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти страницы в веб-выдаче Bing.

    Внимание: выдача Bing по HTTP без браузера нестабильна — часть
    запросов возвращает результаты из чужого кэша. Поэтому канал
    считается запасным и не является основным: сначала пробуются
    RSS-ленты и новости, а этот — в конце.

    Args:
        query: Поисковый запрос.
        limit: Сколько результатов вернуть.

    Returns:
        Список результатов; пустой список, если выдача недоступна.
    """
    url = (
        "https://www.bing.com/search?q="
        + quote_plus(query)
        + f"&count={max(1, min(limit * 2, 30))}"
    )
    try:
        response = httpx.get(
            url,
            headers=_search_headers(),
            timeout=settings.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    hits: list[SearchHit] = []
    seen: set[str] = set()
    for block in _BING_RESULT.findall(response.text):
        title_match = _BING_TITLE.search(block)
        if title_match is None:
            continue
        href_match = _BING_HREF.search(title_match.group(1))
        if href_match is None:
            continue
        # В выдаче амперсанды экранированы: без unquote разбор параметров
        # ломается и достаётся не тот адрес.
        target = decode_bing_url(unquote(href_match.group(1)))
        if not target.startswith("http") or target in seen:
            continue
        snippet_match = _BING_SNIPPET.search(block)
        seen.add(target)
        hits.append(
            SearchHit(
                url=target,
                title=_strip_tags(title_match.group(1)),
                snippet=_strip_tags(snippet_match.group(1)) if snippet_match else "",
                provider="bing",
            )
        )
        if len(hits) >= limit:
            break
    return hits


def _unwrap_news_link(link: str) -> str:
    """Достать настоящий адрес из ссылки новостей Bing.

    Args:
        link: Ссылка из RSS.

    Returns:
        Адрес страницы либо пустая строка.
    """
    if "apiclick.aspx" not in link:
        return link
    raw = parse_qs(urlparse(link).query).get("url", [""])[0]
    if not raw:
        return ""
    if raw.startswith("http"):
        return raw
    return unquote(raw)


def search_bing_news(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти новости через RSS-выдачу Bing.

    RSS удобнее HTML: заголовки и ссылки уже готовы, разбирать ничего не
    надо. Ссылки ведут через ``apiclick.aspx``, поэтому берём параметр
    ``url``, где лежит настоящий адрес.

    Args:
        query: Поисковый запрос.
        limit: Сколько результатов вернуть.

    Returns:
        Список результатов; пустой список, если канал недоступен.
    """
    url = (
        "https://www.bing.com/news/search?q="
        + quote_plus(query)
        + f"&format=rss&count={max(1, min(limit * 2, 30))}"
    )
    hits: list[SearchHit] = []
    for _attempt in range(CHANNEL_RETRIES):
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": _browser_agent()},
                timeout=settings.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return hits

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return hits

        for item in root.findall("./channel/item"):
            target = _unwrap_news_link(item.findtext("link") or "")
            if not target:
                continue
            hits.append(
                SearchHit(
                    url=target,
                    title=_strip_tags(item.findtext("title") or ""),
                    snippet=_strip_tags(item.findtext("description") or ""),
                    provider="bing_news",
                )
            )
            if len(hits) >= limit:
                return hits

        # Пустой ответ — не приговор: канал отдаёт результаты нестабильно.
        # Повторяем попытку, прежде чем считать канал молчащим.
        if hits:
            break
    return hits


#: RSS-ленты разрешённых изданий. Ленты надёжнее поисковой выдачи:
#: в них лежат настоящие статьи с чистыми адресами, и они не отдают
#: мусор из кэша. Отдельный поиск по интернету не нужен — релевантные
#: заголовки отбираются по ключевым словам запроса.
RSS_FEEDS: tuple[tuple[str, str], ...] = (
    ("tass.ru", "https://tass.ru/rss/v2.xml"),
    ("vedomosti.ru", "https://www.vedomosti.ru/rss/news"),
    ("interfax.ru", "https://www.interfax.ru/rss.asp"),
    ("rg.ru", "https://rg.ru/xml/index.xml"),
    ("kommersant.ru", "https://www.kommersant.ru/RSS/news.xml"),
)

#: Открытый XML-документ Банка России с ключевой ставкой. Без ключей
#: и без авторизации — официальный источник макроэкономических данных.
CBR_KEY_RATE_URL = (
    "https://www.cbr.ru/scripts/xml_dynamic.asp"
    "?date_req1=01/01/2026&date_req2=31/12/2026&VAL_NM_RQ=R01235"
)

#: Лента Google Trends: что ищут и что в тренде. Отвечает без ключа —
#: в отличие от Wordstat, где нужен доступ к API.
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=RU"

#: Сколько раз повторять запрос к каналу, который ответил пустотой.
#: Bing News отдаёт результаты нестабильно: один и тот же запрос то
#: даёт статью, то пустоту. Без повтора агент молча терял единственный
#: релевантный источник.
CHANNEL_RETRIES = 2

#: Слова, по которым заголовок новости считается относящимся к запросу.
_STOP_WORDS = frozenset(
    {
        "и",
        "в",
        "на",
        "с",
        "по",
        "для",
        "от",
        "не",
        "что",
        "как",
        "это",
        "поэтому",
        "также",
        "или",
    }
)


#: Слова, которые однозначно указывают на предметную область. Слово
#: «оператор» само по себе ничего не значит — оператор дронов тоже
#: оператор. А вот «отток» или «абонент» — про нашу тему.
_STRONG_TERMS = frozenset(
    {
        "отток",
        "оттока",
        "абонент",
        "абонента",
        "абонентов",
        "абонентский",
        "удержание",
        "удержания",
        "тариф",
        "тарифы",
        "тарифов",
        "телеком",
        "телекоммуникация",
        "связь",
        "мобильный",
        "мобильная",
        "мобильных",
        "оператор",
        "оператора",
        "операторов",
        "подписка",
        "подписки",
        "мегафон",
        "мтс",
        "билайн",
        "ростелеком",
        "телефония",
        "интернет",
        "роуминг",
        "трафик",
        "симкар",
        "сим-карта",
    }
)

#: Основы тех же слов — сравнение идёт по основе, чтобы «отток»,
#: «оттока» и «оттоком» считались одним словом.
_STRONG_STEMS = frozenset(
    {
        "симкар",
        "мобильн",
        "абонент",
        "отток",
        "удержан",
        "тариф",
        "телеком",
        "телефон",
        "подписк",
        "роуминг",
        "трафик",
        "мегафон",
        "мтс",
        "билайн",
        "ростелеком",
        "оператор",
        "связ",
        "интернет",
    }
)


def _stem(word: str) -> str:
    """Основа слова для сопоставления словоформ.

    Запрос «мобильных» должен находить «мобильной» и «мобильном».
    Полная морфология не нужна: достаточно отбросить окончание.

    Args:
        word: Слово в нижнем регистре.

    Returns:
        Основа для сравнения.
    """
    if len(word) <= 5:
        return word
    return word[:-2]


def _query_terms(query: str) -> set[str]:
    """Ключевые слова запроса.

    Args:
        query: Поисковый запрос.

    Returns:
        Основы слов длиной не менее трёх символов без стоп-слов.
    """
    words = re.findall(r"[а-яёa-z]{3,}", query.lower())
    return {_stem(word) for word in words if word not in _STOP_WORDS}


def _stem_set(terms: Iterable[str]) -> set[str]:
    """Основы слов набора.

    Args:
        terms: Слова или основы.

    Returns:
        Множество основ.
    """
    return {_stem(word) for word in terms}


def _matches_terms(text: str, terms: set[str]) -> bool:
    """Подходит ли текст по теме.

    Одного общего слова мало: запрос «тарифы операторов связи» не должен
    подходить к новости про «операторов дронов». Поэтому нужно либо
    совпадение сильного слова, либо два и более обычных.

    Args:
        text: Текст для проверки.
        terms: Основы ключевых слов.

    Returns:
        Текст подходит по теме.
    """
    stems = {_stem(word) for word in re.findall(r"[а-яёa-z]{3,}", text.lower())}
    strong = {term for term in terms if _stem(term) in _STRONG_STEMS}
    if strong & stems:
        return True
    return len(terms & stems) >= 2


def search_rss_feeds(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти материалы в RSS-лентах разрешённых изданий.

    Ленты отдают свежие материалы с чистыми адресами. Отбор по теме —
    по ключевым словам запроса в заголовке и описании.

    Args:
        query: Поисковый запрос.
        limit: Сколько результатов вернуть.

    Returns:
        Список результатов; пустой список, если ленты недоступны.
    """
    terms = _query_terms(query)
    if not terms:
        return []
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for domain, feed_url in RSS_FEEDS:
        if len(hits) >= limit:
            break
        try:
            response = httpx.get(
                feed_url,
                headers=_search_headers(),
                timeout=settings.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except (httpx.HTTPError, ET.ParseError):
            continue
        for item in root.findall("./channel/item"):
            title = _strip_tags(item.findtext("title") or "")
            summary = _strip_tags(item.findtext("description") or "")
            link = item.findtext("link") or ""
            if not link or link in seen:
                continue
            if not _matches_terms(f"{title} {summary}", terms):
                continue
            seen.add(link)
            hits.append(
                SearchHit(
                    url=link,
                    title=title,
                    snippet=summary,
                    provider=f"rss:{domain}",
                )
            )
            if len(hits) >= limit:
                return hits
    return hits


def parse_cbr_key_rate(xml_text: str) -> list[dict[str, str]]:
    """Разобрать XML-документ Банка России с ключевой ставкой.

    Документ приходит в кодировке windows-1251 и содержит ежедневные
    значения. Разбор вынесен отдельно от запроса в сеть: его можно
    проверить на готовом документе, не выходя в интернет.

    Args:
        xml_text: Текст XML-документа.

    Returns:
        Список записей: дата и значение в процентах.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    rows: list[dict[str, str]] = []
    for record in root.findall(".//Record"):
        date = record.get("Date")
        value = record.findtext("VunitRate")
        if date and value:
            rows.append({"дата": date, "значение_процентов": value.replace(",", ".")})
    return rows


def search_cbr(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти данные Банка России по макроэкономике.

    Открытый XML без ключей и авторизации. Канал отвечает на вопросы о
    стоимости денег — а она влияет на решение абонента отключить
    дополнительные услуги или сменить тариф.

    Args:
        query: Поисковый запрос.
        limit: Сколько результатов вернуть.

    Returns:
        Список найденных показателей; пустой список, если канал недоступен.
    """
    terms = _query_terms(query)
    if terms and not _matches_terms("ключевая ставка Банка России", terms):
        return []

    try:
        response = httpx.get(
            CBR_KEY_RATE_URL,
            headers=_search_headers(),
            timeout=settings.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    response.encoding = "windows-1251"
    rows = parse_cbr_key_rate(response.text)
    if not rows:
        return []

    hits: list[SearchHit] = []
    for row in rows[:limit]:
        date = row["дата"]
        value = row["значение_процентов"]
        hits.append(
            SearchHit(
                url=f"{CBR_KEY_RATE_URL}&date={date}",
                title=f"Ключевая ставка Банка России на {date}: {value} %",
                snippet=f"Ключевая ставка: {value} % годовых",
                provider="cbr",
            )
        )
    return hits


def search_trends(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти тренды Google по региону.

    Лента отвечает без ключа и без OAuth — в отличие от Wordstat, где
    нужен доступ к API. Показывает, что в тренде: это ранний сигнал о
    сдвиге в интересе аудитории, ещё до того, как о нём написали
    новости.

    Args:
        query: Поисковый запрос. Используется для отбора по теме.
        limit: Сколько результатов вернуть.

    Returns:
        Список трендов; пустой список, если канал недоступен.
    """
    try:
        response = httpx.get(
            GOOGLE_TRENDS_RSS,
            headers=_search_headers(),
            timeout=settings.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except (httpx.HTTPError, ET.ParseError):
        return []

    terms = _query_terms(query)
    # Тренды отдаются по всей стране, а не по запросу: поиск по конкретному
    # ключевому слову требует API и упирается в лимит. Поэтому отбор
    # идёт по словам запроса **и** по доменным терминам: так канал
    # отвечает на вопрос «тема сейчас в тренде?» хотя бы для
    # предметной области агента.
    domain_terms = _stem_set(DOMAIN_TERMS)
    hits: list[SearchHit] = []
    for item in root.findall("./channel/item"):
        title = _strip_tags(item.findtext("title") or "")
        if not title:
            continue
        traffic = _strip_tags(item.findtext("ht:approx_traffic") or "")
        stems = _stem_set(_query_terms(title))
        if terms and not _matches_terms(title, terms) and not (stems & domain_terms):
            continue
        hits.append(
            SearchHit(
                url=item.findtext("ht:newsItemUrl") or "",
                title=title,
                snippet=f"Тренд в Google{', ' + traffic if traffic else ''}",
                provider="google_trends",
            )
        )
        if len(hits) >= limit:
            break
    return hits


def search_web(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти страницы в интернете по всем доступным каналам.

    Порядок каналов отражает их надёжность:

    1. **Bing News** — новостная выдача: отдаёт настоящие статьи с
       чистыми адресами и стабильно отвечает на русские запросы;
    2. **RSS-ленты** — свежие материалы изданий, отобранные по теме;
    3. **Bing web** — запасной канал. По HTTP без браузера выдача Bing
       частично отдаёт мусор из кэша, поэтому её результаты идут
       последними и только если другие каналы молчат.

    Упавший канал не мешает остальным: это деградация, а не отказ.

    Args:
        query: Поисковый запрос.
        limit: Сколько результатов вернуть.

    Returns:
        Список найденных страниц без повторов.
    """
    hits: list[SearchHit] = []
    seen: set[str] = set()
    terms = _query_terms(query)
    for finder in (
        search_bing_news,
        search_rss_feeds,
        search_cbr,
        search_trends,
        search_bing_web,
    ):
        for hit in finder(query, limit=limit):
            if hit.url in seen:
                continue
            # Канал может вернуть что угодно: Bing web по HTTP без
            # браузера иногда отдаёт результаты из чужого кэша. Поэтому
            # релевантность проверяется здесь, на общем уровне: результат
            # не по теме в источники не попадает.
            if terms and not _matches_terms(f"{hit.title} {hit.snippet}", terms):
                continue
            seen.add(hit.url)
            hits.append(hit)
            if len(hits) >= limit:
                return hits
    return hits


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
    limited = mark_external(support)
    if limited.status is SupportStatus.NEEDS_REVIEW:
        return WebSupport(
            status=limited.status,
            citation=f"{source.source_id} ({source.url})",
            detail=limited.detail,
        )
    return WebSupport(
        status=limited.status,
        citation=source.source_id,
        detail=limited.detail,
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
