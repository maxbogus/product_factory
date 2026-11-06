"""Проверки веб-источников.

Главное правило: внешний источник **никогда** не становится подтверждённым
основанием. Он проходит проверку дословности — но не проходит проверку
«лежит в отобранном локальном срезе», поэтому его статус «требует
проверки».

Если это правило ослабнет, агент начнёт выдавать добытое из веба за
проверенный факт, и весь механизм честности перестанет работать.
"""

from __future__ import annotations

import httpx
import pytest

from pf.ground import SupportStatus
from pf.web import (
    RSS_FEEDS,
    SearchHit,
    WebSource,
    allowed_domains,
    decode_bing_url,
    is_allowed,
    search_bing_news,
    search_web,
    verify_web_quote,
)


def _source(text: str) -> WebSource:
    """Собрать внешний источник.

    Args:
        text: Текст источника.

    Returns:
        Источник.
    """
    return WebSource(
        url="https://ru.wikipedia.org/wiki/Тест",
        title="Тест",
        text=text,
        retrieved_at="2026-11-06T10:00:00+00:00",
    )


def test_allowed_domains_are_restricted() -> None:
    """Собирать можно только по разрешённым доменам."""
    assert is_allowed("https://ru.wikipedia.org/wiki/Мегафон")
    assert is_allowed("https://eprussia.ru/news/1")
    assert not is_allowed("https://example.com/page")
    assert not is_allowed("http://ru.wikipedia.org.evil.com/page")


def test_allowed_domains_list_is_not_empty() -> None:
    """Список разрешённых доменов задан."""
    domains = allowed_domains()
    assert domains
    assert "ru.wikipedia.org" in domains


def test_www_prefix_is_normalized() -> None:
    """``www.`` не делает домен другим."""
    assert is_allowed("https://www.mts.ru/tariffs")
    assert is_allowed("https://mts.ru/tariffs")
    assert not is_allowed("https://example.com/page")


def test_web_quote_is_never_confirmed() -> None:
    """Внешняя цитата не бывает подтверждённой.

    Дословная цитата из веба требует проверки человеком.
    """
    source = _source("Компания обслужила 77,26 миллиона абонентов.")
    support = verify_web_quote(source, "обслужила 77,26 миллиона абонентов")
    assert support.status is SupportStatus.NEEDS_REVIEW
    assert not support.is_usable


def test_web_quote_reports_missing_quote() -> None:
    """Цитаты, которой нет, нет и в вебе."""
    source = _source("Компания обслужила 77,26 миллиона абонентов.")
    support = verify_web_quote(source, "удержание упало на 8 процентных пунктов")
    assert support.status is SupportStatus.NO_DATA
    assert not support.is_usable


def test_web_citation_contains_url() -> None:
    """В ссылке на внешний источник есть адрес."""
    source = _source("Текст для цитаты.")
    support = verify_web_quote(source, "Текст для цитаты")
    assert source.url in support.citation


def test_web_source_id_is_marked_as_external() -> None:
    """Идентификатор внешнего источника помечен."""
    source = _source("Текст.")
    assert source.source_id.startswith("web/")


@pytest.mark.parametrize("domain", sorted(allowed_domains()))
def test_every_allowed_domain_is_valid(domain: str) -> None:
    """Все разрешённые домены имеют осмысленный вид."""
    assert "." in domain
    assert " " not in domain
    assert not domain.startswith("www.")


def test_bing_redirect_is_decoded() -> None:
    """Настоящий адрес достаётся из редиректа Bing.

    Без этого в выдаче лежит адрес самой выдачи, а не сайта — и разбор
    страницы бессмыслен.
    """
    # Bing кладёт в параметр "u" префикс "a1" и base64url адреса
    encoded = "a1" + "aHR0cHM6Ly9leGFtcGxlLmNvbS9wYWdl"
    href = f"https://www.bing.com/ck/a?!&&p=abc&u={encoded}"
    assert decode_bing_url(href) == "https://example.com/page"


def test_plain_url_is_not_decoded() -> None:
    """Обычная ссылка не трогается."""
    assert decode_bing_url("https://example.com/page") == "https://example.com/page"


def test_search_hit_reports_domain() -> None:
    """Домен результата известен без ``www.``."""
    hit = SearchHit(url="https://www.mts.ru/tariffs", title="Тарифы")
    assert hit.domain == "mts.ru"


def test_rg_feed_url_works() -> None:
    """RSS-лента «РГ» должна отдавать материалы.

    Адрес в конфигурации устарел: лента отдавала 404, и канал молча
    выпадал из выдачи — без всякой ошибки.
    """
    feeds = dict(RSS_FEEDS)
    assert "rg.ru" in feeds
    assert feeds["rg.ru"] == "https://rg.ru/xml/index.xml"


def test_google_trends_is_available_as_channel() -> None:
    """Google Trends должен быть доступен как канал.

    Он отвечает без ключа и без OAuth — в отличие от Wordstat — и
    показывает, что ищут и что в тренде.
    """
    from pf.web import search_trends

    hits = search_trends("телеком", limit=3)
    assert isinstance(hits, list)


def test_search_web_uses_all_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Поиск должен опрашивать все каналы, включая RSS.

    Раньше вызывался только Bing News: RSS-ленты были описаны в коде,
    но в выдачу не попадали.
    """
    called: list[str] = []

    def _bing_news(query: str, limit: int = 5) -> list[SearchHit]:
        called.append("bing_news")
        return []

    def _rss(query: str, limit: int = 5) -> list[SearchHit]:
        called.append("rss")
        return []

    def _trends(query: str, limit: int = 5) -> list[SearchHit]:
        called.append("trends")
        return []

    monkeypatch.setattr("pf.web.search_bing_news", _bing_news)
    monkeypatch.setattr("pf.web.search_rss_feeds", _rss)
    monkeypatch.setattr("pf.web.search_trends", _trends)
    monkeypatch.setattr("pf.web.search_bing_web", lambda query, limit=5: [])

    search_web("отток абонентов", limit=3)
    assert "bing_news" in called
    assert "rss" in called
    assert "trends" in called


def test_bing_news_survives_empty_first_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bing News должен переспрашивать, если первый ответ пустой.

    Канал отдаёт результаты нестабильно: один и тот же запрос то даёт
    статью, то пустоту. Без повторной попытки агент молча терял
    единственный релевантный источник.
    """
    attempts = {"count": 0}

    def _fake_get(url: str, **kwargs: object) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            body = "<?xml version='1.0'?><rss><channel></channel></rss>"
        else:
            body = (
                "<?xml version='1.0'?><rss><channel><item>"
                "<title>Операторы показали отток абонентов</title>"
                "<link>https://example.org/news</link>"
                "<description>Сотовые операторы отток абонентов</description>"
                "</item></channel></rss>"
            )
        request = httpx.Request("GET", "https://www.bing.com/news/search")
        return httpx.Response(200, text=body, request=request)

    monkeypatch.setattr("pf.web.httpx.get", _fake_get)

    hits = search_bing_news("отток абонентов операторов", limit=3)
    assert attempts["count"] >= 2, "канал должен повторить запрос"
    assert hits, "второй ответ должен дать результат"


def test_cbr_key_rate_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Открытые данные Банка России должны читаться.

    Это официальный источник без ключей: макро-контекст для
    продуктовых решений. Проверяется разбор, а не сеть: документ
    подставляется в тест.
    """
    from pf.web import parse_cbr_key_rate

    xml = (
        "<?xml version='1.0' encoding='windows-1251'?>"
        "<ValCurs><Record Date='13.01.2026'><Nominal>1</Nominal>"
        "<Value>78,7913</Value><VunitRate>78,7913</VunitRate></Record>"
        "<Record Date='14.01.2026'><Nominal>1</Nominal>"
        "<Value>78,8527</Value><VunitRate>78,8527</VunitRate></Record>"
        "</ValCurs>"
    )
    rows = parse_cbr_key_rate(xml)
    assert len(rows) == 2
    assert rows[0]["дата"] == "13.01.2026"
    assert rows[0]["значение_процентов"] == "78.7913"


def test_cbr_is_a_search_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Банк России должен быть одним из каналов поиска."""
    from pf.web import search_web

    called: list[str] = []

    def _fake(query: str, limit: int = 5) -> list[SearchHit]:
        called.append("cbr")
        return [
            SearchHit(
                url="https://www.cbr.ru/press/event/?id=1",
                title="Ключевая ставка Банка России",
                snippet="Ключевая ставка снижена",
                provider="cbr",
            )
        ]

    monkeypatch.setattr("pf.web.search_cbr", _fake)
    monkeypatch.setattr("pf.web.search_bing_news", lambda query, limit=5: [])
    monkeypatch.setattr("pf.web.search_rss_feeds", lambda query, limit=5: [])
    monkeypatch.setattr("pf.web.search_trends", lambda query, limit=5: [])
    monkeypatch.setattr("pf.web.search_bing_web", lambda query, limit=5: [])

    hits = search_web("ключевая ставка", limit=3)
    assert "cbr" in called
    assert hits
