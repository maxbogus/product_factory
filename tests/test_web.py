"""Проверки веб-источников.

Главное правило: внешний источник **никогда** не становится подтверждённым
основанием. Он проходит проверку дословности — но не проходит проверку
«лежит в отобранном локальном срезе», поэтому его статус «требует
проверки».

Если это правило ослабнет, агент начнёт выдавать добытое из веба за
проверенный факт, и весь механизм честности перестанет работать.
"""

from __future__ import annotations

import pytest

from pf.ground import SupportStatus
from pf.web import (
    ALLOWED_DOMAINS,
    WebSource,
    is_allowed,
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
    assert ALLOWED_DOMAINS
    assert "ru.wikipedia.org" in ALLOWED_DOMAINS


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


@pytest.mark.parametrize("domain", sorted(ALLOWED_DOMAINS))
def test_every_allowed_domain_is_valid(domain: str) -> None:
    """Все разрешённые домены имеют осмысленный вид."""
    assert "." in domain
    assert " " not in domain
