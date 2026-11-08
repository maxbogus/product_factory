"""Канал arXiv: разбор Atom и пометка «не рецензирован».

Проверяется то, что решает статус источника: адрес и заголовок достаются из
записи, битая запись не роняет остальные, а пометка «не рецензирован» едет
вместе с результатом. Сеть в тестах не участвует: разбор проверяется на
фикстуре, потому что тест, ходящий в arXiv, падал бы от чужой доступности.
"""

from __future__ import annotations

from pf.arxiv import parse_atom

#: Ответ arXiv в том виде, в котором он приходит: две записи, вторая битая.
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: all:churn</title>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Churn prediction for mobile tariffs
      with gradient boosting</title>
    <summary>
      We study churn in mobile networks and show that
      boosting beats logistic regression.
    </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.05678v1</id>
    <title></title>
    <summary>Запись без заголовка: разбирать нечего.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.09999v2</id>
    <title>Retention and transparency of tariff plans</title>
    <summary>Transparent tariffs reduce churn.</summary>
  </entry>
</feed>
"""


def test_atom_parses_entries() -> None:
    """Адрес, заголовок и аннотация достаются из записи Atom."""
    hits = parse_atom(ATOM, limit=5)

    assert [hit.title for hit in hits] == [
        "Churn prediction for mobile tariffs with gradient boosting",
        "Retention and transparency of tariff plans",
    ]
    assert hits[0].url == "http://arxiv.org/abs/2401.01234v1"
    assert hits[0].provider == "arxiv"
    assert "boosting beats logistic regression" in hits[0].snippet


def test_atom_entry_is_marked_not_peer_reviewed() -> None:
    """Препринт не выдаётся за статью: пометка едет с результатом."""
    hits = parse_atom(ATOM, limit=5)

    assert hits
    assert all(hit.peer_reviewed is False for hit in hits)


def test_atom_skips_malformed_entry() -> None:
    """Запись без заголовка пропускается, остальные остаются.

    Раньше один посторонний элемент в ленте ронял разбор целиком, и канал
    выглядел пустым — то есть «ничего не нашлось» вместо «сломалось».
    """
    hits = parse_atom(ATOM, limit=5)

    assert len(hits) == 2
    assert all(hit.title for hit in hits)


def test_atom_respects_limit() -> None:
    """Лимит соблюдается — в запросе и в разборе."""
    hits = parse_atom(ATOM, limit=1)

    assert len(hits) == 1


def test_atom_broken_xml_returns_nothing() -> None:
    """Неразобранная лента — пустой список, а не исключение."""
    assert parse_atom("<feed><entry>", limit=5) == []
