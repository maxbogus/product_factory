"""Канал arXiv: препринты с пометкой «не рецензирован».

Зачем отдельный канал. arXiv отдаёт Atom — по форме то же, что RSS, но
статус доверия другой: препринт не проходил рецензирование, и выдавать его
за статью нельзя. Поэтому пометка едет с результатом (``SearchHit.peer_reviewed``),
а не теряется в сниппете.

Зачем отдельная функция, а не канал в ``search_web``. Общий фильтр
релевантности считает основы **русских** слов, поэтому английские заголовки
arXiv он выбросил бы молча (`D-26`). Канал вызывается своей командой
(``pf web --arxiv``), а не в общем списке.

Тесты — ``tests/test_arxiv.py`` (``-k atom``): разбор Atom на фикстуре, без
сети. Упавший канал — деградация, а не отказ: пустой список.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from pf.config import settings
from pf.web import USER_AGENT, SearchHit

#: Адрес API arXiv. Ответ — Atom-лента, как у RSS, но с другой разметкой.
ARXIV_API = "https://export.arxiv.org/api/query"

#: Пространство имён Atom: без него ``find("entry")`` ничего не находит.
_ATOM = "{http://www.w3.org/2005/Atom}"


def _clean(text: str) -> str:
    """Свести пробелы и переводы строк к одному пробелу.

    Atom переносит длинный заголовок на несколько строк, и в документе он
    появлялся бы с рваными пробелами.

    Args:
        text: Текст из ленты.

    Returns:
        Текст в одну строку.
    """
    return " ".join(text.split())


def _text(entry: ET.Element, tag: str) -> str:
    """Текст поля записи — с пространством имён Atom и без него.

    Args:
        entry: Запись ленты.
        tag: Имя поля.

    Returns:
        Текст поля либо пустая строка.
    """
    node = entry.find(f"{_ATOM}{tag}")
    if node is None:
        node = entry.find(tag)
    if node is None:
        return ""
    return _clean("".join(node.itertext()))


def parse_atom(xml_text: str, limit: int = 5) -> list[SearchHit]:
    """Разобрать Atom-ленту arXiv в результаты поиска.

    Битая запись пропускается, а не роняет разбор: посторонний элемент в
    ленте — не повод потерять остальные. Ссылка берётся из ``id``: arXiv
    кладёт туда адрес страницы препринта.

    Args:
        xml_text: Ответ arXiv.
        limit: Сколько результатов вернуть.

    Returns:
        Результаты с пометкой «не рецензирован»; пустой список, если
        разобрать не удалось.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    entries = root.findall(f"{_ATOM}entry") or root.findall("entry")
    hits: list[SearchHit] = []
    for entry in entries:
        url = _text(entry, "id")
        title = _text(entry, "title")
        if not url or not title:
            continue
        hits.append(
            SearchHit(
                url=url,
                title=title,
                snippet=_text(entry, "summary"),
                provider="arxiv",
                peer_reviewed=False,
            )
        )
        if len(hits) >= limit:
            break
    return hits


def search_arxiv(query: str, limit: int = 5) -> list[SearchHit]:
    """Найти препринты в arXiv.

    Упавший канал не мешает остальным: это деградация, а не отказ. Поэтому
    при сетевой ошибке возвращается пустой список, а вызывающий решает, что
    сказать человеку.

    Args:
        query: Поисковый запрос — на английском: arXiv не ищет по-русски.
        limit: Сколько результатов вернуть.

    Returns:
        Найденные препринты; пустой список, если канал недоступен.
    """
    try:
        response = httpx.get(
            ARXIV_API,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max(1, min(limit * 2, 30)),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=settings.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return parse_atom(response.text, limit=limit)
