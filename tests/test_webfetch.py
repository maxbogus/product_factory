"""Проверки разбора веб-страницы.

Главное правило: разбор страницы не должен (а) вытащить в текст меню и
скрипты, (б) обойти ``robots.txt``, (в) принести страницу с
запрещённого домена.

Разбор внешней страницы не делает её подтверждённым основанием — этим
занимается ``pf.web.verify_web_quote``, и там статус всегда «требует
проверки».
"""

from __future__ import annotations

from pf.webfetch import (
    FetchedPage,
    extract_paragraphs,
    extract_title,
    fetch_pages,
)

#: Страница-фикстура: текст, меню, скрипт и подвал.
HTML = """
<html><head><title>Отток абонентов Мегафон</title></head>
<body>
  <nav class="menu"><a href="/">Главная</a><a href="/news">Новости</a></nav>
  <script>var tracking = 1;</script>
  <p>Оператор сообщил, что число абонентов, пользующихся тарифом, выросло
     на 2,4 процента за последний квартал.</p>
  <p>Коротко.</p>
  <div class="footer">Все права защищены, 2026 год, копирование материалов
     запрещено без письменного разрешения редакции.</div>
</body></html>
"""


def test_paragraphs_are_extracted() -> None:
    """Содержательный абзац найден."""
    paragraphs = extract_paragraphs(HTML)
    assert any("выросло" in text for text in paragraphs)


def test_short_paragraphs_are_dropped() -> None:
    """Абзац короче порога не берётся: из него нечего цитировать."""
    paragraphs = extract_paragraphs(HTML)
    assert "Коротко." not in paragraphs


def test_menu_and_scripts_are_removed() -> None:
    """Меню и подвал не попадают в текст."""
    paragraphs = extract_paragraphs(HTML)
    assert not any("Главная" in text for text in paragraphs)
    assert not any("tracking" in text for text in paragraphs)


def test_title_is_extracted() -> None:
    """Заголовок страницы известен."""
    assert extract_title(HTML) == "Отток абонентов Мегафон"


def test_page_without_paragraphs_falls_back_to_body() -> None:
    """Страница без абзацев даёт весь видимый текст."""
    html = "<html><body><div>Текст без тегов абзаца, но достаточно длинный, чтобы быть принятым.</div></body></html>"
    paragraphs = extract_paragraphs(html)
    assert paragraphs
    assert "достаточно длинным" not in paragraphs[0]
    assert "принятым" in paragraphs[0]


def test_fetch_pages_skips_forbidden_domains() -> None:
    """Страницы с запрещённых доменов не скачиваются."""
    from pf.web import SearchHit

    hits = [
        SearchHit(url="https://example.com/page", title="Чужой"),
        SearchHit(url="https://mts.ru/tariffs", title="Тарифы"),
    ]
    pages = fetch_pages(hits, limit=5, allowed_domains=frozenset({"mts.ru"}))
    assert all("example.com" not in page.url for page in pages)


def test_empty_page_has_status() -> None:
    """У страницы без абзацев статус «текст не найден»."""
    page = FetchedPage(url="https://example.com", title="Пусто")
    assert page.status == "ok"
    assert page.text == ""
