"""Проверки оснований: цитата подтверждается только дословно.

Это главный тест безопасности агента. Если он падает — агент способен
выдумывать факты, а весь остальной код теряет смысл.
"""

from __future__ import annotations

import pytest

from pf.ground import (
    SupportStatus,
    find_quote,
    normalize,
    split_source_id,
    token_overlap,
    verify_claim,
    verify_quote,
)

SOURCE = (
    "«Мегафон» — российская телекоммуникационная компания.\n"
    "По состоянию на июнь 2024 года компания обслужила 77,26 миллиона абонентов.\n"
    "Компания существует с 1993 года."
)


def test_real_quote_is_confirmed() -> None:
    """Цитата из источника подтверждается."""
    support = verify_quote(SOURCE, "обслужила 77,26 миллиона абонентов", "test/источник")
    assert support.status is SupportStatus.CONFIRMED
    assert support.is_usable
    assert support.paragraph_index == 1
    assert support.citation == "test/источник#p1"


def test_fabricated_quote_is_rejected() -> None:
    """Выдуманная цитата отбрасывается."""
    support = verify_quote(
        SOURCE,
        "удержание упало на 8 процентных пунктов из-за ценовой политики",
        "test/источник",
    )
    assert support.status is SupportStatus.NO_DATA
    assert not support.is_usable
    assert "не найдена" in support.detail


def test_short_quote_is_rejected() -> None:
    """Слишком короткая цитата не доказывает ничего."""
    support = verify_quote(SOURCE, "да", "test/источник")
    assert support.status is SupportStatus.NO_DATA


def test_quote_is_case_and_space_insensitive() -> None:
    """Регистр и лишние пробелы не мешают найти цитату."""
    support = verify_quote(SOURCE, "ОБСЛУЖИЛА   77,26 МИЛЛИОНА", "test/источник")
    assert support.status is SupportStatus.CONFIRMED


def test_source_id_with_paragraph_is_split() -> None:
    """Склеенный идентификатор разбирается на путь и номер абзаца."""
    path, paragraph = split_source_id("security/Мегафон (компания)#p126")
    assert path == "security/Мегафон (компания)"
    assert paragraph == 126


def test_source_id_without_paragraph() -> None:
    """Обычный идентификатор разбирается без номера абзаца."""
    path, paragraph = split_source_id("security/Мегафон (компания)")
    assert path == "security/Мегафон (компания)"
    assert paragraph is None


def test_claim_is_confirmed_by_paragraph_coverage() -> None:
    """Утверждение, собранное из слов абзаца, подтверждается."""
    support = verify_claim(
        SOURCE,
        "По состоянию на июнь 2024 года компания обслужила 77,26 миллиона абонентов.",
        "test/источник",
        paragraph_index=1,
    )
    assert support.status is SupportStatus.CONFIRMED


def test_claim_with_foreign_words_is_rejected() -> None:
    """Утверждение со словами, которых нет в абзаце, отклоняется."""
    support = verify_claim(
        SOURCE,
        "Тарифы на мобильную связь выросли в четыре раза за последний квартал.",
        "test/источник",
        paragraph_index=1,
    )
    assert support.status is SupportStatus.NO_DATA


def test_quote_takes_priority_over_coverage() -> None:
    """Если цитата есть, проверяется именно она."""
    support = verify_claim(
        SOURCE,
        "любой текст",
        "test/источник",
        paragraph_index=0,
        quote="российская телекоммуникационная компания",
    )
    assert support.status is SupportStatus.CONFIRMED


def test_normalize_collapses_whitespace() -> None:
    """Нормализация схлопывает пробелы и переносы."""
    assert normalize("  много   пробелов \n тут ") == "много пробелов тут"


def test_token_overlap_ratio() -> None:
    """Покрытие считается как доля совпавших значимых слов."""
    overlap = token_overlap("удержание абонентов падает", "удержание абонентов падает")
    assert overlap == pytest.approx(1.0)
    partial = token_overlap("удержание абонентов падает", "удержание абонентов растёт")
    assert 0.0 < partial < 1.0


def test_find_quote_reports_paragraph() -> None:
    """Поиск цитаты возвращает номер абзаца."""
    found, index = find_quote(SOURCE, "с 1993 года")
    assert found
    assert index == 2


def test_split_source_id_understands_row_anchor() -> None:
    """Ссылка на строку таблицы должна разбираться.

    Метрики ссылаются на строки таблиц (`#row3`), а не на абзацы
    (`#p12`). Если разбор не понимает `#row`, метрика не может быть
    проверена и всегда отклоняется как «источник не найден».
    """
    from pf.ground import split_source_id

    path, row = split_source_id("project/data/numbers.csv#row3")
    assert path == "project/data/numbers.csv"
    assert row == 3


def test_split_source_id_understands_plain_path() -> None:
    """Ссылка без якоря остаётся как есть."""
    from pf.ground import split_source_id

    path, row = split_source_id("project/data/numbers.csv")
    assert path == "project/data/numbers.csv"
    assert row is None
