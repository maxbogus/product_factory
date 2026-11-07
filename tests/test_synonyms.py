"""Проверки словаря синонимов: группы не противоречат друг другу.

Словарь — данные, от которых зависит подтверждение утверждений. Ошибка в
нём опаснее ошибки в коде: «удержание» и «отток» в одной группе означают,
что агент начнёт подтверждать противоположные утверждения.
"""

from __future__ import annotations

import pytest

from pf.synonyms import BUILTIN_GROUPS, OPPOSITE_PAIRS, build_map, canonical


def test_map_is_built_from_groups() -> None:
    """Каждая лемма группы указывает на первую лемму."""
    mapping = build_map((("абонент", "клиент", "пользователь"),))
    assert mapping["клиент"] == "абонент"
    assert mapping["абонент"] == "абонент"


def test_unknown_lemma_stays_itself() -> None:
    """Слова вне словаря не меняются."""
    assert canonical("тариф", build_map((("абонент", "клиент"),))) == "тариф"


def test_word_cannot_belong_to_two_groups() -> None:
    """Слово в двух группах делает канонизацию неоднозначной."""
    with pytest.raises(ValueError):
        build_map((("абонент", "клиент"), ("клиент", "пользователь")))


def test_opposite_pairs_are_not_synonyms() -> None:
    """Противоположные понятия не попадают в одну группу."""
    mapping = build_map()
    for left, right in OPPOSITE_PAIRS:
        assert mapping.get(left, left) != mapping.get(right, right), (
            f"«{left}» и «{right}» не могут быть синонимами"
        )


def test_every_builtin_group_starts_with_lowercase_lemma() -> None:
    """Канонические леммы пишутся строчными буквами."""
    for group in BUILTIN_GROUPS:
        assert group[0] == group[0].lower()
        assert group[0].strip() == group[0]
