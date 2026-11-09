"""Проверки лексической близости: леммы, термины, числа, распознавание.

Модуль отвечает за то, чтобы проверка оснований понимала слова, а не
только строки. Здесь проверяется именно лексика: без модели и без сети.
"""

from __future__ import annotations

import pytest

from pf.lexical import (
    TokenStatus,
    damerau_distance,
    exact_terms_in,
    has_negation,
    is_exact_term,
    is_known_word,
    language,
    lemma,
    mask_numbers,
    normalize_homoglyphs,
    number_tolerance,
    numbers_in,
    numbers_match,
    overlap,
    recognize,
    translit_key,
)


def test_lemma_reduces_inflected_forms() -> None:
    """Словоформы сводятся к одной лемме."""
    assert lemma("абоненты") == lemma("абонентов") == "абонент"


def test_homoglyphs_are_normalized_only_in_mixed_words() -> None:
    """Смешанная раскладка исправляется, чистая латиница — нет."""
    assert normalize_homoglyphs("удержаниe") == "удержание"
    assert normalize_homoglyphs("opex") == "opex"
    assert normalize_homoglyphs("абонент") == "абонент"


def test_mixed_layout_word_is_recognized() -> None:
    """Слово с латинской буквой распознаётся как русское."""
    resolution = recognize("удержаниe")
    assert resolution.status is TokenStatus.RESOLVED_HEURISTIC
    assert resolution.lemma == "удержание"
    assert "гомоглифы" in resolution.reason


def test_translit_key_matches_brand_spellings() -> None:
    """«Мегафон» и «megafon» дают один ключ."""
    assert translit_key("мегафон") == translit_key("megafon")


def test_unknown_word_is_oov() -> None:
    """Незнакомое слово не выдумывает лемму, а уходит в OOV."""
    resolution = recognize("криптовалюта")
    assert resolution.status is TokenStatus.OOV
    assert "не распознано" in resolution.reason


def test_typo_is_recognized_with_distance() -> None:
    """Опечатка распознаётся по расстоянию до словаря."""
    resolution = recognize("криптавалюта", {"криптовалюта": "криптовалюта"})
    assert resolution.status is TokenStatus.RESOLVED_HEURISTIC
    assert resolution.candidate == "криптовалюта"
    assert resolution.distance == 1


def test_damerau_counts_transposition_as_one_edit() -> None:
    """Перестановка букв — одна правка, а не две."""
    assert damerau_distance("абонент", "аобнент") == 1


def test_known_word_is_not_oov() -> None:
    """Обычное слово морфология знает."""
    assert is_known_word("абонент")
    assert not is_known_word("криптовалюта")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("МегаФон — оператор связи", "ru"),
        ("Megafon is a telecom operator", "en"),
        ("", "other"),
    ],
)
def test_language_detection(text: str, expected: str) -> None:
    """Язык определяется по алфавиту."""
    assert language(text) == expected


def test_negation_is_detected() -> None:
    """Отрицание находится, хотя «не» — стоп-слово."""
    assert has_negation("удержание не падает")
    assert not has_negation("удержание падает")


def test_numbers_are_parsed_with_russian_separators() -> None:
    """Запятая, точка и пробелы в разрядах читаются как числа."""
    assert numbers_in("77,26 млн и 1 234,5 ₽") == [77.26, 1234.5]


def test_numbers_do_not_come_from_units() -> None:
    """«5G» не даёт ложных чисел."""
    assert numbers_in("Мега5G тариф") == []


def test_number_tolerance_is_relative_for_small_values() -> None:
    """Для малых величин абсолютный допуск не включается."""
    assert number_tolerance(87.0) == pytest.approx(0.1)
    assert number_tolerance(0.2) < 0.1


def test_rounded_number_matches_with_tolerance() -> None:
    """77,3 подтверждает 77,26: это округление, а не подмена."""
    matched, detail = numbers_match([77.3], [77.26])
    assert matched, detail


def test_wrong_number_is_rejected() -> None:
    """78,0 не подтверждает 77,26."""
    matched, detail = numbers_match([78.0], [77.26])
    assert not matched
    assert "не совпадает" in detail


def test_exact_terms_are_numbers_brands_and_acronyms() -> None:
    """Точные термины не заменяются синонимами."""
    assert is_exact_term("МТС")
    assert is_exact_term("ARPU")
    assert is_exact_term("77,26")
    assert not is_exact_term("абонент", {"абонент": "абонент"})


def test_exact_terms_list_skips_numbers() -> None:
    """Числа не попадают в список терминов: они сверяются отдельно."""
    terms = exact_terms_in("МТС обслужила 77,26 млн")
    assert "МТС" in terms
    assert "77,26" not in terms


def test_synonym_covers_claim() -> None:
    """«Клиенты» в утверждении покрываются «абонентами» источника."""
    result = overlap(
        "компания обслужила 77,26 миллиона клиентов",
        "компания обслужила 77,26 миллиона абонентов",
    )
    assert result.coverage == pytest.approx(1.0)
    assert result.matched, result.missing_terms


def test_exact_term_substitution_is_rejected() -> None:
    """Подмена точного термина отклоняется, даже если слова похожи."""
    result = overlap(
        "МТС обслужила 87 миллионов абонентов",
        "МегаФон обслужила 87 миллионов абонентов",
    )
    assert not result.matched
    assert "МТС" in result.missing_terms


def test_negation_added_by_model_is_rejected() -> None:
    """Добавленное отрицание меняет смысл и отклоняется."""
    result = overlap(
        "удержание не падает у абонентов",
        "удержание падает у абонентов",
    )
    assert not result.negation_ok
    assert not result.matched


def test_negation_in_source_does_not_break_match() -> None:
    """Отрицание в длинном абзаце не мешает подтверждению."""
    result = overlap(
        "компания обслужила 77,26 миллиона абонентов",
        "По состоянию на июнь 2024 года компания не раз обслужила 77,26 миллиона абонентов.",
    )
    assert result.negation_ok
    assert result.matched, result.missing_terms


def test_oov_tokens_are_reported() -> None:
    """Незнакомое слово утверждения называется в результате."""
    result = overlap("криптовалюта растёт", "абоненты растут")
    assert "криптовалюта" in result.oov_tokens


def test_overlap_rejects_foreign_words() -> None:
    """Утверждение из чужих слов не покрывается абзацем."""
    result = overlap(
        "Тарифы на мобильную связь выросли в четыре раза",
        "По состоянию на июнь 2024 года компания обслужила 77,26 миллиона абонентов.",
    )
    assert not result.matched


# ── Термины: одно правило и границы слова ───────────────────────────────────


def test_term_short_abbreviation_is_found_by_text() -> None:
    """«МТС» находится там, где есть, и не путается с «МГТС».

    Аббревиатура короче четырёх букв не попадает в мешок слов: пока наличие
    термина проверялось по леммам, «МТС» не находился никогда — даже когда
    строка источника начиналась с «оператор: МТС». Подстрочное сравнение
    ошибается в другую сторону и склеивает разные имена. Здесь проверяются
    обе стороны: термин ищется по тексту, но только целым словом.
    """
    row = "оператор: МТС, показатель: абонентская база в России, значение: 82"
    claim = "Абонентская база МТС в России составляет 82 млн человек"

    assert overlap(claim, row).missing_terms == []

    other = overlap("МТС обслуживает 4 млн абонентов", "МГТС обслуживает 4 млн абонентов")
    assert other.missing_terms == ["МТС"]


def test_numbers_are_masked_for_rounding_comparison() -> None:
    """Маска числа отделяет округление от пересказа.

    «С точностью до допуска чисел» означает, что различие **только** в
    записи числа. Пока это не проверялось, пересказ строки таблицы («МТС,
    абонентская база в России: 82») проходил как округление, хотя из цитаты
    выпали метки колонок.
    """
    assert mask_numbers("обслужила 77,3 миллиона") == mask_numbers("обслужила 77,26 миллиона")
    assert mask_numbers("МТС, абонентская база в России: 82") != mask_numbers(
        "оператор: МТС, показатель: абонентская база в России, значение: 82"
    )
