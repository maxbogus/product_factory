"""Проверки количественного слоя: слова-числа и единицы → величина.

Зачем. Числа сегодня сравниваются только когда они записаны цифрами
(`pf.lexical.numbers_match`), а «почти сорок миллионов» не разбирает никто:
измерено, что такая пара отклоняется при покрытии 0,17, хотя подтверждение
честное (`eval/pairs/out_of_place.jsonl`, пара `p-049`).

Направление одно и оно важное: **слова → число**. Цифры в пропись не
переводятся: на наборе из 60 пар это дало бы +217 токенов к цитатам и потерю
точности (`172,87` → «сто семьдесят две целых восемьдесят семь сотых»).
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Строка таблицы проекта: значение цифрами, без названия единицы.
ROW_VALUE = "оператор: МТС, показатель: абонентская база, значение: 38.9, период: конец июля"

#: Строка с процентом: единица названа в имени колонки.
ROW_PERCENT = "дата: 13.01.2026, значение_процентов: 78.7913"

#: Строка, где единица названа текстом.
ROW_MILLIONS = "значение: 7.39 млн, период: 2019"


def _quantities() -> ModuleType:
    """Загрузить модуль количеств.

    Импорт внутри функции: до правки модуля нет, и красный прогон должен
    показать «failed», а не «collection error».

    Returns:
        Модуль `pf.quantities`.
    """
    return importlib.import_module("pf.quantities")


def test_quantities_words_become_numbers() -> None:
    """«почти сорок миллионов» — это число 40, шкала млн и модификатор «почти»."""
    parsed = _quantities().parse_quantities("почти сорок миллионов абонентов")

    assert len(parsed) == 1
    item = parsed[0]
    assert item.value == 40.0
    assert item.scale == 1_000_000
    assert item.modifier == "почти"


def test_quantities_parses_digits_with_scale() -> None:
    """Цифры разбираются в том же виде: значение, шкала, единица."""
    parsed = _quantities().parse_quantities("172,87 млн абонентов")

    assert len(parsed) == 1
    assert parsed[0].value == 172.87
    assert parsed[0].scale == 1_000_000


def test_quantities_compare_approximate_word_with_exact_row() -> None:
    """«почти сорок миллионов» подтверждается строкой «значение: 38.9»."""
    ok, detail = _quantities().compare_quantities(
        "Tele2 обслуживала почти сорок миллионов абонентов", ROW_VALUE
    )

    assert ok is True, detail


def test_quantities_compare_about_with_exact_row() -> None:
    """«около трёхсот рублей» подтверждается значением 302."""
    ok, detail = _quantities().compare_quantities(
        "ВымпелКом получал около трёхсот рублей с абонента", "значение: 302"
    )

    assert ok is True, detail


def test_quantities_lower_bound_rejects_smaller_value() -> None:
    """«более восьмидесяти процентов» не подтверждается значением 78,79."""
    ok, detail = _quantities().compare_quantities(
        "ключевая ставка более восьмидесяти процентов", ROW_PERCENT
    )

    assert ok is False
    assert "больше" in detail.lower() or "ниже" in detail.lower()


def test_quantities_upper_bound_accepts_smaller_value() -> None:
    """«чуть ниже восьмидесяти процентов» подтверждается значением 78,79."""
    ok, detail = _quantities().compare_quantities(
        "ключевая ставка чуть ниже восьмидесяти процентов", ROW_PERCENT
    )

    assert ok is True, detail


def test_quantities_plain_number_must_match_exactly() -> None:
    """Без модификатора приближение не разрешается: «сорок» — это не 38,9."""
    ok, detail = _quantities().compare_quantities("сорок миллионов абонентов", ROW_VALUE)

    assert ok is False, detail


def test_quantities_unit_mismatch_is_rejected() -> None:
    """Обе стороны назвали единицу, и она разная — подтверждения нет."""
    ok, detail = _quantities().compare_quantities("7,39 % абонентов", ROW_MILLIONS)

    assert ok is False
    assert "единиц" in detail.lower()


def test_quantities_period_terms_are_not_values() -> None:
    """Срок («3 месяца») величиной не считается — как и в проверке чисел плана."""
    parsed = _quantities().parse_quantities("эффект наступит через 3 месяца")

    assert parsed == []
