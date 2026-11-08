"""Проверки слоя чистки: текст для сравнения — не текст для цитирования.

Зачем отдельный модуль и отдельные проверки. До сессии 13 цитату из таблицы
чистили в одном месте (в TF-IDF замера) и не чистили в другом (у эмбеддингов),
а url источника и имена колонок оставались в тексте. Измерено: у SBERT
оракул-полнота 0,07 → 0,23 после чистки, то есть мусор действительно мешал.

Ключевое правило, которое держат эти проверки: **чистка касается только
сравнения**. Дословность цитаты (`pf.ground.find_quote`) по-прежнему
проверяется по сырому тексту, иначе «чистка» стала бы способом подтвердить
пересказ (`docs/decisions.md`, D-38).
"""

from __future__ import annotations

import importlib
from types import ModuleType

from pf.ground import find_quote

#: Строка таблицы проекта в том виде, в каком она лежит в материале.
ROW = (
    "оператор: МТС, показатель: абонентская база в России, значение: 82, "
    "период: конец 3-го квартала 2024, "
    "источник: https://ru.wikipedia.org/wiki/МТС_(компания)"
)


def _sanitize() -> ModuleType:
    """Загрузить модуль чистки.

    Импорт внутри функции: до правки модуля нет, и красный прогон должен
    показать «failed», а не «collection error».

    Returns:
        Модуль `pf.sanitize`.
    """
    return importlib.import_module("pf.sanitize")


def test_sanitize_strips_url_and_provenance() -> None:
    """Адрес источника и слово «источник» уходят из текста сравнения."""
    clean = _sanitize().for_comparison(ROW)

    assert "http" not in clean
    assert "источник" not in clean
    assert "wikipedia" not in clean


def test_sanitize_strips_field_names_keeps_values() -> None:
    """Имена колонок уходят, значения остаются — иначе чистка съест содержание."""
    clean = _sanitize().for_comparison(ROW).lower()

    for name in ("оператор", "показатель", "значение", "период"):
        assert name not in clean, name
    # Сравнение идёт без регистра (`ground.normalize`), поэтому значения
    # проверяются в нижнем регистре — регистр не содержание.
    for value in ("мтс", "82", "абонентская база", "3-го квартала 2024"):
        assert value in clean, value


def test_sanitize_strips_markup_entities_and_page_noise() -> None:
    """Разметка, HTML-сущности и колонтитулы не участвуют в сравнении."""
    raw = "<div>Страница 17.&nbsp;Выручка <b>выросла</b> &mdash; на 12 %</div>"
    clean = _sanitize().for_comparison(raw)

    assert "<" not in clean
    assert "&nbsp;" not in clean
    assert "Страница 17" not in clean
    assert "выручка выросла" in clean.lower()
    assert "12" in clean


def test_sanitize_is_idempotent() -> None:
    """Повторная чистка ничего не меняет — иначе результат зависит от числа прогонов."""
    module = _sanitize()
    once = module.for_comparison(ROW)
    twice = module.for_comparison(once)

    assert once == twice


def test_sanitize_keeps_numbers_and_units() -> None:
    """Числа и единицы остаются: их разбирает количественный слой (`T-43`)."""
    clean = _sanitize().for_comparison("ARPU: 320, период: 2017-Q2, значение: 7,39 млн").lower()

    assert "320" in clean
    assert "7,39 млн" in clean
    assert "2017-q2" in clean


def test_sanitize_does_not_replace_verbatim_check() -> None:
    """Дословность проверяется по сырому тексту, а не по очищенному.

    Если бы `find_quote` начал работать с очищенным текстом, цитата с url
    перестала бы считаться дословной — а цитата без url, наоборот, проходила
    бы там, где в источнике её нет. Это разные тексты и разные проверки.
    """
    module = _sanitize()

    found_raw, _ = find_quote(ROW, ROW)
    clean = module.for_comparison(ROW)
    found_clean, _ = find_quote(clean, ROW)

    assert found_raw is True
    assert found_clean is False
