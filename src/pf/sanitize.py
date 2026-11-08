"""Чистка текста **для сравнения** — не для цитирования.

Зачем модуль. Цитата из таблицы приходит с мусором: адрес источника
(«источник: https://…», девять слов), имена колонок («оператор:»,
«показатель:», «значение:»), иногда разметка и колонтитулы документа. Всё это
к содержанию не относится, но попадает в сравнение.

Измерено на наборе `eval/pairs/out_of_place.jsonl`: после чистки у SBERT
оракул-полнота флага выросла с 0,07 до 0,23 при том же AUC, а токенов в
цитатах стало 404 вместо 921. Мусор мешал не «немного» — он занимал больше
половины текста.

**Что модуль не делает.** Он не участвует в проверке дословности. Цитата
сверяется с **сырым** текстом источника (`pf.ground.find_quote`), иначе чистка
стала бы способом подтвердить пересказ: убрали url — и «цитата» нашлась там,
где её нет. Это правило записано в `docs/decisions.md` (D-38) и проверяется
тестом `test_sanitize_does_not_replace_verbatim_check`.
"""

from __future__ import annotations

import html
import re

from pf.ground import normalize

#: Метка провенанса в строках таблиц: всё после неё — адрес, а не содержание.
PROVENANCE = "источник:"

#: Имена колонок таблиц проекта. Длинные идут первыми: иначе «значение»
#: съело бы начало «значение_процентов».
FIELD_NAMES = (
    "значение_процентов",
    "цена_до_скидки_руб",
    "цена_руб_30_дней",
    "особенности",
    "показатель",
    "оператор",
    "значение",
    "период",
    "источник",
    "регион",
    "тариф",
    "интернет",
    "минуты",
    "мегасилы",
    "дата",
)

#: Поле таблицы: имя колонки и двоеточие после него.
_FIELD = re.compile(r"\b(" + "|".join(FIELD_NAMES) + r")\s*:", re.IGNORECASE)

#: Тег разметки.
_TAG = re.compile(r"<[^>]{1,200}>")

#: Сущность HTML, которую `html.unescape` не знает или знает частично.
_ENTITY = re.compile(r"&[a-zA-Z][a-zA-Z0-9]{1,10};|&#\d{1,6};")

#: Колонтитул или номер страницы в выгрузке документа.
_PAGE = re.compile(r"\b(?:страница|стр\.)\s*\d{1,4}\b\.?", re.IGNORECASE)

#: Неразрывные и нулевой ширины пробелы: для сравнения это пробел.
_INVISIBLE = re.compile("[\u00a0\u2009\u200b\u200e\u200f\ufeff]")


def strip_provenance(text: str) -> str:
    """Отрезать адрес источника вместе с меткой.

    Args:
        text: Текст строки или абзаца.

    Returns:
        Текст без хвоста «источник: …».
    """
    head, separator, _tail = text.partition(PROVENANCE)
    return head if separator else text


def strip_field_names(text: str) -> str:
    """Убрать имена колонок, оставив значения.

    Args:
        text: Текст строки таблицы.

    Returns:
        Текст без «оператор:», «показатель:» и подобных меток.
    """
    return _FIELD.sub(" ", text)


def strip_markup(text: str) -> str:
    """Убрать разметку и сущности HTML.

    Args:
        text: Текст с разметкой.

    Returns:
        Текст без тегов и сущностей.
    """
    without_tags = _TAG.sub(" ", text)
    without_entities = _ENTITY.sub(" ", without_tags)
    return html.unescape(without_entities)


def strip_page_noise(text: str) -> str:
    """Убрать колонтитулы и номера страниц выгрузки.

    Args:
        text: Текст выгрузки.

    Returns:
        Текст без «Страница 17.» и подобного.
    """
    return _PAGE.sub(" ", text)


def for_comparison(text: str) -> str:
    """Привести текст к виду, пригодному для сравнения.

    Порядок важен: провенанс отрезается до имён колонок, иначе метка
    «источник:» исчезнет, а адрес останется.

    Args:
        text: Исходный текст.

    Returns:
        Очищенный и нормализованный текст. Функция идемпотентна.
    """
    without_provenance = strip_provenance(text)
    without_markup = strip_markup(without_provenance)
    without_fields = strip_field_names(without_markup)
    without_pages = strip_page_noise(without_fields)
    return normalize(_INVISIBLE.sub(" ", without_pages))
