"""Периоды как ключ: год, квартал, дата, месяц и интервал.

Зачем отдельный модуль. Квартал — часть года, а не число из цифр периода
(`D-34`). Правило разбора жило в `pf.analysis`, то есть **только в расчётах
ряда**, а период называет и утверждение о факте («в 2016 году», «за 2019
год»), и сам источник («период: сентябрь 2019», «накопленно с 2013 года»).
Проверка оснований такой период не видела вовсе: строка «27,4 млн накопленно
с 2013 года, период: сентябрь 2019» подтверждала утверждение «за 2016 год —
27,4 млн» (`eval/pairs/out_of_place.jsonl`, пары `p-032`, `p-040`).

Второй копии правила быть не должно: «1 кв. 2017» и «2017-Q1» — один период,
и если бы проверка разбирала его по-своему, она бы этого не знала. Поэтому
здесь и ключ сортировки (`key_of`, `key_of_row`), и разбор в интервалы
(`spans`), а `pf.analysis` делегирует сюда.

Шкала одна на всех: **год + квартал / 5** для квартала, **год + (месяц − 1) /
12** для месяца. Тогда год идёт раньше своих кварталов, четвёртый квартал —
раньше следующего года, и никакие два разных периода не получают один ключ.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: Римские номера кварталов: источники пишут и так («IV квартал 2017»).
_ROMAN_QUARTERS: dict[str, int] = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

#: Словесные номера кварталов: «первый квартал 2017».
_WORD_QUARTERS: dict[str, int] = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4}

#: Слово «квартал» в записях: «кв.», «кв», «квартал».
_QUARTER_WORD = r"(?:кв\.|кв\b|квартал)"

#: Цифра квартала перед словом: «1 кв. 2017», «3-й квартал 2024».
_QUARTER_DIGIT = re.compile(rf"\b([1-4])\s*(?:-?\s*й)?\s*{_QUARTER_WORD}")

#: Латинская запись: «Q1 2017», «2017-Q1».
_QUARTER_LATIN = re.compile(r"\bq\s*([1-4])\b")

#: Римская запись: «I квартал 2017», «IV кв.».
_QUARTER_ROMAN = re.compile(rf"\b(iv|iii|ii|i)\s*{_QUARTER_WORD}")

#: Словесная запись: «первый квартал 2017», «четвёртый квартал 2017».
_QUARTER_WORDED = re.compile(rf"\b(четверт|четвёрт|трет|втор|перв)\w*\s*{_QUARTER_WORD}")

#: Год внутри записи периода.
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

#: Месяцы: основа слова → номер. Порядок важен: «март» раньше «мая».
_MONTHS: tuple[tuple[str, int], ...] = (
    ("январ", 1),
    ("феврал", 2),
    ("март", 3),
    ("апрел", 4),
    ("май", 5),
    ("мая", 5),
    ("июн", 6),
    ("июл", 7),
    ("август", 8),
    ("сентябр", 9),
    ("октябр", 10),
    ("ноябр", 11),
    ("декабр", 12),
)

#: Русская дата: «13.01.2026».
_DATE_RU = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")

#: ISO-дата: «2026-01-13».
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

#: Затухание при несовпадении периодов: доля за год разрыва.
GAP_DECAY_YEARS = 1.0

#: Колонки, в которых лежит период строки таблицы.
PERIOD_COLUMNS: tuple[str, ...] = ("период", "дата", "period", "date")


@dataclass(frozen=True, slots=True)
class Period:
    """Период текста как интервал в шкале года.

    Attributes:
        start: Начало интервала в шкале года: «сентябрь 2019» → 2019,667.
        end: Конец интервала; для точки совпадает с началом.
        raw: Как период записан — для объяснения отказа.
    """

    start: float
    end: float
    raw: str

    def overlaps(self, other: Period) -> bool:
        """Пересекаются ли периоды.

        Args:
            other: Второй период.

        Returns:
            Есть ли общая точка времени.
        """
        return self.start <= other.end and other.start <= self.end

    def gap(self, other: Period) -> float:
        """Разрыв между периодами в годах (0 при пересечении).

        Args:
            other: Второй период.

        Returns:
            Расстояние между интервалами в годах.
        """
        if self.overlaps(other):
            return 0.0
        return max(other.start - self.end, self.start - other.end)


def quarter_of(text: str) -> int:
    """Номер квартала из записи периода.

    Квартал — не украшение записи, а часть времени: без него ряд
    «1 кв. 2017 … 4 кв. 2017» упорядочивается по склеенным цифрам.

    Args:
        text: Запись периода в нижнем регистре.

    Returns:
        Номер квартала 1–4 либо 0, если квартал не назван.
    """
    for pattern in (_QUARTER_LATIN, _QUARTER_DIGIT):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    match = _QUARTER_ROMAN.search(text)
    if match:
        return _ROMAN_QUARTERS[match.group(1)]
    match = _QUARTER_WORDED.search(text)
    if match:
        return _WORD_QUARTERS[match.group(1).replace("ё", "е")]
    return 0


def month_of(text: str) -> int:
    """Номер месяца из записи периода.

    Args:
        text: Запись периода в нижнем регистре.

    Returns:
        Номер месяца 1–12 либо 0, если месяц не назван.
    """
    lowered = text.lower().replace("ё", "е")
    for stem, number in _MONTHS:
        if stem in lowered:
            return number
    return 0


def key_of(raw: str) -> tuple[int, float, str]:
    """Ключ сортировки по одной записи периода.

    Args:
        raw: Запись периода: «2018», «13.01.2026», «2017-Q1», «конец июля».

    Returns:
        Кортеж для сортировки: категория, значение в шкале года, исходная
        запись. Текстовая запись без числа остаётся последней — порядок
        сохраняется хотя бы стабильным.
    """
    text = raw.strip()
    if not text:
        return (3, 0.0, "")
    # ISO-дата: 2026-01-13
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return (0, float(text[:10].replace("-", "")), text)
    # Год: 2018
    if text[:4].isdigit() and len(text) == 4:
        return (1, float(text), text)
    # Русская дата: 13.01.2026
    parts = [part for part in text.replace("г.", "").split()[0].split(".") if part.isdigit()]
    if len(parts) == 3:
        day, month, year = parts
        return (0, float(f"{year}{month:0>2}{day:0>2}"), text)
    # Квартал: «2017-Q1», «Q1 2017», «1 кв. 2017», «I квартал 2017».
    # Разбор по цифрам склеил бы «1 кв. 2017» в 12017 и поставил бы первый
    # квартал 2018 года раньше четвёртого квартала 2017-го.
    lowered = text.lower()
    quarter = quarter_of(lowered)
    if quarter:
        year_match = _YEAR.search(lowered)
        if year_match:
            return (1, float(year_match.group(1)) + quarter / 5, text)
    # Число в начале: конец июля, иное текстовое обозначение периода
    digits = "".join(char for char in text if char.isdigit())
    if digits:
        return (2, float(digits), text)
    return (3, 0.0, "")


def key_of_row(row: Mapping[str, str]) -> tuple[int, float, str]:
    """Ключ сортировки строки таблицы по её колонке периода.

    Args:
        row: Строка таблицы.

    Returns:
        Кортеж для сортировки; пустой, если период не назван.
    """
    for column in PERIOD_COLUMNS:
        raw = str(row.get(column, "")).strip()
        if raw:
            return key_of(raw)
    return (3, 0.0, "")


def months(text: str) -> set[int]:
    """Месяцы, названные в тексте без года.

    «Конец июля» — это период, но поставить его на шкалу года нечем: год не
    назван. Такие записи сравниваются между собой по совпадению месяца.

    Args:
        text: Текст утверждения или строка источника.

    Returns:
        Номера месяцев 1–12.
    """
    lowered = text.lower().replace("ё", "е")
    return {number for stem, number in _MONTHS if stem in lowered}


def _month_span(year: int, month: int, raw: str) -> Period:
    """Интервал месяца внутри года.

    Args:
        year: Год.
        month: Номер месяца 1–12.
        raw: Как записан период.

    Returns:
        Период длиной в месяц.
    """
    if not 1 <= month <= 12:
        return Period(float(year), float(year + 1), raw)
    start = year + (month - 1) / 12
    return Period(start, start + 1 / 12, raw)


def mask_dates(text: str) -> str:
    """Заменить записи дат пробелами, чтобы они не считались величинами.

    «На 1 января 2026 года — 11,923 млн абонентов» разбирается как три числа,
    и «1» от даты начинает сравниваться со значениями источника: честное
    утверждение отклонялось «величина 1 против 11.923» (пара `p-025`). Год,
    месяц и день — это период, а не величина.

    Args:
        text: Исходный текст.

    Returns:
        Текст, в котором даты заменены пробелами.
    """
    masked = _DATE_ISO.sub(" ", text)
    masked = _DATE_RU.sub(" ", masked)
    for stem, _number in _MONTHS:
        masked = re.sub(rf"\b\d{{1,2}}\s+{stem}\w*", " ", masked)
    return masked


def spans(text: str) -> list[Period]:
    """Периоды, названные в тексте, как интервалы в шкале года.

    Разбираются даты («13.01.2026»), кварталы («1 кв. 2017», «2017-Q1»,
    «I квартал 2017»), месяцы с годом («сентябрь 2019»), годы («2019») и
    интервалы («с 2013 по сентябрь 2019»).

    **Год без квартала и месяца — это интервал длиной в год**, а не точка:
    утверждение «за 2019 год» покрывает и «2019». «Накопленно с 2013 года»
    без конца интервала даёт год начала, а не «всё время после»: иначе
    накопленный итог подтверждал бы значение любого года внутри накопления —
    ровно та подмена, из-за которой пара `p-032` признавалась верной.

    Args:
        text: Текст утверждения или строка источника.

    Returns:
        Периоды в порядке появления, без повторов.
    """
    lowered = text.lower().replace("ё", "е").replace("\u00a0", " ")
    found: list[Period] = []

    for match in _DATE_ISO.finditer(lowered):
        found.append(_month_span(int(match.group(1)), int(match.group(2)), match.group(0)))
    for match in _DATE_RU.finditer(lowered):
        found.append(_month_span(int(match.group(3)), int(match.group(2)), match.group(0)))

    # Интервал: «с 2013 по сентябрь 2019», «с 2013 года до 2019».
    for match in re.finditer(r"\bс\s+(\d{4})\b.{0,40}?\b(?:по|до)\s+(.{0,20}?\d{4})", lowered):
        year_match = _YEAR.search(match.group(2))
        if not year_match:
            continue
        end_year = int(year_match.group(1))
        month = month_of(match.group(2))
        end = end_year + (month - 1) / 12 if month else end_year + 1
        found.append(Period(float(match.group(1)), float(end), match.group(0).strip()))

    for match in _YEAR.finditer(lowered):
        year = int(match.group(1))
        window = lowered[max(0, match.start() - 24) : match.end() + 24]
        quarter = quarter_of(window)
        if quarter:
            found.append(Period(year + quarter / 5, year + quarter / 5, window.strip()))
            continue
        month = month_of(window)
        if month:
            found.append(_month_span(year, month, window.strip()))
            continue
        found.append(Period(float(year), float(year + 1), match.group(0)))

    unique: list[Period] = []
    for period in found:
        if not any((period.start, period.end) == (seen.start, seen.end) for seen in unique):
            unique.append(period)
    return unique


def share(left: Sequence[Period], right: Sequence[Period]) -> float | None:
    """Насколько периоды совпадают: 1 при пересечении, затухание при разрыве.

    Args:
        left: Периоды утверждения.
        right: Периоды источника.

    Returns:
        Доля от 0 до 1; ``None``, если период не назван хотя бы с одной
        стороны — тогда сверять нечего, и это не отказ.
    """
    if not left or not right:
        return None
    best = 0.0
    for claim_period in left:
        for source_period in right:
            if claim_period.overlaps(source_period):
                return 1.0
            best = max(best, math.exp(-claim_period.gap(source_period) / GAP_DECAY_YEARS))
    return best


def described(items: Iterable[Period]) -> str:
    """Показать периоды для причины отказа.

    Args:
        items: Периоды.

    Returns:
        Строка вида ``2019, 2019.667`` или «не назван».
    """
    shown = [f"{period.start:g}" for period in list(items)[:3]]
    return ", ".join(shown) if shown else "не назван"
