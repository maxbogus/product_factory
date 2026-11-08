"""Количественный слой: слова-числа и единицы → величина.

Зачем модуль. Числа сравнивались только тогда, когда записаны цифрами
(`pf.lexical.numbers_match`). «Почти сорок миллионов абонентов» не разбирал
никто: измерено, что такая пара отклоняется при покрытии 0,17, хотя
подтверждение честное (`eval/pairs/out_of_place.jsonl`, пара `p-049`).

**Направление одно: слова → число.** Цифры в пропись не переводятся — на
наборе из 60 пар это дало бы +217 токенов к цитатам и потерю точности:
`172,87` превратилось бы в «сто семьдесят две целых восемьдесят семь сотых».
Величина становится **полем**, а не токеном: в мешке слов её нет, поэтому
число больше не даёт ложного сходства с чужой строкой, где стоит то же число
(это была прямая причина утечки `p-032`).

Модификатор — часть смысла, а не украшение: «более восьмидесяти» при значении
78,79 не подтверждается, «чуть ниже восьмидесяти» — подтверждается. Поэтому
модификатор хранится в величине, а не выбрасывается.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pf.ground import normalize
from pf.lexical import number_tolerance

#: Числа словами: количественные и падежные формы, которые встречаются в тексте.
_NUMBER_WORDS: dict[str, float] = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "одного": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "три": 3,
    "трёх": 3,
    "трех": 3,
    "четыре": 4,
    "четырёх": 4,
    "четырех": 4,
    "пять": 5,
    "пяти": 5,
    "шесть": 6,
    "шести": 6,
    "семь": 7,
    "семи": 7,
    "восемь": 8,
    "восьми": 8,
    "девять": 9,
    "девяти": 9,
    "десять": 10,
    "десяти": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "двадцати": 20,
    "тридцать": 30,
    "тридцати": 30,
    "сорок": 40,
    "сорока": 40,
    "пятьдесят": 50,
    "пятидесяти": 50,
    "шестьдесят": 60,
    "шестидесяти": 60,
    "семьдесят": 70,
    "семидесяти": 70,
    "восемьдесят": 80,
    "восьмидесяти": 80,
    "девяносто": 90,
    "девяноста": 90,
    "сто": 100,
    "ста": 100,
    "двести": 200,
    "двухсот": 200,
    "триста": 300,
    "трёхсот": 300,
    "трехсот": 300,
    "четыреста": 400,
    "четырёхсот": 400,
    "четырехсот": 400,
    "пятьсот": 500,
    "пятисот": 500,
    "шестьсот": 600,
    "шестисот": 600,
    "семьсот": 700,
    "семисот": 700,
    "восемьсот": 800,
    "восьмисот": 800,
    "девятьсот": 900,
    "девятисот": 900,
}

#: Шкала: слово → множитель.
_SCALES: dict[str, float] = {
    "тыс": 1_000,
    "тысяч": 1_000,
    "тысяча": 1_000,
    "тысячи": 1_000,
    "млн": 1_000_000,
    "миллион": 1_000_000,
    "миллиона": 1_000_000,
    "миллионов": 1_000_000,
    "млрд": 1_000_000_000,
    "миллиард": 1_000_000_000,
    "миллиарда": 1_000_000_000,
    "миллиардов": 1_000_000_000,
}

#: Единицы: слово → код. Проценты, рубли и штуки несравнимы между собой.
_UNITS: dict[str, str] = {
    "%": "%",
    "процент": "%",
    "процента": "%",
    "процентов": "%",
    "п.п.": "п.п.",
    "п.п": "п.п.",
    "руб": "руб",
    "рубль": "руб",
    "рубля": "руб",
    "рублей": "руб",
    "₽": "руб",
    "абонент": "шт",
    "абонента": "шт",
    "абонентов": "шт",
    "человек": "шт",
    "штук": "шт",
}

#: Единицы, к которым не применим масштаб «млн»/«тыс»: процент и рубль.
_DIMENSIONLESS = frozenset({"%", "п.п.", "руб"})

#: Сроки: величиной не считаются (то же правило, что у чисел плана).
_TIME = frozenset(
    {
        "месяц",
        "месяца",
        "месяцев",
        "год",
        "года",
        "лет",
        "день",
        "дня",
        "дней",
        "неделя",
        "недели",
        "недель",
        "квартал",
        "квартала",
        "кварталов",
    }
)

#: Модификаторы точности: слово → как понимать. Длинные сочетания — первыми.
_MODIFIERS: tuple[tuple[str, str], ...] = (
    ("чуть ниже", "менее"),
    ("чуть выше", "более"),
    ("чуть больше", "более"),
    ("чуть меньше", "менее"),
    ("почти", "почти"),
    ("около", "почти"),
    ("примерно", "почти"),
    ("порядка", "почти"),
    ("более", "более"),
    ("больше", "более"),
    ("выше", "более"),
    ("свыше", "более"),
    ("менее", "менее"),
    ("меньше", "менее"),
    ("ниже", "менее"),
)

#: Насколько «почти» и «около» разрешают промахнуться: доля от величины.
APPROXIMATION_SHARE = 0.05

#: Число цифрами. Даты («13.01.2026») под шаблон не подходят намеренно.
_DIGITS = re.compile(r"^\d+(?:[.,]\d+)?$")

#: Знаки, которые прилипают к слову и к содержанию не относятся.
_EDGE = ".,;:()«»\"'"


@dataclass(frozen=True, slots=True)
class Quantity:
    """Величина из текста.

    Attributes:
        value: Число в той шкале, в какой оно записано (40 для «сорок миллионов»).
        scale: Множитель шкалы: 1, 1e3, 1e6, 1e9.
        unit: Единица: ``%``, ``п.п.``, ``руб``, ``шт`` либо пусто, если не названа.
        modifier: Модификатор точности: пусто, ``почти``, ``более``, ``менее``.
        raw: Исходное написание — для объяснения отказа.
    """

    value: float
    scale: float
    unit: str
    modifier: str
    raw: str


def _tokens(text: str) -> list[str]:
    """Разбить текст на значимые токены без прилипшей пунктуации.

    Args:
        text: Исходный текст.

    Returns:
        Список токенов в нижнем регистре.
    """
    return [stripped for token in normalize(text).split() if (stripped := token.strip(_EDGE))]


def _modifier_at(tokens: list[str], index: int) -> tuple[str, int]:
    """Найти модификатор точности в позиции.

    Args:
        tokens: Токены текста.
        index: Позиция.

    Returns:
        Пара «значение модификатора» и число съеденных слов.
    """
    if index >= len(tokens):
        return "", 0
    pair = " ".join(tokens[index : index + 2]).lower()
    for phrase, meaning in _MODIFIERS:
        if " " in phrase and pair.startswith(phrase):
            return meaning, 2
    single = tokens[index].lower()
    for phrase, meaning in _MODIFIERS:
        if " " not in phrase and single == phrase:
            return meaning, 1
    return "", 0


def _number_words_at(tokens: list[str], index: int) -> tuple[float | None, int]:
    """Собрать число, записанное словами («триста тридцать» → 330).

    Args:
        tokens: Токены текста.
        index: Позиция.

    Returns:
        Пара «значение» и число съеденных слов; ``None``, если числа нет.
    """
    total = 0.0
    consumed = 0
    while index + consumed < len(tokens):
        value = _NUMBER_WORDS.get(tokens[index + consumed].lower())
        if value is None:
            break
        total += value
        consumed += 1
    if not consumed:
        return None, 0
    return total, consumed


def _scale_unit_at(tokens: list[str], index: int) -> tuple[float, str, bool, int]:
    """Прочитать шкалу и единицу после числа.

    Args:
        tokens: Токены текста.
        index: Позиция сразу после числа.

    Returns:
        Шкала, единица, признак срока и число съеденных слов.
    """
    scale = 1.0
    unit = ""
    timed = False
    consumed = 0
    while index + consumed < len(tokens) and consumed < 2:
        token = tokens[index + consumed].lower()
        if token in _SCALES and scale == 1.0:
            scale = _SCALES[token]
        elif token in _UNITS and not unit:
            unit = _UNITS[token]
        elif token in _TIME:
            timed = True
        else:
            break
        consumed += 1
    return scale, unit, timed, consumed


def parse_quantities(text: str) -> list[Quantity]:
    """Разобрать величины текста: цифрами и словами, с единицей и модификатором.

    Args:
        text: Исходный текст.

    Returns:
        Величины в порядке появления. Сроки («3 месяца») и даты не считаются
        величинами.
    """
    tokens = _tokens(text)
    found: list[Quantity] = []
    index = 0
    while index < len(tokens):
        modifier, modifier_len = _modifier_at(tokens, index)
        position = index + modifier_len
        if position >= len(tokens):
            break

        raw: str = " ".join(tokens[index : position + 1])
        if _DIGITS.match(tokens[position]):
            value = float(tokens[position].replace(",", "."))
            scale, unit, timed, extra = _scale_unit_at(tokens, position + 1)
            index = position + 1 + extra
            if not timed:
                found.append(Quantity(value, scale, unit, modifier, raw))
            continue

        number, consumed = _number_words_at(tokens, position)
        if number is not None:
            scale, unit, timed, extra = _scale_unit_at(tokens, position + consumed)
            index = position + consumed + extra
            if not timed:
                found.append(Quantity(number, scale, unit, modifier, raw))
            continue

        index += 1
    return found


def _compatible(claim: Quantity, source: Quantity) -> bool:
    """Сравнимы ли единицы величин.

    Args:
        claim: Величина утверждения.
        source: Величина источника.

    Returns:
        Единицы совместимы (или одна из сторон единицу не назвала и масштаб
        не противоречит).
    """
    if claim.unit and source.unit and claim.unit != source.unit:
        return False
    # «7,39 млн» не может означать процент или рубль: масштаб — признак счёта.
    return not (
        claim.unit in _DIMENSIONLESS and (source.unit == "шт" or source.scale > 1.0)
    ) and not (source.unit in _DIMENSIONLESS and (claim.unit == "шт" or claim.scale > 1.0))


def _values(claim: Quantity, source: Quantity) -> tuple[float, float]:
    """Привести величины к одной шкале.

    Args:
        claim: Величина утверждения.
        source: Величина источника.

    Returns:
        Пара чисел для сравнения. Если обе стороны считают штуки, сравнение
        идёт в базовой шкале; иначе — как записано.
    """
    if claim.unit == source.unit == "шт":
        return claim.value * claim.scale, source.value * source.scale
    return claim.value, source.value


def _matches(
    claim: Quantity,
    source: Quantity,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> bool:
    """Совпадает ли величина источника с величиной утверждения.

    Args:
        claim: Величина утверждения.
        source: Величина источника.
        abs_tolerance: Абсолютный допуск округления.
        rel_tolerance: Относительный допуск округления.

    Returns:
        Совпадение с учётом модификатора точности.
    """
    left, right = _values(claim, source)
    tolerance = number_tolerance(right, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance)
    if claim.modifier == "почти":
        tolerance = max(tolerance, APPROXIMATION_SHARE * abs(left))
    if claim.modifier == "более":
        return right >= left - tolerance
    if claim.modifier == "менее":
        return right <= left + tolerance
    return abs(left - right) <= tolerance


def _shown(items: list[Quantity]) -> str:
    """Показать значения источника для причины отказа.

    Args:
        items: Величины источника.

    Returns:
        Строка вида ``38.9, 2024``.
    """
    return ", ".join(f"{item.value:g}" for item in items[:3])


def compare_quantities(
    claim: str,
    passage: str,
    *,
    abs_tolerance: float = 0.1,
    rel_tolerance: float = 5e-4,
) -> tuple[bool, str]:
    """Сверить величины утверждения с величинами источника.

    Каждая величина утверждения должна найтись в источнике: с той же единицей,
    в той же шкале и с учётом модификатора («более», «менее», «почти»).

    Args:
        claim: Утверждение.
        passage: Абзац или строка источника.
        abs_tolerance: Абсолютный допуск округления.
        rel_tolerance: Относительный допуск округления.

    Returns:
        Пара «совпало, пояснение». Пояснение заполняется при расхождении и
        называет причину — единицу, модификатор или число.
    """
    claim_items = parse_quantities(claim)
    source_items = parse_quantities(passage)
    if not claim_items:
        return True, "в утверждении нет величин"
    if not source_items:
        return False, "в источнике нет величин: сверить не с чем"

    for item in claim_items:
        if any(
            _matches(item, candidate, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance)
            for candidate in source_items
            if _compatible(item, candidate)
        ):
            continue
        incompatible = [candidate for candidate in source_items if not _compatible(item, candidate)]
        if incompatible:
            return False, f"единицы не совпадают: {item.raw} против {incompatible[0].raw}"
        if item.modifier == "более":
            return False, (
                f"«более» не подтверждается: в источнике {_shown(source_items)}, "
                f"а утверждение требует больше {item.value:g}"
            )
        if item.modifier == "менее":
            return False, (
                f"«менее» не подтверждается: в источнике {_shown(source_items)}, "
                f"а утверждение требует меньше {item.value:g}"
            )
        return False, f"величина {item.value:g} не совпала с источником ({_shown(source_items)})"
    return True, f"величины совпали: {len(claim_items)}"
