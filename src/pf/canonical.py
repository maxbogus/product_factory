"""Каноническое сравнение: поля вместо одного порога покрытия.

Зачем модуль. Решение «основание подтверждено» принималось одним порогом по
покрытию (`pf.lexical.overlap`, 0,8). Из этого следовали оба измеренных
провала сразу:

- **ложный отказ:** «почти сорок миллионов» против строки «значение: 38.9» —
  покрытие 0,17, отказ, хотя подтверждение честное (`p-049`);
- **утечка:** «27,4 млн за 2016 год» против строки «накопленно с 2013 года,
  период: сентябрь 2019» — покрытие 1,00, подтверждение, хотя период другой
  (`p-032`).

Порог не различал эти случаи, потому что числа, единицы и период сравнивались
как проза. Здесь они сравниваются как **поля**: понятие показателя, субъект,
регион, период (`pf.periods`), величины (`pf.quantities`), точные термины,
отрицание и опора утверждения (одна строка или сводка).

**Ответ — число, а не вердикт.** Каждое поле даёт долю от 0 до 1, итог —
произведение долей в степенях весов (`FIELD_WEIGHTS`): свидетельства
перемножаются, поэтому провал решающего поля не размывается хорошими
остальными. Число важно само по себе: «39,4 против 40» и «7,39 против 3,9» —
это 0,96 и 0,02, а не два одинаковых «нет». Порог, при котором число
превращается в статус, — настройка (`claim_support_threshold`), а не константа.

**Чего модуль не делает.** Он не участвует в проверке дословности: цитата
сверяется с сырым текстом (`pf.ground.find_quote`, `D-38`). Он не поднимает
статус выгрузок и переводов (`D-29`, `D-27`): статус по-прежнему ставит только
`grounding-gate`. И он не заменяет расчёт: агрегат подтверждается пересчётом по
имени (`D-35`), а не совпадением полей.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from pf import domain
from pf.analysis import parse_number
from pf.config import settings
from pf.entities import KIND_AMBIGUOUS, KIND_MISMATCH, Identity
from pf.entities import identity as entity_identity
from pf.lexical import (
    GENERIC_WORDS,
    has_negation,
    lemma,
    normalize_homoglyphs,
    number_tolerance,
    stems,
    term_present,
    translit_key,
    words,
)
from pf.lexical import overlap as lexical_overlap
from pf.periods import Period, described, mask_dates, months, spans
from pf.periods import share as period_share
from pf.quantities import APPROXIMATION_SHARE, Quantity, compatible, parse_quantities
from pf.sanitize import for_comparison

#: Веса полей в итоговой степени поддержки.
#:
#: Веса априорные, из таксономии провалов замера (`D-39`): подмена оператора —
#: 12 случаев из 30, показателя — 9, периода — 4, единицы — 2. Подгонять семь
#: весов на тридцати парах значило бы выучить набор, поэтому веса не
#: подбираются, а называются. Порог — подбирается и живёт в настройке.
FIELD_WEIGHTS: dict[str, float] = {
    "показатель": 2.0,
    "субъект": 2.0,
    "период": 1.5,
    "величина": 1.5,
    "опора": 1.5,
    "регион": 1.0,
    "термин": 1.0,
    "отрицание": 1.0,
}

#: Порядок полей в объяснении: сначала решающие.
FIELD_ORDER: tuple[str, ...] = (
    "показатель",
    "субъект",
    "период",
    "величина",
    "опора",
    "регион",
    "термин",
    "отрицание",
)

#: Нижняя граница доли поля: провал остаётся различимым по величине, а не
#: превращается в ноль. Тот же пол, что у величин и терминов.
SHARE_FLOOR = 0.02

#: Поле таблицы: имя колонки и двоеточие после него. Длинные имена — первыми.
_FIELD_LABELS: tuple[str, ...] = (
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

#: Разделитель полей строки таблицы.
_FIELD_SPLIT = re.compile(r"\b(" + "|".join(_FIELD_LABELS) + r")\s*:\s*", re.IGNORECASE)

#: Слова шкалы: «млн» в шапке колонки относится к значению, а не к соседнему слову.
_SCALE_WORDS: tuple[tuple[str, float], ...] = (
    ("тыс", 1_000.0),
    ("тысяч", 1_000.0),
    ("млн", 1_000_000.0),
    ("миллион", 1_000_000.0),
    ("млрд", 1_000_000_000.0),
    ("миллиард", 1_000_000_000.0),
)

#: Единицы, названные в шапке колонки. Счёт («шт») здесь намеренно не
#: выводится: «рост абонентской базы» — это проценты, хотя слово «абонент»
#: в шапке есть. Проценты, рубли и п.п. шапка называет прямо.
_UNIT_WORDS: tuple[tuple[str, str], ...] = (
    ("процент", "%"),
    ("доля", "%"),
    ("%", "%"),
    ("п.п", "п.п."),
    ("рубл", "руб"),
    ("руб", "руб"),
    ("₽", "руб"),
)


#: Субъекты, которые называет утверждение. Список приходит из домена проекта
#: (`knowledge/domain/entities.yaml`), а не из кода: иначе второй проект на
#: другом домене потребовал бы правки модуля (`T-67`). Порядок — порядок файла.
def subjects() -> tuple[str, ...]:
    """Имена субъектов домена.

    Returns:
        Канонические имена в порядке файла домена.
    """
    return tuple(subject.canonical for subject in domain.subjects())


#: Регионы как леммы: «России» и «Россия» — одно место. Усечение до шести
#: букв их не связывает («россия» и «россии» — разные обрезки), поэтому здесь
#: лемма, а не основа.
REGION_LEMMAS: frozenset[str] = frozenset(
    {"россия", "беларусь", "казахстан", "украина", "санкт-петербург", "петербург", "москва"}
)

#: Слова, которые не называют предмет (основы). Служебные и общие.
STOP_CONCEPTS: frozenset[str] = frozenset(
    {word[:6] for word in GENERIC_WORDS if len(word) >= 5}
) | frozenset(
    {
        "соста",
        "состав",
        "являт",
        "имеет",
        "дости",
        # Модификаторы точности и связки: «около тридцати миллионов» — это
        # число 30, а не понятие «около».
        "около",
        "пример",
        "почти",
        "более",
        "менее",
        "свыше",
        # В телеком-тексте «связь» есть почти везде: различает мобильную и
        # фиксированную не она, а слово «мобильный»/«фиксированный».
        "связи",
        "связь",
        "связью",
        "пользу",
        "пользо",
    }
)

#: Основы единиц: «рублей на абонента в месяц» называет единицу, а не
#: показатель. Без этого утверждение об ARPU не подтверждалось, потому что не
#: упоминало «рублей» и «месяц» буквально (пара `p-005`).
UNIT_CONCEPTS: frozenset[str] = frozenset().union(
    *(
        stems(word)
        for word in (
            "рублей",
            "рубль",
            "процентов",
            "процент",
            "месяц",
            "месяцев",
            "тысяч",
            "миллионов",
            "миллиарда",
        )
    )
)

#: Признаки утверждения о динамике: одна точка их не подтверждает.
TREND_WORDS: tuple[str, ...] = (
    "пада",
    "упал",
    "сокращ",
    "растёт",
    "растет",
    "растут",
    "вырос",
    "увеличи",
    "снижа",
    "снизи",
    "динамик",
    "тренд",
    "лидир",
)

#: Признаки утверждения о сводке: строка одного субъекта её не подтверждает.
AGGREGATE_WORDS: tuple[str, ...] = ("средн", "по рынку", "в целом", "всего по")

#: Токен точного термина: буквы и цифры вместе («Tele2», «МТС», «FTTB»).
_TERM_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]*")

#: Скобка в тексте: примечание в названии метрики или сам показатель в абзаце.
_PARENTHESIS = re.compile(r"\(([^)]{1,200})\)")


@dataclass(frozen=True, slots=True)
class CanonicalText:
    """Текст, разобранный на поля.

    Attributes:
        raw: Исходный текст.
        declared: Поля строки таблицы («оператор», «показатель», …) — пусто
            у прозы.
        concepts: Основы значимых слов, кроме региона и служебных.
        terms: Точные термины: аббревиатуры, бренды, латиница с цифрами.
        quantities: Величины с единицей и шкалой.
        periods: Периоды как интервалы в шкале года.
        month_tags: Месяцы без года: «конец июля».
        regions: Основы названных регионов.
        subjects: Субъекты домена, названные в тексте.
        negated: Есть ли отрицание.
    """

    raw: str
    declared: Mapping[str, str]
    concepts: frozenset[str]
    terms: tuple[str, ...]
    quantities: tuple[Quantity, ...]
    periods: tuple[Period, ...]
    month_tags: frozenset[int]
    regions: frozenset[str]
    subjects: tuple[str, ...]
    negated: bool


@dataclass(frozen=True, slots=True)
class FieldResult:
    """Исход сверки одного поля.

    Attributes:
        share: Доля от 0 до 1 — степень совпадения по этому полю.
        note: Что именно не сошлось — словами, для журнала.
    """

    share: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class SupportScore:
    """Степень поддержки утверждения источником.

    Attributes:
        support: Итоговая степень от 0 до 1. Порог, при котором она
            превращается в статус, задаётся настройкой, а не здесь.
        fields: Исход по каждому полю.
        coverage: Прежняя доля покрытия значимых слов — сообщается, но
            решением больше не является.
        oov_tokens: Нераспознанные слова утверждения — в очередь распознания.
        reason: Причина отказа: называет поле, а не «не совпало».
        entity_share: Доля прочтений утверждения, поддержанных источником.
        entity_kind: Исход сравнения прочтений: пусто, «другая сущность» или
            «прочтение не установлено» (`pf.entities`).
        entity_note: Причина словами — что названо, сколько прочтений и где
            проверить.
    """

    support: float
    fields: Mapping[str, FieldResult]
    coverage: float
    oov_tokens: tuple[str, ...]
    reason: str
    entity_share: float = 1.0
    entity_kind: str = ""
    entity_note: str = ""
    support_without_subject: float = 1.0

    def supported(self, threshold: float) -> bool:
        """Превратить степень в решение по настройке.

        Args:
            threshold: Порог: ниже него утверждение не подтверждается.

        Returns:
            Подтверждено ли утверждение.
        """
        return self.support >= threshold

    def explain(self) -> str:
        """Показать вклады полей одной строкой.

        Returns:
            Строка вида ``поддержка 0,87 · период 0,05``.
        """
        worst = sorted(
            ((name, result.share) for name, result in self.fields.items() if result.share < 1.0),
            key=lambda item: (FIELD_ORDER.index(item[0]), item[1]),
        )
        parts = [f"поддержка {self.support:.2f}"]
        parts.extend(f"{name} {share:.2f}" for name, share in worst[:3])
        return " · ".join(parts)


def parse_fields(text: str) -> dict[str, str]:
    """Разобрать строку таблицы на поля по именам колонок.

    Args:
        text: Строка вида «оператор: Оператор А, показатель: …, значение: 82».

    Returns:
        Отображение «поле → значение». Пусто, если имён колонок меньше двух:
        одно двоеточие в прозе — это не таблица.
    """
    matches = list(_FIELD_SPLIT.finditer(text))
    if len(matches) < 2:
        return {}
    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip().rstrip(",").strip()
        parsed.setdefault(match.group(1).lower(), value)
    return parsed


def column_semantics(declared: Mapping[str, str]) -> tuple[float, str]:
    """Шкала и единица значения из шапки колонки.

    «Показатель: абонентская база, значение: 38.9» — значение в миллионах,
    даже если слово «млн» стоит далеко от числа. Без этого «почти сорок
    миллионов» против 38.9 сравнивались бы как 40 и 38.9 (так и выходило:
    пара `p-049` проходила по случайности, а не по правилу).

    Args:
        declared: Поля строки таблицы.

    Returns:
        Пара «шкала, единица»: 1e6 либо 1.0, ``%``/``руб``/``шт`` либо пусто.
    """
    if "значение_процентов" in declared:
        return 1.0, "%"
    header = " ".join(
        declared.get(name, "") for name in ("показатель", "значение", "тариф")
    ).lower()
    scale = 1.0
    for word, multiplier in _SCALE_WORDS:
        if word in header:
            scale = multiplier
            break
    unit = ""
    for word, code in _UNIT_WORDS:
        if word in header:
            unit = code
            break
    return scale, unit


def _lemmas(text: str) -> frozenset[str]:
    """Леммы текста: «России» и «Россия» — одно место.

    Мешок слов (`lexical.words`) выбрасывает слова короче четырёх букв,
    поэтому аббревиатуры в леммах нет: короткие названия сравниваются
    отдельно, по подстроке (`_mentions`).

    Args:
        text: Очищенный текст.

    Returns:
        Множество лемм.
    """
    return frozenset(words(text))


def _indicator_concepts(text: str) -> frozenset[str]:
    """Понятия названия показателя: без содержания скобок.

    Скобка в имени метрики — примечание («абонентская база
    (консолидированная)»), а не второе название. Требовать её от утверждения
    значило бы отклонять «абонентская база 87 млн» за отсутствие слова
    «консолидированная» (пара `p-053`). Правило отбрасывать скобки —
    `lexical.stems`, второе здесь не заводится.

    Args:
        text: Название показателя.

    Returns:
        Множество основ.
    """
    return frozenset(stems(text)) - STOP_CONCEPTS - UNIT_CONCEPTS


def _passage_concepts(text: str) -> frozenset[str]:
    """Понятия текста: содержание скобок входит.

    В абзаце скобка часто несёт сам показатель: «…11,923 млн абонентов
    (проникновение — более 130 %)». Без неё честное утверждение о
    проникновении не подтверждалось (пара `p-037`).

    Args:
        text: Очищенный текст.

    Returns:
        Множество основ.
    """
    notes = " ".join(_PARENTHESIS.findall(text))
    return _indicator_concepts(text) | _indicator_concepts(notes)


def _regions(text: str) -> frozenset[str]:
    """Леммы названных регионов.

    Args:
        text: Очищенный текст.

    Returns:
        Множество лемм регионов.
    """
    return frozenset(_lemmas(text) & REGION_LEMMAS)


def strict_terms(text: str) -> list[str]:
    """Точные термины текста: аббревиатуры, бренды и латиница с цифрами.

    Правило узкое намеренно. Обычное слово — понятие: «абонентскую» и
    «абонентов» связывает основа (`pf.lexical.stems`). Термин же означает
    конкретное имя или обозначение, и подмена имени на другое имя — это
    подмена, а не пересказ.

    Аббревиатура короче четырёх букв в мешок слов не попадает вовсе
    (`corpus._TOKEN_PATTERN` — `[а-яёa-z]{4,}`), поэтому до проверки терминов
    она раньше не доходила, и утверждение с аббревиатурой отклонялось всегда
    («термин не найден»), даже когда строка начиналась с «оператор: …».
    Здесь у терминов свой разбор.

    Args:
        text: Текст утверждения или строки источника.

    Returns:
        Термины в исходном написании, без повторов.
    """
    found: list[str] = []
    for raw in _TERM_TOKEN.findall(text):
        letters = [char for char in raw if char.isalpha()]
        if len(letters) < 2:
            continue
        upper = all(char.isupper() for char in letters)
        latin_with_digit = any(char.isdigit() for char in raw) and all(
            char.isascii() for char in raw
        )
        if not (upper or latin_with_digit):
            continue
        if raw not in found:
            found.append(raw)
    return found


def _mentions(subject: str, lowered: str, lemmas: frozenset[str]) -> bool:
    """Назван ли субъект в тексте.

    Односложные названия (аббревиатура из двух-трёх букв, латиница с цифрой)
    ищутся подстрокой: мешок слов их не видит вовсе (слова короче четырёх
    букв), а лемматизатор не знает.
    Составные (категория из двух слов) — по леммам: иначе «виртуальных
    операторов» не совпадёт с названием из списка, и строка чужого субъекта
    подтверждала бы утверждение.

    Args:
        subject: Название субъекта из списка.
        lowered: Текст в нижнем регистре без провенанса.
        lemmas: Леммы текста.

    Returns:
        Назван ли субъект.
    """
    normalized = normalize_homoglyphs(subject.lower().replace("ё", "е"))
    parts = [part for part in re.split(r"[^0-9a-zа-я]+", normalized) if part]
    if len(parts) <= 1:
        return normalized in lowered
    return all(lemma(part) in lemmas for part in parts)


def _subjects(text: str) -> tuple[str, ...]:
    """Субъекты домена, названные в тексте.

    Args:
        text: Очищенный текст.

    Returns:
        Названия субъектов в порядке списка домена.
    """
    lowered = normalize_homoglyphs(text.lower().replace("ё", "е"))
    lemmas = _lemmas(text)
    return tuple(subject for subject in subjects() if _mentions(subject, lowered, lemmas))


def _is_year(item: Quantity) -> bool:
    """Год — это период, а не величина.

    «В 2016 году — 27,4 млн» разбирается как два числа, но год сравнивать с
    числами источника как величину нельзя: значение сходится, а утверждение
    отклонялось бы «за год среди значений» (пара `p-032`). Год проверяет поле
    периода.

    Args:
        item: Величина из текста.

    Returns:
        Похожа ли величина на запись года.
    """
    if not 1800 <= item.value <= 2100:
        return False
    return item.raw.strip().rstrip(",") == f"{item.value:g}"


def canonicalize(text: str) -> CanonicalText:
    """Разобрать текст на поля.

    Сначала читаются поля строки таблицы (`parse_fields`) — пока метки
    колонок на месте: чистка снимает «оператор:» и «период:», и после неё
    «сентябрь 2019» уже не отличить от случайного месяца в тексте. Дальше
    остальное разбирается по очищенному тексту (`sanitize.for_comparison`):
    адрес источника в сравнении не участвует (`D-38`).

    Args:
        text: Текст утверждения, строка таблицы или абзац источника.

    Returns:
        Поля текста.
    """
    declared = parse_fields(text)
    clean = for_comparison(text)
    value_quantities: list[Quantity] = []
    if "значение" in declared or "значение_процентов" in declared:
        raw_value = declared.get("значение") or declared.get("значение_процентов", "")
        number = parse_number(raw_value)
        if number is not None:
            scale, unit = column_semantics(declared)
            value_quantities.append(Quantity(number, scale, unit, "", raw_value))
    quantities = tuple(value_quantities) or tuple(
        item for item in parse_quantities(mask_dates(clean)) if not _is_year(item)
    )
    period_spans: list[Period] = list(spans(clean))
    period_text = declared.get("период", "")
    if period_text:
        period_spans.extend(spans(period_text))
    unique_periods: list[Period] = []
    for period in period_spans:
        if not any((period.start, period.end) == (seen.start, seen.end) for seen in unique_periods):
            unique_periods.append(period)
    # Величины не участвуют в понятиях: «сто тридцать» — это число 130, а не
    # понятие «тридца». Иначе утверждение не подтверждалось из-за того, что
    # слова-числа не нашлись среди слов абзаца (пара `p-037`).
    concept_text = clean
    for item in quantities:
        if item.raw:
            concept_text = concept_text.replace(item.raw.lower(), " ")
    return CanonicalText(
        raw=text,
        declared=declared,
        concepts=_passage_concepts(concept_text),
        # Термины разбираются по **сырому** тексту: чистка приводит к нижнему
        # регистру, а аббревиатура в нижнем регистре перестаёт быть
        # аббревиатурой. Пока термины брались после чистки, «МТС» не был
        # термином вовсе, и подмена бренда держалась только на списке
        # субъектов (`D-45`).
        terms=tuple(strict_terms(text)),
        quantities=quantities,
        periods=tuple(unique_periods),
        month_tags=frozenset(months(clean)),
        regions=_regions(clean),
        subjects=_subjects(clean),
        negated=has_negation(clean),
    )


def _stem_groups() -> Mapping[str, str]:
    """Отображение «основа → каноническая основа» из словаря синонимов.

    Нужно, чтобы «клиенты» и «абоненты» считались одним понятием с частичным
    весом: это не точный термин, но и не разные понятия.

    Кэша здесь нет намеренно: словарь подтверждает человек, и пополнение
    должно действовать сразу. Пока таблица кэшировалась, сценарий
    «подтверждённый синоним расширяет близость» оставался красным — синоним
    добавлялся, а сравнение видело старую таблицу.

    Returns:
        Отображение основ. Импорт ленивый: `pf.lexicon` читает файлы.
    """
    from pf.lexicon import groups

    table: dict[str, str] = {}
    for word_lemma, canonical in groups().items():
        for word in (word_lemma, canonical):
            if len(word) >= 5:
                table.setdefault(word[:6], canonical[:6])
    return table


def _credit(concept: str, other: CanonicalText, table: Mapping[str, str]) -> float:
    """Насколько понятие подтверждается текстом другой стороны.

    Args:
        concept: Основа понятия.
        other: Другая сторона сравнения.
        table: Отображение «основа → каноническая основа» из словаря.

    Returns:
        1,0 — то же понятие; 0,7 — синоним из словаря; 0,0 — нет.
    """
    if concept in other.concepts:
        return 1.0
    mine = table.get(concept)
    if mine and any(table.get(candidate) == mine for candidate in other.concepts):
        return 0.7
    return 0.0


def _short_source(text: str) -> bool:
    """Короткий ли источник: в нём каждое слово — содержание.

    На длинном абзаце покрытие слов ничего не решает: измерено, что подмена
    периода даёт покрытие 1,00, а честный пересказ — 0,17 (`D-39`). На
    источнике из одного предложения всё наоборот: «Компания обслуживает
    абонентов связи» против «…абонентов криптовалютой» — это не пересказ, а
    другое утверждение, и покрытие его ловит.

    Args:
        text: Текст источника.

    Returns:
        Не больше пяти значимых слов: одно короткое предложение без контекста.
    """
    return len(words(text)) <= 5


def _concepts_field(claim: CanonicalText, source: CanonicalText) -> FieldResult:
    """Показатель: называет ли утверждение то понятие, что стоит в источнике.

    Направление сверки не то, что у покрытия. Когда источник сам называет
    показатель колонкой («показатель: абонентская база»), проверяется, что
    утверждение этот показатель называет: так ловится подмена «база» на
    «ARPU». Когда показатель не назван (проза), считается покрытие понятий
    утверждения — как раньше, только с частичным весом за синоним.

    Args:
        claim: Поля утверждения.
        source: Поля источника.

    Returns:
        Доля названных понятий и причина.
    """
    indicator = source.declared.get("показатель", "")
    target = _indicator_concepts(indicator) if indicator else frozenset()
    table = _stem_groups()
    if target:
        credited = sum(_credit(concept, claim, table) for concept in target)
        missing = sorted(concept for concept in target if _credit(concept, claim, table) == 0.0)
        note = f"показатель источника не назван: {', '.join(missing[:3])}" if missing else ""
        return FieldResult(credited / len(target), note)
    if not claim.concepts:
        return FieldResult(1.0, "в утверждении нет понятий предмета")
    credited = sum(_credit(concept, source, table) for concept in claim.concepts)
    missing = sorted(
        concept for concept in claim.concepts if _credit(concept, source, table) == 0.0
    )
    coverage = credited / len(claim.concepts)
    if missing and _short_source(source.raw) and coverage < settings.claim_min_overlap:
        # Короткий источник: непокрытое слово — не «не хватило синонима», а
        # другое утверждение. Здесь порог покрытия остаётся признаком, но
        # только здесь: на длинном абзаце он решением быть перестал.
        return FieldResult(
            0.02,
            f"короткий источник: покрытие {coverage:.0%} "
            f"ниже порога {settings.claim_min_overlap:.0%}",
        )
    note = f"понятия не найдены в источнике: {', '.join(missing[:3])}" if missing else ""
    return FieldResult(coverage, note)


def _subject_field(
    claim: CanonicalText,
    source: CanonicalText,
    identity: Identity | None = None,
) -> FieldResult:
    """Субъект: тот ли оператор (или рынок) назван.

    Сначала — прочтения аббревиатуры (`pf.entities`): «дочерняя компания» и
    «материнская» — разные сущности, и пересечение прочтений даёт **долю**, а
    не вердикт.
    Прочтения не перечислены (нет страницы значений) — работает прежняя
    проверка по названию, и это названо в отчёте.

    Args:
        claim: Поля утверждения.
        source: Поля источника.
        identity: Исход сравнения прочтений, если он считался.

    Returns:
        Долю найденных субъектов. Ноль заменяется малым числом, чтобы
        провал оставался различимым по величине, а не превращался в «нет».
    """
    if identity is not None and identity.kind == KIND_MISMATCH:
        return FieldResult(SHARE_FLOOR, identity.note)
    if identity is not None and identity.kind == KIND_AMBIGUOUS:
        share = max(SHARE_FLOOR, identity.share)
        return FieldResult(share, identity.note)
    if not claim.subjects:
        note = "субъект в утверждении не назван"
        if identity is not None and identity.note:
            note = f"{note}; {identity.note}"
        return FieldResult(1.0, note)
    scope = source.declared.get("оператор") or source.raw
    clean = for_comparison(scope)
    lowered = normalize_homoglyphs(clean.lower().replace("ё", "е"))
    lemmas = _lemmas(clean)
    found = [subject for subject in claim.subjects if _mentions(subject, lowered, lemmas)]
    if len(found) == len(claim.subjects):
        return FieldResult(1.0, "")
    missing = [subject for subject in claim.subjects if subject not in found]
    if not found:
        return FieldResult(SHARE_FLOOR, f"в источнике другой: {missing[0]}")
    return FieldResult(len(found) / len(claim.subjects), f"не найден: {missing[0]}")


def _region_field(claim: CanonicalText, source: CanonicalText) -> FieldResult:
    """Регион: та ли страна или город, о которой говорит утверждение.

    Args:
        claim: Поля утверждения.
        source: Поля источника.

    Returns:
        Доля совпадения регионов. Регион, названный в источнике вместо
        названного в утверждении, — это подмена, а не «не хватило слов».
    """
    if not claim.regions:
        return FieldResult(1.0, "регион в утверждении не назван")
    if claim.regions & source.regions:
        return FieldResult(1.0, "")
    if source.regions:
        return FieldResult(0.05, f"в источнике другой регион: {', '.join(sorted(source.regions))}")
    return FieldResult(0.35, "регион утверждения в источнике не назван")


def _period_field(claim: CanonicalText, source: CanonicalText) -> FieldResult:
    """Период: то ли время, за которое взято значение.

    Args:
        claim: Поля утверждения.
        source: Поля источника.

    Returns:
        Доля совпадения периодов и причина с обеими записями.
    """
    shared = period_share(claim.periods, source.periods)
    if shared is None:
        if claim.month_tags and source.month_tags:
            if claim.month_tags & source.month_tags:
                return FieldResult(1.0, "")
            return FieldResult(0.2, "месяцы не совпали")
        return FieldResult(1.0, "период не назван")
    if shared >= 1.0:
        return FieldResult(1.0, "")
    return FieldResult(
        shared,
        f"в утверждении {described(claim.periods)}, в источнике {described(source.periods)}",
    )


def _aligned(claim: Quantity, source: Quantity) -> tuple[float, float]:
    """Привести величины к одной шкале.

    Если шкалу называет только одна сторона, сравниваются записанные числа:
    «82 млн» против «82» из колонки, где миллионы подразумеваются, — это одно
    и то же число. Обратное правило (сравнивать только базовые величины)
    отклоняло бы каждое утверждение о миллионах против строки без единицы —
    измерено: 21 из 30 честных пар.

    Args:
        claim: Величина утверждения.
        source: Величина источника.

    Returns:
        Пара чисел для сравнения.
    """
    if claim.scale > 1.0 and source.scale > 1.0:
        return claim.value * claim.scale, source.value * source.scale
    return claim.value, source.value


def _quantity_share(
    claim: Quantity,
    source: Quantity,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[float, str]:
    """Степень совпадения одной величины с одной величиной источника.

    Args:
        claim: Величина утверждения.
        source: Величина источника.
        abs_tolerance: Абсолютный допуск округления.
        rel_tolerance: Относительный допуск округления.

    Returns:
        Пара «доля, причина». Внутри допуска доля падает плавно (округление
        не равно подмене), за допуском — затухает показательно.
    """
    if not compatible(claim, source):
        return 0.02, f"единицы не совпадают: {claim.raw} против {source.raw}"
    left, right = _aligned(claim, source)
    tolerance = number_tolerance(right, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance)
    if claim.modifier == "почти":
        tolerance = max(tolerance, APPROXIMATION_SHARE * abs(left))
    if claim.modifier == "более":
        diff = max(0.0, left - right)
    elif claim.modifier == "менее":
        diff = max(0.0, right - left)
    else:
        diff = abs(left - right)
    if tolerance <= 0.0:
        return (1.0, "") if diff == 0.0 else (0.02, f"величина: {claim.raw} против {source.raw}")
    if diff <= tolerance:
        return 1.0 - 0.15 * (diff / tolerance), ""
    return (
        max(1e-3, 0.85 * math.exp(-(diff - tolerance) / tolerance)),
        f"{claim.raw} против {source.raw}",
    )


def _quantity_field(
    claim: CanonicalText,
    source: CanonicalText,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> FieldResult:
    """Величины: сошлись ли числа, единицы и модификаторы.

    Args:
        claim: Поля утверждения.
        source: Поля источника.
        abs_tolerance: Абсолютный допуск округления.
        rel_tolerance: Относительный допуск округления.

    Returns:
        Худшая доля по величинам утверждения: свидетельства не складываются.
    """
    if not claim.quantities:
        return FieldResult(1.0, "в утверждении нет величин")
    if not source.quantities:
        return FieldResult(0.02, "в источнике нет величин: сверить не с чем")
    worst = FieldResult(1.0, "")
    for item in claim.quantities:
        best = FieldResult(-1.0, "")
        for candidate in source.quantities:
            share, note = _quantity_share(
                item,
                candidate,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            if share > best.share:
                best = FieldResult(share, note)
        if best.share < worst.share:
            worst = best
    return worst


def _terms_field(claim: CanonicalText, source: CanonicalText) -> FieldResult:
    """Точные термины: тот ли бренд, аббревиатура, обозначение.

    Args:
        claim: Поля утверждения.
        source: Поля источника.

    Returns:
        Худшая доля по терминам: одно имя не подтверждается другим ни при
        каком совпадении остальных слов.
    """
    if not claim.terms:
        return FieldResult(1.0, "точных терминов нет")
    source_translit = {translit_key(term) for term in source.terms}
    shares: list[tuple[str, float]] = []
    for term in claim.terms:
        # Одно правило с лексической проверкой (`lexical.term_present`):
        # целым словом, без учёта регистра, «ё» и гомоглифов. Подстрока
        # склеивала «МТС» с «МГТС», а чистка перед разбором отключала
        # аббревиатуры вовсе (`D-45`).
        if term_present(term, source.raw):
            shares.append((term, 1.0))
        elif translit_key(term) in source_translit:
            shares.append((term, 0.5))
        else:
            shares.append((term, 0.02))
    worst = min(share for _term, share in shares)
    missing = [term for term, share in shares if share < 1.0]
    note = f"не найден в источнике: {', '.join(missing[:3])}" if missing else ""
    return FieldResult(worst, note)


def _derivation_field(claim: CanonicalText, source: CanonicalText) -> FieldResult:
    """Опора: на что может опираться утверждение.

    Одна строка таблицы — это одна точка. Утверждение «база падала» или
    «средний ARPU по рынку» ею не подтверждается ни при каком совпадении
    полей: это не совпадение значений, а другой вид основания. Раньше такие
    пары отсекались случайно, вместе с покрытием; при переходе на поля их
    надо называть отдельно.

    Args:
        claim: Поля утверждения.
        source: Поля источника.

    Returns:
        Малую долю, если утверждение говорит о динамике или сводке, а
        источник — одна строка; иначе единицу.
    """
    lowered = claim.raw.lower().replace("ё", "е")
    trend = [word for word in TREND_WORDS if word in lowered]
    aggregate = [word for word in AGGREGATE_WORDS if word in lowered]
    if (trend or aggregate) and source.declared:
        kind = "динамику" if trend else "среднее"
        return FieldResult(0.1, f"одна строка не подтверждает {kind}")
    return FieldResult(1.0, "")


def score(
    claim_text: str,
    passage: str,
    *,
    groups: Mapping[str, str] | None = None,
    abs_tolerance: float = 0.1,
    rel_tolerance: float = 5e-4,
) -> SupportScore:
    """Степень поддержки утверждения абзацем источника.

    Args:
        claim_text: Утверждение.
        passage: Абзац или строка источника.
        groups: Отображение синонимов; по умолчанию — словарь лексикона.
        abs_tolerance: Абсолютный допуск округления чисел.
        rel_tolerance: Относительный допуск округления чисел.

    Returns:
        Степень поддержки, вклады полей, покрытие и причину отказа.
    """
    base = lexical_overlap(
        claim_text,
        passage,
        groups=groups,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    claim = canonicalize(claim_text)
    source = canonicalize(passage)
    # Прочтения аббревиатуры: «МТС Беларусь» и «МТС» — разные сущности, и
    # пересечение прочтений даёт долю, а не вердикт (`D-51`).
    identity = entity_identity(
        claim_text,
        passage,
        claim_row=bool(claim.declared),
        source_row=bool(source.declared),
    )
    fields: dict[str, FieldResult] = {
        "показатель": _concepts_field(claim, source),
        "субъект": _subject_field(claim, source, identity),
        "период": _period_field(claim, source),
        "величина": _quantity_field(
            claim,
            source,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        ),
        "опора": _derivation_field(claim, source),
        "регион": _region_field(claim, source),
        "термин": _terms_field(claim, source),
        "отрицание": FieldResult(
            1.0 if base.negation_ok else 0.05,
            "" if base.negation_ok else "в утверждении появилось отрицание",
        ),
    }
    support = 1.0
    support_without_subject = 1.0
    for name, result in fields.items():
        share = min(max(result.share, 1e-6), 1.0)
        support *= share ** FIELD_WEIGHTS[name]
        if name != "субъект":
            # Степень поддержки без поля «субъект»: по ней видно, решает ли
            # неоднозначное прочтение или отказ вызван другим полем.
            support_without_subject *= share ** FIELD_WEIGHTS[name]
    weak = sorted(
        (
            (FIELD_ORDER.index(name), name, result)
            for name, result in fields.items()
            if result.share < 0.5
        ),
        key=lambda item: item[0],
    )
    reason = ""
    if weak:
        _order, name, result = weak[0]
        reason = f"{name}: {result.note}" if result.note else name
    return SupportScore(
        support=support,
        fields=fields,
        coverage=base.coverage,
        oov_tokens=tuple(base.oov_tokens),
        reason=reason,
        entity_share=identity.share,
        entity_kind=identity.kind,
        entity_note=identity.note,
        support_without_subject=support_without_subject,
    )
