"""Противоречия между фактами: что не сошлось и по какой оси.

Два факта могут быть подтверждены дословно и говорить разное об одном
показателе: «удержание падает» и «удержание не изменилось за последний
квартал». Проверка цитат смотрит на дословность, а не на смысл, поэтому
оба проходят проверку и встают рядом в разборе. Молчать об этом нельзя:
вывод, собранный из них, опирается на противоречие.

Противоречие находит **код**, а не модель. Скил
(``skills/detect-contradictions.md``) описывает процедуру, хук
(``hooks/sources.yaml``, событие ``facts.contradicted``) запускает реакцию,
а ось расхождения вычисляет :func:`find_contradictions`. Ось называется:
читатель видит, **по чему** источники разошлись — направление изменения или
сама величина, — а не голое «противоречие».

Чего правило не делает. Оно не судит, какой из фактов верен: из данных это
не следует, и отчёт так и говорит. Оно не сравнивает факты с источником —
это делает проверка оснований (`pf.ground`). И оно молчит, когда период не
совпадает: «80 % в 2019» и «70 % в 2018» — разные срезы, а не два ответа на
один вопрос.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pf.lexical import numbers_in, numbers_match, polarity, topic_words
from pf.periods import Period, mask_dates, spans

#: Оси расхождения, поддерживаемые правилом. Порядок — от решающей к частной:
#: направление меняет смысл вывода, величина — только число.
AXES: tuple[str, ...] = ("направление", "величина")

#: Подпись оси для человека.
AXIS_TITLES: dict[str, str] = {
    "направление": "направление изменения",
    "величина": "величина",
}


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Противоречие между двумя фактами.

    Attributes:
        axis: Ось, по которой факты расходятся: ``направление`` либо ``величина``.
        topic: Общий предмет — слова, которые называют один показатель.
        left: Текст первого факта.
        right: Текст второго факта.
    """

    axis: str
    topic: str
    left: str
    right: str

    @property
    def axis_title(self) -> str:
        """Как назвать ось человеку.

        Returns:
            Подпись оси: «направление изменения» либо «величина».
        """
        return AXIS_TITLES.get(self.axis, self.axis)

    def gap(self) -> str:
        """Текст пробела: ось, предмет и оба факта.

        Returns:
            Строка для раздела «Противоречия» — отчёт берёт её как есть.
        """
        return (
            f"противоречие по оси «{self.axis_title}»: «{self.left}» и «{self.right}» "
            f"говорят разное об одном показателе («{self.topic}») — "
            "какой из фактов верен, из данных не следует"
        )


@dataclass(frozen=True, slots=True)
class _Statement:
    """Факт, разобранный для сравнения.

    Attributes:
        text: Исходный текст.
        topics: Слова предмета — показатель без служебных.
        direction: Направление изменения либо ``None``.
        numbers: Числа факта без дат.
        periods: Названные периоды как интервалы года.
    """

    text: str
    topics: frozenset[str]
    direction: str | None
    numbers: tuple[float, ...]
    periods: tuple[Period, ...]


def _numbers(text: str) -> tuple[float, ...]:
    """Числа факта без дат: год — период, а не величина.

    Args:
        text: Текст факта.

    Returns:
        Числа, из которых убраны записи дат.
    """
    return tuple(numbers_in(mask_dates(text)))


def _same_slice(left: Period, right: Period) -> bool:
    """Относятся ли два периода к одному срезу, а не к соседним.

    :meth:`pf.periods.Period.overlaps` считает соседние годы пересечением
    по общей границе: 2018–2019 и 2019–2020 «касаются» в точке 2019.
    Для противоречия этого мало: «80 % в 2019» и «70 % в 2018» — разные
    срезы, и общая граница не делает их одним вопросом.

    Args:
        left: Первый период.
        right: Второй период.

    Returns:
        ``True``, если периоды пересекаются по существу.
    """
    if left.start == left.end or right.start == right.end:
        # Точечный период (квартал) лежит внутри другого или совпадает с ним.
        return left.start <= right.end and right.start <= left.end
    return left.start < right.end and right.start < left.end


def _periods_compatible(left: _Statement, right: _Statement) -> bool:
    """Сравнимы ли факты по периоду.

    Величины разных лет не противоречат друг другу. Если период назван
    только с одной стороны, сверять нечем — правило молчит, а не угадывает
    (иначе «80 % в 2019» и «70 %» без периода станут «противоречием»).

    Args:
        left: Первый факт.
        right: Второй факт.

    Returns:
        ``True``, если период не мешает сравнению.
    """
    if not left.periods and not right.periods:
        return True
    if not left.periods or not right.periods:
        return False
    return any(_same_slice(a, b) for a in left.periods for b in right.periods)


def _numbers_differ(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    """Расходятся ли величины двух фактов.

    Сравнение идёт с тем же допуском округления, что в проверке оснований:
    «77,3» и «77,26» — одно значение, а «80» и «70» — разные.

    Args:
        left: Числа первого факта.
        right: Числа второго факта.

    Returns:
        ``True``, если наборы чисел не совпадают.
    """
    left_ok, _ = numbers_match(left, right)
    right_ok, _ = numbers_match(right, left)
    return not (left_ok and right_ok)


def _axis(left: _Statement, right: _Statement) -> str | None:
    """Назвать ось, по которой два факта расходятся.

    Args:
        left: Первый факт.
        right: Второй факт.

    Returns:
        ``направление``, ``величина`` либо ``None``, если расхождения нет.
    """
    if not _periods_compatible(left, right):
        return None
    if left.direction and right.direction and left.direction != right.direction:
        return "направление"
    if left.numbers and right.numbers and _numbers_differ(left.numbers, right.numbers):
        return "величина"
    return None


def _statements(claims: Iterable[Any]) -> list[_Statement]:
    """Разобрать утверждения для сравнения.

    Args:
        claims: Утверждения с атрибутом ``text``.

    Returns:
        Разобранные факты, у которых есть предмет.
    """
    prepared: list[_Statement] = []
    for claim in claims:
        text = str(getattr(claim, "text", "")).strip()
        if not text:
            continue
        topics = topic_words(text)
        if not topics:
            continue
        prepared.append(
            _Statement(
                text=text,
                topics=frozenset(topics),
                direction=polarity(text),
                numbers=_numbers(text),
                periods=tuple(spans(text)),
            )
        )
    return prepared


def find_contradictions(claims: Iterable[Any]) -> list[Contradiction]:
    """Найти факты, которые расходятся об одном показателе.

    Args:
        claims: Проверенные утверждения разбора.

    Returns:
        Противоречия с названной осью в порядке пар.
    """
    statements = _statements(claims)
    found: list[Contradiction] = []
    for index, left in enumerate(statements):
        for right in statements[index + 1 :]:
            common = left.topics & right.topics
            if not common:
                continue
            axis = _axis(left, right)
            if axis is None:
                continue
            found.append(
                Contradiction(
                    axis=axis,
                    topic=", ".join(sorted(common)),
                    left=left.text,
                    right=right.text,
                )
            )
    return found
