"""Вердикт гипотезы: поддержана, опровергнута или неизвестна.

Зачем модуль. Гипотезы формулировались и больше не использовались: в ответе
стояло «гипотеза: отток из-за цены», и читатель не мог понять, проверяли её
или забыли. Это худший вид утверждения — выглядит как работа, а проверить
нечего.

Три правила, ради которых всё написано:

1. **Вердикт ставит код.** Модель формулирует гипотезу; сказать «подтверждено»
   — не её право. Вердикт выводится из подтверждённых оснований, а
   подтверждает основания проверка дословности (`pf.ground`).
2. **`unknown` — результат, а не провал.** Гипотеза, о которой в данных ничего
   нет, остаётся неизвестной и называется в отчёте. Молчание модели о
   вердикте не превращается в подтверждение.
3. **Опровержение требует противоположного направления.** «Отток из-за цен»
   не утверждает, куда идёт отток: и рост, и падение показателя совместимы с
   гипотезой. Опровергает только утверждение о противоположном направлении
   того же предмета.

Правило направления и предмета одно на репозиторий — ``pf.lexical.polarity``
и ``pf.lexical.topic_words``: противоречия в фактах (`Workflow`) и вердикт
гипотез обязаны считать направление одинаково, иначе «падает» в одном месте
и «снижается» в другом перестают быть одним и тем же.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pf.lexical import polarity, topic_words


class Verdict(StrEnum):
    """Вердикт гипотезы."""

    #: Есть подтверждённое основание о том же предмете.
    SUPPORTED = "supported"
    #: Есть подтверждённое основание о противоположном направлении.
    REFUTED = "refuted"
    #: Проверять нечем: подтверждённых оснований о предмете нет.
    UNKNOWN = "unknown"

    @property
    def title_ru(self) -> str:
        """Человекочитаемое название вердикта."""
        return {
            Verdict.SUPPORTED: "поддержано",
            Verdict.REFUTED: "опровергнуто",
            Verdict.UNKNOWN: "требует проверки",
        }[self]


@dataclass(frozen=True, slots=True)
class HypothesisVerdict:
    """Вердикт гипотезы и то, на чём он держится.

    Attributes:
        verdict: Поддержана, опровергнута или неизвестна.
        basis: Подтверждённое утверждение, из которого следует вердикт.
            Пусто у ``unknown`` — и это честнее, чем выдуманное основание.
    """

    verdict: Verdict = Verdict.UNKNOWN
    basis: str = ""


def verdict_of(hypothesis: str, grounds: Iterable[str]) -> HypothesisVerdict:
    """Вывести вердикт гипотезы из подтверждённых утверждений.

    Основания перебираются по порядку, и первое совпадение по предмету решает:
    набор подтверждённых утверждений — это данные, а не голосование, поэтому
    результат не зависит от повторных вызовов.

    Args:
        hypothesis: Текст гипотезы.
        grounds: Тексты подтверждённых утверждений (дословность уже проверена).

    Returns:
        Вердикт и основание, на котором он держится.
    """
    text = hypothesis.strip()
    if not text:
        return HypothesisVerdict()

    subjects = topic_words(text)
    direction = polarity(text)
    for ground in grounds:
        candidate = str(ground).strip()
        if not candidate or not (topic_words(candidate) & subjects):
            continue
        candidate_direction = polarity(candidate)
        if direction is None:
            # Гипотеза не называет направление: совпадение предмета — уже
            # поддержка, а опровергать нечего.
            return HypothesisVerdict(verdict=Verdict.SUPPORTED, basis=candidate)
        if candidate_direction is None:
            # Основание о том же предмете, но о направлении молчит. Это не
            # противоречие: сравнивать не с чем.
            continue
        if candidate_direction == direction:
            return HypothesisVerdict(verdict=Verdict.SUPPORTED, basis=candidate)
        return HypothesisVerdict(verdict=Verdict.REFUTED, basis=candidate)
    return HypothesisVerdict()
