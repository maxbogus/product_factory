"""Ограничения задачи: модель «действие + полярность» и проверка направления.

Зачем модуль. Ограничения собираются на входе (``TaskFrame.constraints``) и
раньше только подставлялись в промпт. В демо-прогоне до ``demo-v9``
ограничение брифа «цену снижать нельзя» потерялось, и выбранным направлением
стало «снизить цену» — никто этого не заметил. Передача в запрос не заменяет
проверку: модель может выбрать линию, которая запрещена постановкой задачи.

Здесь ограничение разбирается в структуру: **действие** («снизить»),
**объект** («цена») и **полярность** (запрет / рамка / требование). Сравнение
идёт по леммам, а не по точной строке: «снижение» и «снизить» — одно
действие. Отрицание учитывается по частям фразы: «не снижать цену» —
соблюдение запрета, а не его нарушение.

**Чего модуль не делает.** Он не меняет статус оснований и не решает, что
делать с нарушением: это делает ``pf.workflow``. Ложное срабатывание опаснее
пропуска, поэтому проверяется действие, а не слово: запрет «не менять цену»
не должен блокировать «сделать тариф гибким».
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pf.corpus import tokenize
from pf.lexical import has_negation, lemma

#: Слово с позицией: нужно, чтобы найти отрицание в той же части фразы.
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
#: Граница части фразы: конец предложения или «, а» / «, но».
_CLAUSE_BREAK = re.compile(r"[.;:!?]|,\s*(?:а|но|зато)\b", re.IGNORECASE)


class ConstraintKind(StrEnum):
    """Вид ограничения.

    Attributes:
        FORBIDDEN: Запрет действия («цену снижать нельзя»).
        LIMITED: Рамка, а не запрет («учитывать бюджет», «сроки ограничены»).
        REQUIREMENT: Обязательное условие («необходимо согласовать»).
        OTHER: Не разобрано в структуру.
    """

    FORBIDDEN = "forbidden"
    LIMITED = "limited"
    REQUIREMENT = "requirement"
    OTHER = "other"


#: Семейства действий: нормализованное действие → леммы, которые его называют.
#: Леммы, а не основы: «снизить» и «снижение» — одно действие, «ценность» и
#: «цена» — разные понятия.
_ACTION_LEMMAS: dict[str, frozenset[str]] = {
    "снизить": frozenset(
        {
            "снизить",
            "снижать",
            "снижение",
            "понизить",
            "понижать",
            "понижение",
            "уменьшить",
            "уменьшать",
            "уменьшение",
            "сократить",
            "сокращать",
            "сокращение",
            "урезать",
            "урезание",
            "сбросить",
        }
    ),
    "повысить": frozenset(
        {
            "повысить",
            "повышать",
            "повышение",
            "поднять",
            "поднимать",
            "увеличить",
            "увеличивать",
            "увеличение",
            "нарастить",
            "наращивать",
            "рост",
            "подорожать",
        }
    ),
    "изменить": frozenset(
        {
            "изменить",
            "изменять",
            "изменение",
            "измениться",
            "менять",
            "поменять",
            "пересмотреть",
            "пересмотр",
            "корректировать",
            "корректировка",
            "переработать",
        }
    ),
    "отказаться": frozenset(
        {
            "отказаться",
            "отказ",
            "закрыть",
            "закрытие",
            "прекратить",
            "прекращение",
            "свернуть",
        }
    ),
}

#: Семейства объектов: нормализованный объект → корни лемм. Длинные корни
#: сравниваются по началу слова («тарифный», «бюджетный»), короткие — точно,
#: иначе «срочный» (спешный) выдал бы себя за «срок».
_TARGET_ROOTS: dict[str, tuple[str, ...]] = {
    "цена": ("цена", "тариф", "стоимость", "прайс"),
    "бюджет": ("бюджет", "затрат", "расход", "финанс"),
    "канал": ("канал",),
    "сегмент": ("сегмент",),
    "срок": ("срок", "дедлайн"),
}

#: Маркеры прямого запрета. «нельзя» не ловится общей проверкой отрицания:
#: это отдельное слово, а не частица «не».
_FORBID_MARKERS: tuple[str, ...] = ("нельзя", "запрещ", "недопустим", "не разрешается")
#: Маркеры рамки: ограничение бюджета или срока — не запрет действия.
_LIMIT_MARKERS: tuple[str, ...] = ("ограничен", "ограничива", "учитыва", "учесть", "в пределах")
#: Маркеры обязательного условия.
_REQUIRE_MARKERS: tuple[str, ...] = ("необходимо", "обязательн", "должен", "требуется")


@dataclass(frozen=True)
class Constraint:
    """Ограничение задачи в структуре.

    Attributes:
        text: Исходная формулировка.
        kind: Вид ограничения.
        action: Нормализованное действие (пусто, если не названо).
        target: Нормализованный объект действия (пусто, если не назван).
    """

    text: str
    kind: ConstraintKind
    action: str = ""
    target: str = ""


@dataclass(frozen=True)
class Violation:
    """Нарушение ограничения выбранным направлением.

    Attributes:
        constraint: Нарушенное ограничение.
        claim: Текст, в котором найдено запрещённое действие.
        reason: Объяснение для человека и отчёта.
    """

    constraint: Constraint
    claim: str
    reason: str


def parse_constraints(items: Iterable[str]) -> list[Constraint]:
    """Разобрать формулировки ограничений в структуру.

    Args:
        items: Строки из ``TaskFrame.constraints``.

    Returns:
        Список ограничений; пустые строки пропускаются.
    """
    result: list[Constraint] = []
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        result.append(_parse_one(text))
    return result


def check_direction(direction: str, constraints: Sequence[Constraint]) -> list[Violation]:
    """Найти запреты, которые нарушает выбранное направление.

    Args:
        direction: Текст выбранного направления.
        constraints: Разобранные ограничения.

    Returns:
        Список нарушений; пустой, если направление ограничений не нарушает.
    """
    text = (direction or "").strip()
    if not text:
        return []
    direction_target = _target_of(_lemmas_of(tokenize(text)))
    violations: list[Violation] = []
    for constraint in constraints:
        if constraint.kind is not ConstraintKind.FORBIDDEN or not constraint.action:
            continue
        # Объект обязателен, если он назван в ограничении: «снизить цену»
        # и «снизить численность» — разные действия.
        if constraint.target and direction_target != constraint.target:
            continue
        if not _asserts_action(text, constraint.action):
            continue
        violations.append(
            Violation(
                constraint=constraint,
                claim=text,
                reason=(
                    f"направление «{text}» выполняет действие, которое "
                    f"ограничение «{constraint.text}» запрещает"
                ),
            )
        )
    return violations


def _parse_one(text: str) -> Constraint:
    """Разобрать одну формулировку.

    Args:
        text: Непустая строка ограничения.

    Returns:
        Ограничение с видом, действием и объектом (если названы).
    """
    normalized = _normalized(text)
    lemmas = _lemmas_of(tokenize(text))
    action = _family_of(lemmas, _ACTION_LEMMAS)
    target = _target_of(lemmas)

    # Запрет — это действие + отрицание. «Бюджет не может быть увеличен» —
    # тоже запрет: действие «увеличить» при отрицании «не».
    forbidden = bool(action) and (
        has_negation(text) or any(marker in normalized for marker in _FORBID_MARKERS)
    )
    if forbidden:
        kind = ConstraintKind.FORBIDDEN
    elif any(marker in normalized for marker in _LIMIT_MARKERS):
        kind = ConstraintKind.LIMITED
    elif any(marker in normalized for marker in _REQUIRE_MARKERS):
        kind = ConstraintKind.REQUIREMENT
    else:
        kind = ConstraintKind.OTHER
    return Constraint(text=text, kind=kind, action=action, target=target)


def _normalized(text: str) -> str:
    """Текст для поиска маркеров: нижний регистр, «ё» сведена к «е».

    Args:
        text: Любой текст.

    Returns:
        Нормализованная строка.
    """
    return text.lower().replace("ё", "е")


def _lemmas_of(words: Iterable[str]) -> list[str]:
    """Леммы слов без пустых.

    Args:
        words: Слова в любом регистре.

    Returns:
        Список непустых лемм.
    """
    return [normal for normal in (lemma(word) for word in words) if normal]


def _family_of(lemmas: Sequence[str], table: dict[str, frozenset[str]]) -> str:
    """Найти семейство, к которому относится первое известное слово.

    Args:
        lemmas: Леммы текста.
        table: Таблица «семейство → допустимые леммы».

    Returns:
        Имя семейства либо пустая строка.
    """
    for family, known in table.items():
        if any(word in known for word in lemmas):
            return family
    return ""


def _target_of(lemmas: Sequence[str]) -> str:
    """Найти объект действия по корням лемм.

    Args:
        lemmas: Леммы текста.

    Returns:
        Имя объекта либо пустая строка.
    """
    for target, roots in _TARGET_ROOTS.items():
        for word in lemmas:
            for root in roots:
                if word == root or (len(root) >= 5 and word.startswith(root)):
                    return target
    return ""


def _asserts_action(text: str, family: str) -> bool:
    """Выполняет ли текст действие семейства (хотя бы раз без отрицания).

    Args:
        text: Проверяемый текст.
        family: Имя семейства действий.

    Returns:
        Найдено неотрицаемое упоминание действия.
    """
    known = _ACTION_LEMMAS[family]
    for match in _WORD.finditer(text):
        if lemma(match.group()) not in known:
            continue
        if not _negated_at(text, match.start()):
            return True
    return False


def _negated_at(text: str, position: int) -> bool:
    """Отрицается ли действие в своей части фразы.

    Отрицание считается только в той же части фразы: «не снижать цену, а
    повысить качество» — запрет соблюдён, а «снизить цену, а не качество» —
    действие выбрано.

    Args:
        text: Проверяемый текст.
        position: Начало слова действия.

    Returns:
        Отрицание или прямой запрет относятся к действию.
    """
    start = 0
    for match in _CLAUSE_BREAK.finditer(text, 0, position):
        start = match.end()
    tail = _CLAUSE_BREAK.search(text, position)
    end = tail.start() if tail else len(text)

    before = text[start:position]
    # «, а повысить» — новая мысль: отрицание к ней не относится.
    before = re.split(r",\s*(?:а|но|зато)\b", before, flags=re.IGNORECASE)[-1]
    if has_negation(before):
        return True
    segment = _normalized(text[start:end])
    return any(marker in segment for marker in _FORBID_MARKERS)
