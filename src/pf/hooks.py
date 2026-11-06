"""Хуки: действия по событиям рабочего процесса.

Хук — правило «когда произошло событие X, сделай Y». Правила читаются из
``hooks/*.yaml`` и выполняются в коде, а не остаются описанием: иначе
обещание в README расходится с поведением.

Сейчас реализованы два хука:

- ``budget.exhausted`` — расход достиг предела: работа останавливается с
  отчётом, а не продолжает жечь токены;
- ``sources.collected`` — подтверждённых оснований мало: план
  исследования расширяется, потому что вывод пока нечем подкрепить.

Хук не может отменить уже сделанное: он либо расширяет вход, либо
останавливает работу. Это важно — иначе «действие по событию» стало бы
правом переписать историю шага.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pf.config import settings


@dataclass(frozen=True, slots=True)
class Hook:
    """Правило, срабатывающее на событие.

    Attributes:
        event: Событие, на которое реагирует хук.
        when: Условие срабатывания (сравнение с порогом).
        threshold: Порог, с которым сравнивается значение.
        action: Что делать при срабатывании.
    """

    event: str
    when: str
    threshold: float
    action: str

    @property
    def operator(self) -> str:
        """Оператор сравнения из условия.

        Returns:
            Символ оператора.
        """
        for symbol in ("<=", ">=", "<", ">", "="):
            if symbol in self.when:
                return symbol
        return "<"

    def matches(self, value: float) -> bool:
        """Сработать ли правило.

        Args:
            value: Текущее значение.

        Returns:
            Условие выполнено.
        """
        if self.operator == "<":
            return value < self.threshold
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == ">":
            return value > self.threshold
        if self.operator == ">=":
            return value >= self.threshold
        return value == self.threshold


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """Что сделал хук.

    Attributes:
        event: Событие.
        fired: Сработал ли хук.
        action: Предпринятое действие.
        detail: Пояснение для человека.
    """

    event: str
    fired: bool
    action: str = ""
    detail: str = ""


def load_hooks(directory: Path | None = None) -> list[Hook]:
    """Прочитать хуки из каталога.

    Args:
        directory: Каталог с файлами хуков. По умолчанию — ``hooks/``.

    Returns:
        Список правил.
    """
    path = directory or settings.hooks_dir
    if not path.is_dir():
        return []
    hooks: list[Hook] = []
    for file in sorted(path.glob("*.yaml")):
        payload = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        # Ключ называется «rules», а не «on»: YAML 1.1 читает «on» как
        # булево True, и список правил не находился бы вовсе.
        entries = payload.get("rules") or payload.get("on") or payload.get(True) or []
        for entry in entries:
            condition = str(entry.get("when", ""))
            threshold = _extract_threshold(condition)
            action = entry.get("action") or entry.get("do") or ""
            if isinstance(action, list):
                action = "; ".join(str(item) for item in action)
            hooks.append(
                Hook(
                    event=str(entry.get("event", "")),
                    when=condition,
                    threshold=threshold,
                    action=str(action),
                )
            )
    return hooks


def _extract_threshold(condition: str) -> float:
    """Достать порог из условия.

    Args:
        condition: Условие вида «подтверждённых оснований < 2».

    Returns:
        Порог; если разобрать не удалось — ноль.
    """
    for token in condition.replace(";", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def fire(event: str, value: float, hooks: list[Hook] | None = None) -> HookOutcome:
    """Проверить, сработает ли хук на событие.

    Args:
        event: Имя события.
        value: Текущее значение для сравнения.
        hooks: Правила. По умолчанию — из каталога.

    Returns:
        Что сделал хук.
    """
    rules = hooks if hooks is not None else load_hooks()
    for hook in rules:
        if hook.event != event:
            continue
        if not hook.matches(value):
            continue
        return HookOutcome(
            event=event,
            fired=True,
            action=hook.action,
            detail=f"{hook.when} (значение {value:g})",
        )
    return HookOutcome(event=event, fired=False)


def expand_plan(state: Any) -> int:
    """Расширить план исследования, когда оснований мало.

    Args:
        state: Состояние сессии.

    Returns:
        Сколько вопросов добавлено.
    """
    plan = state.results.get("planning")
    if not isinstance(plan, dict):
        return 0
    questions = plan.setdefault("questions", [])
    added = [
        "Какие ещё открытые источники могут содержать данные по теме?",
        "Что говорят косвенные признаки, раз прямых данных нет?",
    ]
    new = [q for q in added if q not in questions]
    questions.extend(new)
    return len(new)
