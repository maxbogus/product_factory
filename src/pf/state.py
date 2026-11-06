"""Состояние сессии: сохранение и продолжение после перерыва.

Работа агента можно прервать в любой момент — в том числе закрыв чат. Чтобы
продолжить с того же места, состояние лежит на диске в **текстовом** виде:
его можно посмотреть глазами, исправить руками и перенести на другую машину.

В состоянии хранится не «что думает модель», а факты о ходе работы:
текущая фаза, результаты завершённых шагов, заданные вопросы человеку
и накопленная стоимость. Это делает продолжение предсказуемым: агент
начинает с той же фазы и с теми же данными.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pf.config import settings


class Phase(StrEnum):
    """Фазы рабочего процесса агента."""

    #: Постановка задачи: уточняем цель, получателя, ограничения.
    INTAKE = "intake"
    #: План исследования: какие вопросы могут изменить решение.
    PLANNING = "planning"
    #: Сбор источников.
    COLLECTING = "collecting"
    #: Разбор: факты, гипотезы, пробелы.
    ANALYZING = "analyzing"
    #: Генерация вариантов.
    OPTIONS = "options"
    #: Критическая проверка вариантов.
    REDTEAM = "redteam"
    #: Разработка стратегии.
    STRATEGY = "strategy"
    #: Фиксация решения и обязательств.
    KEEPER = "keeper"
    #: Работа завершена.
    DONE = "done"

    @property
    def title_ru(self) -> str:
        """Человекочитаемое название фазы."""
        return {
            Phase.INTAKE: "Постановка задачи",
            Phase.PLANNING: "План исследования",
            Phase.COLLECTING: "Сбор источников",
            Phase.ANALYZING: "Разбор источников",
            Phase.OPTIONS: "Варианты решения",
            Phase.REDTEAM: "Критическая проверка",
            Phase.STRATEGY: "Разработка стратегии",
            Phase.KEEPER: "Фиксация решения",
            Phase.DONE: "Завершено",
        }[self]

    @property
    def next(self) -> Phase:
        """Следующая фаза по порядку работы."""
        order = list(Phase)
        if self is Phase.DONE:
            return Phase.DONE
        return order[order.index(self) + 1]


@dataclass(slots=True)
class SessionState:
    """Состояние сессии агента.

    Attributes:
        session_id: Идентификатор сессии.
        phase: Текущая фаза.
        question: Исходный вопрос руководителя.
        results: Результаты завершённых шагов по фазам.
        pending_question: Вопрос, ждущий ответа человека.
        cost_units: Накопленная стоимость.
        steps_done: Сколько шагов выполнено.
        updated_at: Когда состояние обновлялось.
    """

    session_id: str
    phase: Phase = Phase.INTAKE
    question: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    pending_question: str = ""
    cost_units: float = 0.0
    steps_done: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_json(self) -> str:
        """Сериализовать состояние в JSON."""
        payload = asdict(self)
        payload["phase"] = self.phase.value
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SessionState:
        """Восстановить состояние из JSON.

        Args:
            raw: Текст JSON.

        Returns:
            Восстановленное состояние.

        Raises:
            ValueError: JSON не разбирается.
        """
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"состояние повреждено: {exc}") from exc
        payload["phase"] = Phase(payload.get("phase", Phase.INTAKE.value))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def state_path(session_id: str) -> Path:
    """Путь к файлу состояния сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Путь к файлу состояния.
    """
    return settings.state_dir / f"session-{session_id}.json"


def save(state: SessionState) -> Path:
    """Сохранить состояние на диск.

    Args:
        state: Состояние сессии.

    Returns:
        Путь, куда записано состояние.
    """
    path = state_path(state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json(), encoding="utf-8")
    return path


def load(session_id: str) -> SessionState | None:
    """Загрузить состояние сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Состояние или ``None``, если сессии ещё не было.
    """
    path = state_path(session_id)
    if not path.exists():
        return None
    return SessionState.from_json(path.read_text(encoding="utf-8"))


def list_sessions() -> list[str]:
    """Идентификаторы сохранённых сессий, свежие первыми.

    Returns:
        Список идентификаторов.
    """
    directory = settings.state_dir
    if not directory.is_dir():
        return []
    sessions = []
    for path in directory.glob("session-*.json"):
        sessions.append(path.stem.removeprefix("session-"))
    return sorted(sessions, key=lambda sid: state_path(sid).stat().st_mtime, reverse=True)
