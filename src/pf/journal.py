"""Журнал шагов: append-only запись и учёт стоимости.

Журнал отвечает на два вопроса, которые задаёт себе руководитель продукта:
«что агент делал» и «сколько это стоило». Поэтому запись содержит и
содержательную часть (роль, вход, выход, основания), и техническую
(длительность, токены, стоимость).

Файл **не переписывается**: новые шаги дописываются в конец. Это даёт
append-only семантику на уровне файловой системы — даже при сбое
испорченным может оказаться только последняя строка, а не весь журнал.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pf.config import settings


def _now() -> str:
    """Метка времени в ISO-формате."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class StepRecord:
    """Запись о выполненном шаге.

    Attributes:
        ordinal: Порядковый номер шага в сессии.
        role: Роль, которая выполняла шаг.
        phase: Фаза рабочего процесса.
        summary: Кратко о результате шага.
        detail: Полный результат (структурированный объект).
        citations: Основания со ссылками на источники.
        prompt_tokens: Токены в запросе.
        completion_tokens: Токены в ответе.
        cost_units: Стоимость шага в условных единицах.
        duration_ms: Длительность шага.
        created_at: Время шага.
    """

    ordinal: int
    role: str
    phase: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    rejected_details: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_units: float = 0.0
    duration_ms: int = 0
    created_at: str = field(default_factory=_now)

    @property
    def total_tokens(self) -> int:
        """Всего токенов на шаге."""
        return self.prompt_tokens + self.completion_tokens


class Journal:
    """Append-only журнал шагов сессии.

    Attributes:
        path: Путь к файлу журнала.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: StepRecord) -> None:
        """Дописать запись в конец журнала.

        Args:
            record: Запись о шаге.
        """
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(self) -> list[StepRecord]:
        """Прочитать все записи журнала.

        Returns:
            Записи в порядке выполнения. Битая строка в конце файла
            игнорируется: журнал дописывается, поэтому обрыв на границе
            записи — ожидаемое состояние после сбоя.
        """
        if not self.path.exists():
            return []

        records: list[StepRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(StepRecord(**payload))
        return records

    @property
    def total_cost_units(self) -> float:
        """Суммарная стоимость всех шагов."""
        return sum(record.cost_units for record in self.read())

    @property
    def total_tokens(self) -> int:
        """Суммарное число токенов."""
        return sum(record.total_tokens for record in self.read())

    def cost_by_role(self) -> dict[str, float]:
        """Стоимость по ролям — чтобы понять, что дороже всего."""
        totals: dict[str, float] = {}
        for record in self.read():
            totals[record.role] = totals.get(record.role, 0.0) + record.cost_units
        return dict(sorted(totals.items(), key=lambda pair: pair[1], reverse=True))


def journal_path(session_id: str) -> Path:
    """Путь к журналу сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Путь к файлу журнала.
    """
    return settings.journal_dir / f"session-{session_id}.jsonl"
