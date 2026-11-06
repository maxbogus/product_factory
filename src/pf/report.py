"""Отчёт по прогону: числа вместо пересказа.

Демонстрация должна показывать не «агент работал», а измеримые
результаты: сколько оснований подтверждено, сколько отклонено, сколько
стоит прогон и что именно замер **не** показывает.

Поэтому отчёт строится по журналу шагов (append-only, значит, цифры
воспроизводимы) и содержит:

- числа по основаниям: подтверждено, отклонено, доля подтверждённых;
- расход: итог и по ролям;
- раздел «чего этот замер не показывает» — обязательный. Метрика без
  оговорок о границах выглядит точнее, чем она есть.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pf.journal import Journal, journal_path
from pf.state import load


@dataclass(frozen=True, slots=True)
class SessionReport:
    """Отчёт по одной сессии.

    Attributes:
        session_id: Идентификатор сессии.
        steps: Сколько шагов выполнено.
        confirmed_citations: Сколько оснований подтверждено.
        rejected_citations: Сколько оснований отклонено.
        total_cost_units: Расход в условных единицах.
        cost_by_role: Расход по ролям.
        tokens: Токены по ролям.
        rejected_details: Что и почему отклонено.
        question: Исходный вопрос.
        phase: Фаза, на которой остановилась сессии.
    """

    session_id: str
    steps: int = 0
    confirmed_citations: int = 0
    rejected_citations: int = 0
    total_cost_units: float = 0.0
    cost_by_role: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    rejected_details: list[str] = field(default_factory=list)
    question: str = ""
    phase: str = ""

    @property
    def total_citations(self) -> int:
        """Всего оснований: подтверждённых и отклонённых."""
        return self.confirmed_citations + self.rejected_citations

    @property
    def grounding_rate(self) -> float:
        """Доля подтверждённых оснований.

        Returns:
            Доля от 0 до 1; ноль, если оснований не было.
        """
        if not self.total_citations:
            return 0.0
        return self.confirmed_citations / self.total_citations

    @property
    def limitations(self) -> list[str]:
        """Чего этот замер не показывает.

        Returns:
            Список оговорок.
        """
        return [
            "Подтверждение основания — дословное совпадение цитаты с "
            "источником. Оно не гарантирует, что цитата поддерживает "
            "утждение по смыслу: честная цитата из настоящей статьи "
            "проходит проверку.",
            "Доля подтверждённых оснований не измеряет полноту: "
            "несколько подтверждённых оснований не означают, что "
            "пропущенных не было.",
            "Расход в условных единицах — это объём токенов. Деньги "
            "зависят от тарифа модели, а время — от её скорости.",
            "Замер сделан на одной модели и одной задаче: разброс между "
            "прогонами локальных моделей велик, поэтому сравнения "
            "считаются по нескольким прогонам.",
        ]

    def as_markdown(self) -> str:
        """Оформить отчёт в markdown.

        Returns:
            Текст отчёта.
        """
        lines: list[str] = []
        lines.append(f"# Отчёт по прогону: {self.session_id}")
        lines.append("")
        if self.question:
            lines.append(f"**Задача:** {self.question}")
            lines.append("")

        lines.append("## Основы")
        lines.append("")
        lines.append("| Показатель | Значение |")
        lines.append("|---|---:|")
        lines.append(f"| Шагов выполнено | {self.steps} |")
        lines.append(f"| Оснований подтверждено | {self.confirmed_citations} |")
        lines.append(f"| Оснований отклонено | {self.rejected_citations} |")
        lines.append(f"| Доля подтверждённых | {self.grounding_rate:.0%} |")
        lines.append("")

        lines.append("## Расход")
        lines.append("")
        lines.append("| Роль | Расход, усл. ед. | Токенов |")
        lines.append("|---|---:|---:|")
        for role, cost in sorted(self.cost_by_role.items(), key=lambda item: -item[1]):
            lines.append(f"| {role} | {cost:.6f} | {self.tokens.get(role, 0)} |")
        lines.append(f"| **Итого** | **{self.total_cost_units:.6f}** | |")
        lines.append("")

        if self.rejected_details:
            lines.append("## Отклонённые основания")
            lines.append("")
            for detail in self.rejected_details:
                lines.append(f"- {detail}")
            lines.append("")

        lines.append("## Чего этот замер не показывает")
        lines.append("")
        for item in self.limitations:
            lines.append(f"1. {item}")
        lines.append("")
        return "\n".join(lines)


def build_session_report(session_id: str) -> SessionReport:
    """Собрать отчёт по сессии из журнала шагов.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Отчёт. Для несуществующей сессии — пустой, без ошибки.
    """
    records = Journal(journal_path(session_id)).read()
    cost_by_role: dict[str, float] = {}
    tokens: dict[str, int] = {}
    confirmed = 0
    rejected = 0
    rejected_details: list[str] = []
    total_cost = 0.0

    for record in records:
        confirmed += len(record.citations)
        rejected += len(record.rejected_details)
        rejected_details.extend(record.rejected_details)
        total_cost += record.cost_units
        cost_by_role[record.role] = cost_by_role.get(record.role, 0.0) + record.cost_units
        used = record.prompt_tokens + record.completion_tokens
        tokens[record.role] = tokens.get(record.role, 0) + used

    state = load(session_id)
    return SessionReport(
        session_id=session_id,
        steps=len(records),
        confirmed_citations=confirmed,
        rejected_citations=rejected,
        total_cost_units=total_cost,
        cost_by_role=cost_by_role,
        tokens=tokens,
        rejected_details=rejected_details,
        question=state.question if state else "",
        phase=state.phase.value if state else "",
    )


def report_as_dict(report: SessionReport) -> dict[str, Any]:
    """Отчёт в виде словаря — для машинного чтения.

    Args:
        report: Отчёт.

    Returns:
        Словарь с показателями.
    """
    return {
        "session_id": report.session_id,
        "steps": report.steps,
        "confirmed_citations": report.confirmed_citations,
        "rejected_citations": report.rejected_citations,
        "grounding_rate": report.grounding_rate,
        "total_cost_units": report.total_cost_units,
        "cost_by_role": report.cost_by_role,
        "tokens": report.tokens,
        "question": report.question,
        "phase": report.phase,
        "limitations": report.limitations,
    }
