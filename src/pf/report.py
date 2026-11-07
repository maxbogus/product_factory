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
class MetricRecord:
    """Расчётная метрика в отчёте.

    У каждой метрики есть статус: посчитанное и проверенное значение
    и отклонённое — это разные вещи, и путать их нельзя. Раньше
    отклонённые расчёты попадали в документ как «посчитанные кодом по
    строкам таблицы».

    Attributes:
        name: Название метрики.
        value: Значение.
        unit: Единица измерения.
        citation: Ссылка на строку источника.
        formula: Как считалось.
        inputs: Входные значения.
        status: Статус проверки: «подтверждено» или «отклонено».
        detail: Пояснение для человека.
    """

    name: str
    value: float
    unit: str = ""
    citation: str = ""
    formula: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    status: str = "подтверждено"
    detail: str = ""

    @property
    def formatted(self) -> str:
        """Значение с единицей измерения."""
        if not self.unit:
            return f"{self.value:.6g}"
        return f"{self.value:.6g} {self.unit}"


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
    metrics: list[MetricRecord] = field(default_factory=list)
    unique_citations: int = 0
    unique_sources: int = 0

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
        lines.append(f"| Подтверждённых записей | {self.confirmed_citations} |")
        lines.append(f"| Подтверждённых оснований (уникальных) | {self.unique_citations} |")
        lines.append(f"| Использовано источников | {self.unique_sources} |")
        lines.append(f"| Оснований отклонено | {self.rejected_citations} |")
        lines.append(f"| Доля подтверждённых | {self.grounding_rate:.0%} |")
        lines.append("")

        if self.metrics:
            lines.append("## Расчёты по данным")
            lines.append("")
            lines.append("| Метрика | Значение | Строка источника | Как считалось | Статус |")
            lines.append("|---|---:|---|---|---|")
            for metric in self.metrics:
                lines.append(
                    f"| {metric.name} | {metric.formatted} | "
                    f"`{metric.citation}` | {metric.formula} | {metric.status} |"
                )
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


def _citation_key(citation: str) -> str:
    """Ключ основания: путь и якорь, без повторов.

    Ссылка может выглядеть как ``проект/файл.csv#row3#p3`` — с
    продублированным якорем. Для счёта уникальных оснований это один
    и тот же факт, а не два.

    Args:
        citation: Ссылка на основание.

    Returns:
        Нормализованный ключ.
    """
    path, _, fragment = citation.partition("#")
    parts = [part for part in fragment.split("#") if part]
    if not parts:
        return path
    return f"{path}#{parts[0]}"


def _collect_metrics(records: list[Any]) -> list[MetricRecord]:
    """Собрать расчёты из журнала со статусом проверки.

    Расчёт попадает в отчёт только вместе со статусом: подтверждённый
    и отклонённый нельзя показывать одинаково. Статус берётся из
    строк отклонения того же шага.

    Args:
        records: Записи журнала.

    Returns:
        Список расчётов.
    """
    metrics: list[MetricRecord] = []
    for record in records:
        detail = record.detail if isinstance(record.detail, dict) else {}
        raw_metrics = detail.get("metrics")
        if not isinstance(raw_metrics, list):
            continue
        rejected_names = {line.split(":", 1)[0].strip() for line in record.rejected_details}
        for raw in raw_metrics:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            status = "подтверждено"
            detail_text = ""
            if name in rejected_names:
                status = "отклонено"
                detail_text = next(
                    (
                        line
                        for line in record.rejected_details
                        if line.split(":", 1)[0].strip() == name
                    ),
                    "",
                )
            source_id = str(raw.get("source_id", "")).strip()
            row = raw.get("row")
            citation = source_id
            if row is not None and "#" not in source_id:
                citation = f"{source_id}#row{row}"
            inputs = raw.get("inputs")
            metrics.append(
                MetricRecord(
                    name=name,
                    value=float(raw.get("value", 0.0)),
                    unit=str(raw.get("unit", "") or ""),
                    citation=citation,
                    formula=str(raw.get("formula", "") or ""),
                    inputs=inputs if isinstance(inputs, dict) else {},
                    status=status,
                    detail=detail_text,
                )
            )
    return metrics


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
    metrics = _collect_metrics(records)

    all_citations: list[str] = []
    for record in records:
        confirmed += len(record.citations)
        rejected += len(record.rejected_details)
        rejected_details.extend(record.rejected_details)
        all_citations.extend(record.citations)
        total_cost += record.cost_units
        cost_by_role[record.role] = cost_by_role.get(record.role, 0.0) + record.cost_units
        used = record.prompt_tokens + record.completion_tokens
        tokens[record.role] = tokens.get(record.role, 0) + used

    # Основания и записи — разные числа. Сборщик и аналитик цитируют
    # одни и те же абзацы: четыре записи — это два разных факта.
    # Называть их четырьмя основаниями неверно, так доля выглядит
    # лучше, чем она есть.
    unique_citations = {_citation_key(citation) for citation in all_citations}
    unique_sources = {citation.split("#", 1)[0] for citation in all_citations}

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
        metrics=metrics,
        unique_citations=len(unique_citations),
        unique_sources=len(unique_sources),
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
