"""Субагенты: помощники для узких задач.

Субагент — не ещё одна роль в основном потоке, а помощник, который
подключается по необходимости: основная цепочка ролей не нужна, нужна
одна проверка. Здесь — ``fact-check``: проверка оснований по конкретному
источнику.

Он полезен тем, что даёт **второе мнение о том же материале**. Основной
проверяющий закрывает факты по цитате; субагент идёт от факта к
источнику и отвечает на другой вопрос: а относится ли этот источник к
задаче? Цитата может быть настоящей, а статья — посторонней.

Результат субагента пишется в журнал отдельной записью: человек видит,
что основание проверено дважды и кем.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Finding:
    """Находка субагента по одному основанию.

    Attributes:
        source_id: Идентификатор источника.
        paragraph: Номер абзаца, если указан.
        status: Статус проверки: ``confirmed``, ``needs_review`` или ``no_data``.
        detail: Пояснение для человека.
    """

    source_id: str
    paragraph: int | None = None
    status: str = "no_data"
    detail: str = ""


@dataclass(slots=True)
class FactCheckReport:
    """Отчёт субагента по набору оснований.

    Attributes:
        findings: Находки по каждому основанию.
        confirmed: Сколько оснований подтверждено.
        needs_review: Сколько требует проверки.
        no_data: Сколько не подтверждено вовсе.
    """

    findings: list[Finding] = field(default_factory=list)
    confirmed: int = 0
    needs_review: int = 0
    no_data: int = 0

    @property
    def total(self) -> int:
        """Всего проверено оснований."""
        return len(self.findings)

    def summary(self) -> str:
        """Краткое описание отчёта.

        Returns:
            Строка для журнала.
        """
        return (
            f"субагент fact-check: проверено {self.total}, "
            f"подтверждено {self.confirmed}, требует проверки {self.needs_review}, "
            f"нет данных {self.no_data}"
        )


def check_facts(
    claims: list[Any],
    verify: Any,
) -> FactCheckReport:
    """Проверить основания по источникам.

    Функция проверки передаётся снаружи: субагент не знает, откуда взялся
    источник — локальный корпус или веб. Это позволяет проверять оба
    вида одинаково и не тянуть в модуль ни корпус, ни сеть.

    Args:
        claims: Утверждения с источниками (объекты схемы ``Claim``).
        verify: Функция проверки: ``(text, source_id, paragraph, quote)``
            → объект со статусом и пояснением.

    Returns:
        Отчёт по основаниям.
    """
    report = FactCheckReport()
    for claim in claims:
        claim_text = getattr(claim, "text", "")
        sources = getattr(claim, "sources", None) or []
        for source in sources:
            source_id = getattr(source, "source_id", "")
            paragraph = getattr(source, "paragraph", None)
            quote = getattr(source, "quote", "")
            support = verify(claim_text, source_id, paragraph, quote)
            status = str(getattr(support, "status", "no_data"))
            detail = str(getattr(support, "detail", ""))
            report.findings.append(
                Finding(
                    source_id=source_id,
                    paragraph=paragraph,
                    status=status,
                    detail=detail,
                )
            )
            if status == "confirmed":
                report.confirmed += 1
            elif status == "needs_review":
                report.needs_review += 1
            else:
                report.no_data += 1
    return report
