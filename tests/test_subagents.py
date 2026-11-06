"""Проверки субагента fact-check.

Субагент отвечает на другой вопрос, чем основной проверяющий: не
«найдена ли цитата», а «относится ли источник к задаче». Цитата может
быть настоящей, а статья — посторонней.
"""

from __future__ import annotations

from typing import Any

from pf.ground import Support, SupportStatus
from pf.subagents import check_facts


class FakeClaim:
    """Утверждение со источником."""

    def __init__(self, text: str, source_id: str, paragraph: int, quote: str) -> None:
        self.text = text
        self.sources = [FakeSource(source_id, paragraph, quote)]


class FakeSource:
    """Источник утверждения."""

    def __init__(self, source_id: str, paragraph: int, quote: str) -> None:
        self.source_id = source_id
        self.paragraph = paragraph
        self.quote = quote


def _support(status: SupportStatus, detail: str = "") -> Support:
    return Support(quote="цитата", source_id="s", status=status, detail=detail)


def test_confirmed_claim_is_counted() -> None:
    """Подтверждённое основание учтено."""
    claims = [FakeClaim("текст", "telecom/Мегафон", 10, "цитата")]
    report = check_facts(claims, lambda *_: _support(SupportStatus.CONFIRMED))

    assert report.total == 1
    assert report.confirmed == 1
    assert report.no_data == 0


def test_rejected_claim_is_counted() -> None:
    """Отклонённое основание учтено и объяснено."""
    claims = [FakeClaim("текст", "physics/Механика", 10, "выдумка")]
    report = check_facts(claims, lambda *_: _support(SupportStatus.NO_DATA, "цитата не найдена"))

    assert report.no_data == 1
    assert "не найдена" in report.findings[0].detail


def test_report_covers_every_source() -> None:
    """Каждый источник каждого утверждения проверен."""

    class TwoSources(FakeClaim):
        def __init__(self) -> None:
            self.text = "текст"
            self.sources = [
                FakeSource("a", 1, "цитата"),
                FakeSource("b", 2, "цитата"),
            ]

    report = check_facts([TwoSources()], lambda *_: _support(SupportStatus.CONFIRMED))
    assert report.total == 2
    assert report.confirmed == 2


def test_summary_is_readable() -> None:
    """Сводка читается человеком."""
    claims = [FakeClaim("текст", "s", 1, "цитата")]
    report = check_facts(claims, lambda *_: _support(SupportStatus.CONFIRMED))
    summary = report.summary()
    assert "fact-check" in summary
    assert "проверено 1" in summary


def test_empty_claims_produce_empty_report() -> None:
    """Нет утверждений — нет отчёта."""

    def verify(*args: Any) -> Any:
        raise AssertionError("проверка не должна вызываться")

    report = check_facts([], verify)
    assert report.total == 0
