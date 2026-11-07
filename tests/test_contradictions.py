"""Проверки противоречий между фактами.

Дословность цитаты не гарантирует смысла. В демо-прогоне рядом стояли два
«подтверждённых» факта: «удержание падает» и «удержание не изменилось за
последний квартал». Оба прошли проверку, потому что цитаты нашлись в
строках таблицы — и оба попали в разбор как основание.

Противоречие обязано быть названо: иначе вывод строится на двух
взаимоисключающих утверждениях, и читатель об этом не знает.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.journal import Journal, StepRecord, journal_path
from pf.report import build_session_report
from pf.schemas import Claim, SourceRef
from pf.workflow import Workflow


def _fact(text: str) -> Claim:
    """Факт с основанием.

    Args:
        text: Текст утверждения.

    Returns:
        Факт, пригодный для разбора.
    """
    return Claim(
        kind="факт",
        text=text,
        sources=[SourceRef(source_id="project/data/base.csv#row0", quote="МТС: 87")],
    )


def test_contradicting_facts_become_gaps() -> None:
    """Противоположные утверждения об одном показателе — пробел, а не факты."""
    facts = [
        _fact("Удержание клиентов падает, что требует анализа причин."),
        _fact("Удержание клиентов не изменилось за последний квартал."),
    ]

    gaps = Workflow._find_contradictions(facts)

    assert gaps
    assert "противоречие" in gaps[0]
    assert "удержание" in gaps[0].lower()


def test_consistent_facts_produce_no_gap() -> None:
    """Согласованные факты противоречия не дают."""
    facts = [
        _fact("Удержание клиентов падает второй квартал подряд."),
        _fact("Удержание клиентов снижается быстрее плана."),
    ]

    assert Workflow._find_contradictions(facts) == []


def test_report_shows_contradictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Разбор с противоречием показывает его в отчёте.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    journal = Journal(journal_path("session-contradiction"))
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="фактов: 2",
            detail={
                "facts": [],
                "hypotheses": [],
                "forecasts": [],
                "gaps": [
                    "противоречие: «удержание падает» и «удержание не изменилось» "
                    "говорят разное об одном показателе"
                ],
            },
        )
    )

    report = build_session_report("session-contradiction")
    text = report.as_markdown()

    assert report.contradictions
    assert "противоречие" in text.lower()
