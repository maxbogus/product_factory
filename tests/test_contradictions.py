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

from pf.config import settings
from pf.contradictions import find_contradictions
from pf.hooks import fire
from pf.journal import Journal, StepRecord, journal_path
from pf.knowledge import ROLE_SKILLS
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


def test_contradiction_axis_is_named() -> None:
    """Ось расхождения называет код, а не формулировка модели.

    «Говорят разное об одном показателе» не отвечало на вопрос, **по чему**
    источники разошлись. Ось нужна, чтобы читатель видел предмет спора —
    направление изменения или сама величина, — а не голое «противоречие».
    """
    facts = [
        _fact("Удержание клиентов падает второй квартал подряд."),
        _fact("Удержание клиентов не изменилось за последний квартал."),
    ]

    found = find_contradictions(facts)

    assert found, "противоположные направления — противоречие"
    assert found[0].axis == "направление"
    assert "удержание" in found[0].topic.lower()
    assert "направление" in found[0].gap()


def test_contradiction_axes_cover_value() -> None:
    """Разные числа об одном показателе — ось «величина»."""
    facts = [
        _fact("Удержание клиентов составило 80 процентов."),
        _fact("Удержание клиентов составило 70 процентов."),
    ]

    found = find_contradictions(facts)

    assert found, "разные значения одного показателя — противоречие"
    assert found[0].axis == "величина"
    assert "величина" in found[0].gap()


def test_contradiction_axes_respect_period() -> None:
    """Одна величина в разные годы — не противоречие.

    «80 % в 2019» и «70 % в 2018» — разные срезы, а не два ответа на один
    вопрос. Без проверки периода правило объявило бы противоречием любое
    изменение показателя во времени.
    """
    facts = [
        _fact("Удержание клиентов составило 80 процентов за 2019 год."),
        _fact("Удержание клиентов составило 70 процентов за 2018 год."),
    ]

    assert find_contradictions(facts) == []


def test_contradiction_axis_skill_is_wired() -> None:
    """Скил поиска противоречий подключён к роли, а не лежит файлом.

    Скил описывает процедуру для человека и модели; подключение проверяется
    по реестру ролей, а не по наличию файла: файл может лежать и не
    подставляться.
    """
    assert "detect-contradictions" in ROLE_SKILLS.get("analyst", ())
    assert (settings.skills_dir / "detect-contradictions.md").is_file()


def test_contradiction_axis_hook_fires() -> None:
    """Хук «противоречия найдены» срабатывает и называет ось в действии."""
    fired = fire("facts.contradicted", 1.0)
    assert fired.fired
    assert "ось" in fired.action.lower()
    assert not fire("facts.contradicted", 0.0).fired
