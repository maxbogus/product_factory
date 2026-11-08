"""Проверки вердикта гипотез.

Бриф требует «провести настоящее исследование и проверить гипотезы». Гипотезы
формулировались и всё: читатель видел «гипотеза: отток из-за цены» и не мог
понять, проверяли её или забыли.

Три правила, ради которых всё написано:

1. **Вердикт ставит код, а не модель.** Модель формулирует гипотезу; сказать
   «подтверждено» — не её право.
2. **`unknown` — результат, а не провал.** Гипотеза без проверки не считается
   подтверждённой и не прячется.
3. **Опровержение требует противоположного направления.** Молчание о
   направлении — не противоречие.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.journal import Journal, StepRecord, journal_path


def test_verdict_defaults_to_unknown() -> None:
    """Гипотеза без проверки — `unknown`, а не «подтверждено»."""
    from pf.hypotheses import Verdict
    from pf.schemas import Claim

    claim = Claim(kind="гипотеза", text="Отток растёт из-за роста цен на тарифы")
    assert claim.verdict is Verdict.UNKNOWN
    assert claim.verdict_basis == ""


def test_verdict_from_model_is_ignored() -> None:
    """Модель не может объявить гипотезу подтверждённой.

    Маркер вердикта — это утверждение модели о своей работе, а не свойство
    оснований: в сохранённых прогонах такие утверждения уже расходились с
    фактами (см. `docs/decisions.md`, D-23).
    """
    from pf.hypotheses import Verdict
    from pf.schemas import Claim

    claim = Claim(
        kind="гипотеза",
        text="Отток растёт из-за роста цен на тарифы",
        verdict="supported",
        verdict_basis="модель так решила",
    )
    assert claim.verdict is Verdict.UNKNOWN
    assert claim.verdict_basis == ""


def test_verdict_supported_needs_grounded_basis() -> None:
    """`supported` — только когда предмет гипотезы подтверждён дословно."""
    from pf.hypotheses import Verdict, verdict_of

    hypothesis = "Отток абонентов растёт из-за цен на тарифы"
    supported = verdict_of(hypothesis, ["Отток абонентов растёт третий квартал подряд"])
    assert supported.verdict is Verdict.SUPPORTED
    assert "Отток абонентов растёт" in supported.basis

    unknown = verdict_of(hypothesis, ["Выручка оператора выросла на 8 %"])
    assert unknown.verdict is Verdict.UNKNOWN
    assert unknown.basis == ""


def test_verdict_refuted_on_opposite_polarity() -> None:
    """Обратное направление по тому же предмету — `refuted`."""
    from pf.hypotheses import Verdict, verdict_of

    result = verdict_of(
        "Удержание абонентов падает",
        ["Удержание абонентов выросло до 87 % за квартал"],
    )
    assert result.verdict is Verdict.REFUTED
    assert result.basis


def test_verdict_without_direction_is_not_refuted() -> None:
    """Гипотеза без направления не опровергается ростом показателя.

    «Отток из-за цен» не утверждает, куда отток идёт: рост или падение
    показателя её не опровергает, а подтверждает предмет.
    """
    from pf.hypotheses import Verdict, verdict_of

    result = verdict_of(
        "Отток абонентов из-за цен на тарифы",
        ["Отток абонентов снизился на 2 п.п. за квартал"],
    )
    assert result.verdict is Verdict.SUPPORTED


def test_verdict_ignores_unconfirmed_grounds() -> None:
    """Неподтверждённое основание вердикта не даёт.

    `_apply_hypothesis_verdicts` берёт только подтверждённые вердикты:
    гипотеза, опирающаяся на «требует проверки», остаётся `unknown`.
    """
    from pf.ground import SupportStatus
    from pf.workflow import ClaimVerdict, GroundingResult, Workflow

    grounding = GroundingResult(
        verdicts=[
            ClaimVerdict(
                claim="Отток абонентов растёт",
                source_id="project/data/x.csv",
                status=SupportStatus.NEEDS_REVIEW,
                detail="иностранный источник",
            )
        ]
    )
    assert Workflow._confirmed_grounds(grounding) == []


def test_verdict_reaches_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отчёт называет: поддержано столько, опровергнуто столько, неизвестно."""
    from pf.report import build_session_report

    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    session_id = "session-verdict"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="фактов: 1, гипотез: 3, пробелов: 0",
            detail={
                "facts": [{"kind": "факт", "text": "Отток абонентов растёт", "sources": []}],
                "hypotheses": [
                    {
                        "kind": "гипотеза",
                        "text": "Отток абонентов растёт",
                        "sources": [],
                        "verdict": "supported",
                        "verdict_basis": "Отток абонентов растёт",
                    },
                    {
                        "kind": "гипотеза",
                        "text": "Удержание абонентов падает",
                        "sources": [],
                        "verdict": "refuted",
                        "verdict_basis": "Удержание абонентов выросло до 87 %",
                    },
                    {
                        "kind": "гипотеза",
                        "text": "Цены на тарифы вырастут",
                        "sources": [],
                        "verdict": "unknown",
                        "verdict_basis": "",
                    },
                ],
                "gaps": [],
            },
        )
    )

    report = build_session_report(session_id)
    assert report.hypotheses == {"supported": 1, "refuted": 1, "unknown": 1}
    markdown = report.as_markdown()
    assert "поддержано 1" in markdown
    assert "опровергнуто 1" in markdown
    assert "требует проверки 1" in markdown
