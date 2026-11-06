"""Проверки состояния и журнала: работа продолжается после перерыва.

Требование брифа: работу можно остановить и продолжить с сохранённого места,
в том числе открыв новый чат. Здесь проверяется именно это — на диске.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pf.config import settings
from pf.journal import Journal, StepRecord, journal_path
from pf.state import Phase, SessionState, list_sessions, load, save


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать состояние и журнал в временный каталог."""
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")


def _record(ordinal: int, role: str = "intake") -> StepRecord:
    return StepRecord(
        ordinal=ordinal,
        role=role,
        phase="intake",
        summary=f"шаг {ordinal}",
        cost_units=0.001 * ordinal,
    )


def test_journal_is_append_only() -> None:
    """Новые шаги дописываются, старые не переписываются."""
    journal = Journal(settings.journal_dir / "s.jsonl")
    journal.append(_record(1))
    journal.append(_record(2))
    journal.append(_record(3))

    records = journal.read()
    assert [r.ordinal for r in records] == [1, 2, 3]


def test_journal_survives_broken_last_line() -> None:
    """Обрыв на границе записи не ломает чтение журнала."""
    path = settings.journal_dir / "s.jsonl"
    journal = Journal(path)
    journal.append(_record(1))
    journal.append(_record(2))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"ordinal": 3, "role": "brok')

    assert len(journal.read()) == 2


def test_journal_counts_cost_by_role() -> None:
    """Стоимость считается и по итогу, и по ролям."""
    journal = Journal(settings.journal_dir / "s.jsonl")
    journal.append(_record(1, "intake"))
    journal.append(_record(2, "strategy"))

    assert journal.total_cost_units == pytest.approx(0.003)
    assert journal.cost_by_role()["strategy"] == pytest.approx(0.002)


def test_state_roundtrip() -> None:
    """Состояние сохраняется и читается без потерь."""
    state = SessionState(
        session_id="s1",
        phase=Phase.ANALYZING,
        question="почему падает удержание",
        steps_done=3,
        cost_units=0.007,
    )
    state.results["analysis"] = {"facts": 4}
    save(state)

    loaded = load("s1")
    assert loaded is not None
    assert loaded.phase is Phase.ANALYZING
    assert loaded.steps_done == 3
    assert loaded.results["analysis"] == {"facts": 4}


def test_state_is_plain_readable_json() -> None:
    """Состояние — текст на диске: его можно прочитать и поправить руками."""
    state = SessionState(session_id="s2", phase=Phase.OPTIONS)
    save(state)

    path = settings.state_dir / "session-s2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "options"
    assert payload["session_id"] == "s2"


def test_missing_session_returns_none() -> None:
    """Несуществующая сессия — не ошибка, а отсутствие."""
    assert load("нет-такой") is None


def test_phase_order_is_sequential() -> None:
    """Фазы идут в заданном порядке — это и есть рабочий процесс."""
    assert Phase.INTAKE.next is Phase.PLANNING
    assert Phase.PLANNING.next is Phase.ROUTING
    assert Phase.ROUTING.next is Phase.COLLECTING
    assert Phase.COLLECTING.next is Phase.ANALYZING
    assert Phase.ANALYZING.next is Phase.OPTIONS
    assert Phase.OPTIONS.next is Phase.REDTEAM
    assert Phase.REDTEAM.next is Phase.STRATEGY
    assert Phase.STRATEGY.next is Phase.KEEPER
    assert Phase.KEEPER.next is Phase.DONE
    assert Phase.DONE.next is Phase.DONE


def test_every_phase_has_a_role() -> None:
    """У каждой фазы есть роль, которая её выполняет.

    Фаза без роли — это фаза, которая молча ничего не делает: работа
    проходит мимо неё, а человек этого не видит.
    """
    from pf.workflow import Workflow

    state = SessionState(session_id="roles")
    for phase in Phase:
        if phase is Phase.DONE:
            continue
        state.phase = phase
        role, _payload, _context = Workflow()._step_inputs(state, None)
        assert role != "", f"у фазы {phase.value} нет роли"


def test_work_can_resume_from_saved_phase() -> None:
    """Работа продолжается с сохранённой фазы, а не с начала."""
    state = SessionState(session_id="s3", phase=Phase.REDTEAM, steps_done=5)
    save(state)

    # «Новый чат»: состояние читается заново, как будто процесса не было.
    reloaded = load("s3")
    assert reloaded is not None
    assert reloaded.phase is Phase.REDTEAM
    assert reloaded.steps_done == 5

    reloaded.phase = reloaded.phase.next
    reloaded.steps_done += 1
    save(reloaded)

    assert load("s3").phase is Phase.STRATEGY


def test_sessions_are_listed() -> None:
    """Сохранённые сессии находятся."""
    save(SessionState(session_id="alpha"))
    save(SessionState(session_id="beta"))
    found = list_sessions()
    assert "alpha" in found
    assert "beta" in found


def test_session_report_counts_grounding_and_cost(tmp_path: Path) -> None:
    """Отчёт по сессии должен считать качество оснований и расход.

    Журнал хранит по каждому шагу подтверждённые и отклонённые
    основания. Из него можно посчитать долю подтверждённых — без
    этого «качество оснований» остаётся самооценкой.
    """
    from pf.report import build_session_report

    session_id = "report-test"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=0,
            role="collector",
            phase="collecting",
            summary="факты",
            citations=["a#p1", "b#p2"],
            cost_units=0.001,
        )
    )
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="факты",
            citations=["c#p3"],
            rejected_details=["d#p4: цитата не найдена"],
            cost_units=0.002,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    assert report.steps == 2
    assert report.confirmed_citations == 3
    assert report.rejected_citations == 1
    assert report.grounding_rate == pytest.approx(0.75)
    assert report.total_cost_units == pytest.approx(0.003)
    assert report.cost_by_role["collector"] == pytest.approx(0.001)


def test_session_report_handles_missing_session() -> None:
    """Отчёт по несуществующей сессии не должен падать."""
    from pf.report import build_session_report

    report = build_session_report("нет-такой-сессии")
    assert report.steps == 0
    assert report.grounding_rate == 0.0
