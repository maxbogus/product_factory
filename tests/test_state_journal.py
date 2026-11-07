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


def test_session_report_lists_metrics_with_formula(tmp_path: Path) -> None:
    """Расчёты попадают в отчёт вместе с формулой и строкой.

    Отчёт должен показывать, что посчитано и по чему. Без формулы
    таблица метрик превращается в набор чисел, которым нельзя
    поверить: их нечем пересчитать.
    """
    from pf.report import build_session_report

    session_id = "report-metrics"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            citations=["project/data/x.csv#row1"],
            detail={
                "metrics": [
                    {
                        "name": "Абонентская база МТС",
                        "value": 82.0,
                        "unit": "млн",
                        "source_id": "project/data/x.csv#row1",
                        "row": 1,
                        "formula": "значение из колонки «значение», строка 2",
                        "inputs": {"значение": "82"},
                    }
                ]
            },
            cost_units=0.001,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    assert report.metrics, "расчёты должны попасть в отчёт"
    metric = report.metrics[0]
    assert metric.name == "Абонентская база МТС"
    assert metric.formula == "значение из колонки «значение», строка 2"
    assert metric.citation == "project/data/x.csv#row1"

    text = report.as_markdown()
    assert "Расчёты по данным" in text
    assert "значение из колонки" in text
    assert "Как считалось" in text


def test_session_report_counts_unique_grounds(tmp_path: Path) -> None:
    """Отчёт различает записи и уникальные основания.

    Сборщик и аналитик цитируют одни и те же абзацы: семь подтверждённых
    записей — это три разных факта, посчитанных дважды. Называть это
    «семью подтверждёнными основаниями» неверно: так доля выглядит
    лучше, чем она есть.
    """
    from pf.report import build_session_report

    session_id = "report-unique"
    journal = Journal(journal_path(session_id))
    for ordinal, role in enumerate(("collector", "analyst")):
        journal.append(
            StepRecord(
                ordinal=ordinal,
                role=role,
                phase="collecting" if role == "collector" else "analyzing",
                summary="факты",
                citations=["a#p1", "b#p2"],
                cost_units=0.001,
            )
        )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    assert report.confirmed_citations == 4
    assert report.unique_citations == 2
    assert report.unique_sources == 2

    text = report.as_markdown()
    assert "Подтверждённых записей" in text
    assert "Подтверждённых оснований" in text


def test_session_report_does_not_show_rejected_metric_as_confirmed(tmp_path: Path) -> None:
    """Отклонённый расчёт не подаётся как посчитанный.

    В демо-документе расчёты, отклонённые проверкой, были поданы как
    «посчитанные кодом по строкам таблицы». Отчёт так делать не должен:
    у каждого расчёта есть статус.
    """
    from pf.report import build_session_report

    session_id = "report-rejected-metric"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            rejected_details=["Абонентская база МТС: значение не совпадает со строкой"],
            detail={
                "metrics": [
                    {
                        "name": "Абонентская база МТС",
                        "value": 82.0,
                        "unit": "млн",
                        "source_id": "project/data/x.csv#row0",
                        "row": 0,
                        "formula": "значение из колонки «значение», строка 1",
                        "inputs": {},
                    }
                ]
            },
            cost_units=0.001,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    assert report.metrics
    assert report.metrics[0].status != "подтверждено"
    assert "отклонено" in report.metrics[0].status


def test_session_report_merges_same_metric_from_two_roles() -> None:
    """Один и тот же расчёт из двух ролей — одна строка в отчёте.

    Коллектор и аналитик считают одни и те же метрики по тем же строкам.
    В отчёте это один расчёт, а не два: иначе таблица расчётов выглядит
    вдвое убедительнее, чем есть.
    """
    from pf.report import build_session_report

    session_id = "report-metric-dedup"
    journal = Journal(journal_path(session_id))
    metric = {
        "name": "Абонентская база МТС",
        "value": 82.0,
        "unit": "млн",
        "source_id": "project/data/x.csv#row0",
        "row": 0,
        "formula": "значение из колонки «значение», строка 1",
        "inputs": {"значение": "82"},
    }
    for ordinal, role in enumerate(("collector", "analyst")):
        journal.append(
            StepRecord(
                ordinal=ordinal,
                role=role,
                phase="collecting" if role == "collector" else "analyzing",
                summary="расчёты",
                detail={"metrics": [dict(metric)]},
                cost_units=0.001,
            )
        )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    assert len(report.metrics) == 1, "одинаковый расчёт посчитан за два"


def test_session_report_marks_metric_without_formula() -> None:
    """Расчёт без формулы помечен: пересчитать его нечем.

    Журналы прошлых прогонов содержат метрики без формулы. Показывать их
    так же, как проверенные, нельзя: пустая колонка «Как считалось»
    означает «проверить нечем», и это должно быть видно в отчёте.
    """
    from pf.report import build_session_report

    session_id = "report-metric-no-formula"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="расчёты",
            detail={
                "metrics": [
                    {
                        "name": "Абонентская база МТС",
                        "value": 82.0,
                        "unit": "млн",
                        "source_id": "project/data/x.csv#row0",
                        "row": 0,
                    }
                ]
            },
            cost_units=0.001,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    text = report.as_markdown()
    assert "формула не указана" in text


def test_session_report_shows_quality_control() -> None:
    """Отчёт показывает, чем проверялись основания и что решил роутер.

    Без этого «контроль качества» существует только в state: журнал
    хранит подтверждённое и отклонённое, но не решения и не второе
    мнение субагента. Артефакт без них выглядит лучше, чем прогон.
    """
    from pf.report import build_session_report

    session_id = "report-quality"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            citations=["project/data/x.csv#row0"],
            rejected_details=["Показатели стабильны: цитата не найдена в строке таблицы"],
            detail={
                "metrics": [
                    {
                        "name": "Абонентская база МТС",
                        "value": 82.0,
                        "unit": "млн",
                        "source_id": "project/data/x.csv#row0",
                        "row": 0,
                        "formula": "значение из колонки «значение», строка 1",
                    }
                ],
                "routing": {
                    "role": "план",
                    "target": "analyzing",
                    "accepted": False,
                    "note": "маршрут «план» неизвестен — идём по порядку",
                },
                "fact_check": {
                    "summary": "субагент fact-check: проверено 2, подтверждено 1",
                    "confirmed": 1,
                    "needs_review": 0,
                    "no_data": 1,
                },
            },
            cost_units=0.001,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    assert report.metrics_with_formula == 1
    assert report.rejection_kinds
    assert report.routing_notes

    text = report.as_markdown()
    assert "Контроль качества" in text
    assert "Расчётов с формулой" in text
    assert "Повторная проверка" in text
    assert "Решения роутера" in text


def test_step_record_keeps_routing_decision() -> None:
    """Решение роутера попадает в журнал следующего шага.

    Роутер решает, кто работает следующим, а код может это решение
    исправить. Человек должен видеть исправление в отчёте, а не только
    в состоянии сессии: иначе требование «исправлять ошибочный выбор»
    недоказуемо.
    """
    from pf.schemas import NextStep
    from pf.workflow import StepResult, Workflow

    state = SessionState(session_id="journal-routing", phase=Phase.COLLECTING)
    state.results["routing_decision"] = {
        "role": "план",
        "target": "collecting",
        "accepted": False,
        "note": "маршрут «план» неизвестен — идём по порядку: Сбор источников",
    }
    workflow = Workflow(client=object(), config=None)  # type: ignore[arg-type]
    workflow._record(
        state,
        StepResult(
            role="collector",
            phase=Phase.COLLECTING,
            data=NextStep(role="collector", reason="порядок"),
        ),
    )

    records = Journal(journal_path("journal-routing")).read()
    routing = records[-1].detail.get("routing")
    assert isinstance(routing, dict)
    assert routing["accepted"] is False


def test_report_separates_needs_review_from_rejections() -> None:
    """«Ожидает проверки» — не отказ: в отчёте это разные разделы."""
    from pf.report import build_session_report

    session_id = "report-needs-review"
    Journal(journal_path(session_id)).append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            citations=["project/data/x.csv#row1"],
            rejected_details=["bad#p1: цитата не найдена"],
            needs_review_details=["web/Новости: внешний источник, требует проверки"],
            cost_units=0.001,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    text = report.as_markdown()

    assert len(report.needs_review_details) == 1
    assert "Ожидают проверки человеком" in text
    assert "## Отклонённые основания" in text
    # Доля подтверждённых считается по подтверждённым и отклонённым:
    # «ждёт проверки» не должно улучшать или ухудшать её молча.
    assert report.grounding_rate == pytest.approx(0.5)


def test_report_shows_lexical_coverage() -> None:
    """Отчёт показывает, сколько слов проверка поняла."""
    from pf.report import build_session_report

    session_id = "report-lexical"
    Journal(journal_path(session_id)).append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            detail={
                "lexical": {
                    "total": 10,
                    "known": 8,
                    "oov": 2,
                    "rate": 0.8,
                    "oov_tokens": ["криптовалюта"],
                }
            },
            cost_units=0.001,
        )
    )
    save(SessionState(session_id=session_id, phase=Phase.DONE))

    report = build_session_report(session_id)
    text = report.as_markdown()

    assert report.lexical is not None
    assert report.lexical["oov"] == 2
    assert "Лексическое покрытие" in text
    assert "криптовалюта" in text
