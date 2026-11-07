"""Проверки входов ролей: что план и роутер получают на вход.

План без пробелов и посчитанных чисел — это повтор, а не уточнение:
планировщик видел только постановку задачи и потому либо повторял её,
либо спрашивал заново то, на что уже есть ответ. Роутер без данных о
прогрессе решал «куда дальше» вслепую — в прогоне `demo-v5` он честно
написал «неясно, что именно выяснять», хотя разбор уже был.

Здесь проверяется, что оба получают то, что уже выяснено: подтверждённые
основания, пробелы разбора, гипотезы и посчитанные кодом метрики.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pf.config import settings
from pf.journal import Journal, StepRecord, journal_path
from pf.state import Phase, SessionState
from pf.workflow import Workflow


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать материалы, журнал и состояние.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Подмена настроек.
    """
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")
    monkeypatch.setattr(settings, "project_dir", tmp_path / "project")
    settings.project_dir.mkdir(parents=True, exist_ok=True)


def _workflow() -> Workflow:
    """Рабочий процесс без модели: входы не требуют вызова роли."""
    return Workflow(client=object(), config=None)  # type: ignore[arg-type]


def _state(phase: Phase) -> SessionState:
    """Состояние с постановкой, планом и результатом разбора."""
    state = SessionState(session_id="inputs", phase=phase, question="почему падает удержание")
    state.results[Phase.INTAKE.value] = {
        "question": "почему падает удержание",
        "goal": "снизить отток",
    }
    state.results[Phase.PLANNING.value] = {
        "questions": ["какой отток по сегментам"],
        "priority_order": [1],
        "gaps_expected": ["нет телеметрии по причинам"],
    }
    state.results[Phase.ANALYZING.value] = {
        "facts": [],
        "hypotheses": [
            {"kind": "гипотеза", "text": "отток из-за цены", "sources": []},
        ],
        "forecasts": [],
        "gaps": ["нет разбивки оттока по причинам"],
        "metrics": [],
    }
    return state


def _csv(name: str, rows: str) -> None:
    """Записать таблицу в каталог материалов.

    Args:
        name: Имя файла.
        rows: Содержимое таблицы.
    """
    (settings.project_dir / name).write_text(rows, encoding="utf-8")


def test_computed_metrics_have_structured_form() -> None:
    """Посчитанные метрики доступны как данные, а не только строкой.

    Плану нужно передать числа, а не текстовый блок: иначе «передать
    метрики в план» нечем — блок РАСЧЁТЫ существует только как текст для
    контекста модели.
    """
    _csv("numbers.csv", "оператор,показатель,значение\nМТС,абонентская база,87\n")

    records = Workflow._computed_metric_records()
    assert records, "метрики должны быть посчитаны по материалам"
    first = records[0]
    assert first.source_id == "project/numbers.csv"
    assert first.citation.endswith("#row0")
    assert first.formula


def test_planner_receives_gaps_and_metrics() -> None:
    """Планировщик получает пробелы разбора, гипотезы и метрики.

    Без этого повторный проход плана — повтор: у планировщика нет ни
    одного нового факта о том, что уже выяснено.
    """
    _csv("numbers.csv", "оператор,показатель,значение\nМТС,абонентская база,87\n")
    state = _state(Phase.PLANNING)

    role, payload, _context = _workflow()._phase_inputs(state, None)

    assert role == "planner"
    progress = payload["progress"]
    assert progress["gaps"] == ["нет разбивки оттока по причинам"]
    assert progress["hypotheses"] == ["отток из-за цены"]
    assert progress["metrics"], "метрики должны попасть в план"
    metric = progress["metrics"][0]
    assert metric["citation"].endswith("#row0")
    assert metric["formula"]
    assert payload["previous_plan"]["questions"] == ["какой отток по сегментам"]


def test_planner_first_pass_has_empty_progress() -> None:
    """На первом проходе прогресса ещё нет — и это не ошибка."""
    state = SessionState(session_id="inputs", phase=Phase.PLANNING, question="вопрос")
    state.results[Phase.INTAKE.value] = {"goal": "цель"}

    role, payload, _context = _workflow()._phase_inputs(state, None)

    assert role == "planner"
    assert payload["progress"]["gaps"] == []
    assert payload["progress"]["confirmed"] == []
    assert "previous_plan" not in payload


def test_router_sees_progress() -> None:
    """Роутер видит, что уже выяснено, а не только постановку и план."""
    state = _state(Phase.ROUTING)
    journal = Journal(journal_path(state.session_id))
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            citations=["project/data/x.csv#row0"],
            rejected_details=["Показатели: цитата не найдена в строке таблицы"],
        )
    )

    role, payload, _context = _workflow()._phase_inputs(state, None)

    assert role == "router"
    progress = payload["progress"]
    assert progress["confirmed"] == ["project/data/x.csv#row0"]
    assert progress["rejected"] == 1
    assert progress["gaps"]


def test_calculations_block_still_renders_per_source() -> None:
    """Блок РАСЧЁТЫ по-прежнему группируется по источнику.

    Структурированные метрики — добавка, а не замена: сборщик, аналитик и
    стратег получают блок текстом, и он обязан остаться читаемым.
    """
    _csv("numbers.csv", "оператор,показатель,значение\nМТС,абонентская база,87\n")
    _csv("rates.csv", "оператор,показатель,значение\nМТС,ключевая ставка,16\n")

    text = Workflow._computed_metrics(_state(Phase.COLLECTING))

    assert "РАСЧЁТЫ по project/numbers.csv" in text
    assert "РАСЧЁТЫ по project/rates.csv" in text
    assert "source_id копируется отсюда" in text


def test_progress_is_capped() -> None:
    """Прогресс обрезан: длинный вход ломает формат ответа модели."""
    state = _state(Phase.PLANNING)
    analysis: dict[str, Any] = state.results[Phase.ANALYZING.value]
    analysis["gaps"] = [f"пробел {index}" for index in range(20)]
    analysis["hypotheses"] = [
        {"kind": "гипотеза", "text": f"гипотеза {index}", "sources": []} for index in range(20)
    ]

    _role, payload, _context = _workflow()._phase_inputs(state, None)

    assert len(payload["progress"]["gaps"]) <= 8
    assert len(payload["progress"]["hypotheses"]) <= 8
    assert len(payload["progress"]["metrics"]) <= 8
