"""Проверки трекинга прогонов.

Главное правило: наблюдаемость не должна ломать то, что наблюдают.
Если MLflow не установлен, выключен или падает — прогон идёт дальше,
а измерение всё равно попадает в файл.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pf.tracking import Accumulator, RunMetrics, is_enabled, log_run


def test_metrics_are_numeric() -> None:
    """Метрики — числа: иначе MLflow их не примет."""
    metrics = RunMetrics(
        task="удержание",
        model="qwen3:8b",
        provider="ollama",
        steps=9,
        citations=4,
        total_tokens=21683,
        duration_sec=108.5,
        cost_units=0.021683,
    )
    values = metrics.as_metrics()
    assert values["citations"] == 4
    assert values["total_tokens"] == 21683
    assert values["duration_sec"] == 108.5
    assert all(isinstance(value, float) for value in values.values())


def test_params_skip_empty_values() -> None:
    """Пустые параметры не пишутся."""
    metrics = RunMetrics(task="удержание", model="qwen3:8b")
    params = metrics.as_params()
    assert params["task"] == "удержание"
    assert "errors" not in params


def test_log_run_is_noop_without_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без MLflow запись не падает и не выдаёт идентификатор."""
    monkeypatch.setattr("pf.tracking._mlflow", lambda: None)
    assert log_run(RunMetrics(task="t")) == ""


def test_tracking_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Трекинг выключается флагом — без удаления пакета."""
    monkeypatch.setenv("MLFLOW_DISABLED", "1")
    assert is_enabled() is False


def test_accumulator_writes_jsonl(tmp_path: Path) -> None:
    """Накопитель пишет измерения в файл построчно."""
    path = tmp_path / "runs.jsonl"
    accumulator = Accumulator(path=str(path))
    accumulator.add(RunMetrics(task="t", model="qwen3:8b", citations=3))
    accumulator.add(RunMetrics(task="t", model="qwen3:8b", citations=5))

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["citations"] == 3.0
    assert first["model"] == "qwen3:8b"


def test_accumulator_summary_has_min_max_avg(tmp_path: Path) -> None:
    """Сводка показывает разброс — он и есть главный вывод о моделях."""
    accumulator = Accumulator(path=str(tmp_path / "runs.jsonl"))
    accumulator.add(RunMetrics(citations=2, total_tokens=1000))
    accumulator.add(RunMetrics(citations=6, total_tokens=3000))

    summary = accumulator.summary()
    assert summary["runs"] == 2
    assert summary["citations_avg"] == 4.0
    assert summary["citations_min"] == 2.0
    assert summary["citations_max"] == 6.0


def test_empty_accumulator_summary(tmp_path: Path) -> None:
    """Пустой накопитель не ломает отчёт."""
    accumulator = Accumulator(path=str(tmp_path / "runs.jsonl"))
    assert accumulator.summary() == {"runs": 0}
