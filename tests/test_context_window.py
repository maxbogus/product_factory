"""Проверки окна контекста: обрезка промпта не должна быть незаметной.

Ollama режет промпт по ``num_ctx``. На сохранённых журналах видно, что шаги
``collector``, ``analyst`` и ``strategy`` упирались ровно в 4096 токенов:
начало контекста отбрасывалось молча, а наши блоки (ограничения задачи,
бюджет ответа) стояли в конце и выживали — то есть вытесняли материалы.

Здесь проверяется вторая половина починки: окно задаётся явно (см.
``tests/test_llm_client.py``), а признак обрезки попадает в журнал и в
отчёт. Без этого нельзя сказать, помогла настройка или нет: «модель
ответила» и «модель видела весь контекст» — разные утверждения.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pf.config import settings
from pf.journal import Journal, journal_path
from pf.llm import LLMResponse, LLMUsage
from pf.report import build_session_report
from pf.state import Phase, SessionState
from pf.workflow import Workflow

#: Окно, с которым проверяется обрезка.
LIMIT = 8192


class _Client:
    """Модель с заранее заданным расходом.

    Расход — единственное, что здесь важно: обрезка видна именно по
    ``prompt_eval_count``, а не по тексту ответа.
    """

    def __init__(self, usage: LLMUsage) -> None:
        """Запомнить расход.

        Args:
            usage: Расход вызова.
        """
        self.usage = usage

    def complete(self, system: str, user: str, *, model: str = "") -> LLMResponse:
        """Ответить заготовкой интика.

        Args:
            system: Системная инструкция.
            user: Текст задачи.

        Returns:
            Ответ модели с заданным расходом.
        """
        answer: dict[str, Any] = {
            "question": "Почему падает удержание?",
            "goal": "увеличить удержание мобильного тарифа",
            "audience": "команда по управлению тарифом",
            "constraints": ["цену снижать нельзя"],
            "available_data": [],
            "open_questions": [],
        }
        return LLMResponse(content=json.dumps(answer, ensure_ascii=False), usage=self.usage)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать состояние, журнал и очередь распознания.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Подмена настроек.
    """
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")
    monkeypatch.setattr(settings, "lexicon_dir", tmp_path / "lexicon")


def _run_step(session_id: str, usage: LLMUsage) -> SessionState:
    """Выполнить один шаг интика с заданным расходом.

    Args:
        session_id: Идентификатор сессии.
        usage: Расход вызова модели.

    Returns:
        Состояние после шага.
    """
    state = SessionState(
        session_id=session_id,
        phase=Phase.INTAKE,
        question="Почему падает удержание?",
    )
    Workflow(client=_Client(usage)).run_step(state, None)
    return state


def test_truncated_prompt_is_written_to_journal() -> None:
    """Признак обрезки попадает в журнал шага, а не только в память."""
    _run_step("ctx-truncated", LLMUsage(prompt_tokens=LIMIT, context_limit=LIMIT))

    record = Journal(journal_path("ctx-truncated")).read()[0]
    context = record.detail["context"]

    assert context == {"limit": LIMIT, "prompt_tokens": LIMIT, "truncated": True}


def test_prompt_inside_window_is_not_called_truncated() -> None:
    """Промпт, влезший в окно, обрезкой не считается."""
    _run_step("ctx-ok", LLMUsage(prompt_tokens=3701, context_limit=LIMIT))

    record = Journal(journal_path("ctx-ok")).read()[0]

    assert record.detail["context"]["truncated"] is False
    assert record.detail["context"]["limit"] == LIMIT


def test_report_shows_context_window() -> None:
    """Отчёт называет окно и число обрезанных шагов."""
    _run_step("ctx-report", LLMUsage(prompt_tokens=LIMIT, context_limit=LIMIT))

    report = build_session_report("ctx-report")
    markdown = report.as_markdown()

    assert report.context == {"steps": 1, "truncated": 1, "limit": LIMIT}
    assert "Окно контекста" in markdown
    assert f"| {LIMIT} |" in markdown
    assert any("окно контекста" in item for item in report.limitations)


def test_report_without_truncation_has_no_such_limitation() -> None:
    """Если обрезки не было, оговорки о ней тоже нет.

    Оговорка, которая висит всегда, перестаёт читаться: границы замера
    должны быть правдой о конкретном прогоне.
    """
    _run_step("ctx-clean", LLMUsage(prompt_tokens=3701, context_limit=LIMIT))

    report = build_session_report("ctx-clean")

    assert report.context == {"steps": 1, "truncated": 0, "limit": LIMIT}
    assert not any("окно контекста" in item for item in report.limitations)
