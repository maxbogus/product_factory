"""Проверки хуков: действия по событиям.

Хук — правило «когда произошло событие X, сделай Y». Правила лежат в
``hooks/*.yaml``, а выполняются в коде: иначе обещание в README
расходится с поведением.

Проверяются три вещи: правило читается, условие срабатывает там, где
должно, и действие действительно происходит.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.config import settings
from pf.hooks import Hook, expand_plan, fire, load_hooks
from pf.state import SessionState


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать состояние и журнал."""
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")


def test_hooks_are_loaded_from_files() -> None:
    """Правила читаются из каталога hooks/."""
    hooks = load_hooks()
    assert hooks
    events = {hook.event for hook in hooks}
    assert "sources.collected" in events
    assert "budget.exhausted" in events


def test_hook_condition_works() -> None:
    """Условие «< 2» срабатывает на одном основании и не на пяти."""
    hook = Hook(event="e", when="оснований < 2", threshold=2.0, action="a")
    assert hook.matches(1)
    assert not hook.matches(2)
    assert not hook.matches(5)


def test_hook_operator_variants() -> None:
    """Операторы сравнения распознаются."""
    assert Hook(event="e", when="x <= 2", threshold=2.0, action="a").operator == "<="
    assert Hook(event="e", when="x >= 2", threshold=2.0, action="a").operator == ">="
    assert Hook(event="e", when="x < 2", threshold=2.0, action="a").operator == "<"


def test_fire_returns_matching_hook() -> None:
    """При срабатывании возвращается действие."""
    hooks = [Hook(event="e", when="x < 2", threshold=2.0, action="расширить")]
    outcome = fire("e", 1, hooks)
    assert outcome.fired
    assert outcome.action == "расширить"


def test_fire_ignores_other_events() -> None:
    """Хук по чужому событию не срабатывает."""
    hooks = [Hook(event="e", when="x < 2", threshold=2.0, action="a")]
    assert not fire("other", 1, hooks).fired


def test_expand_plan_adds_questions() -> None:
    """План расширяется новыми вопросами."""
    state = SessionState(session_id="h1")
    state.results["planning"] = {"questions": ["первый"]}
    added = expand_plan(state)
    assert added > 0
    questions = state.results["planning"]["questions"]
    assert len(questions) == 1 + added
    assert "первый" in questions


def test_expand_plan_is_idempotent() -> None:
    """Повторное расширение не дублирует вопросы."""
    state = SessionState(session_id="h2")
    state.results["planning"] = {"questions": []}
    expand_plan(state)
    first = len(state.results["planning"]["questions"])
    expand_plan(state)
    assert len(state.results["planning"]["questions"]) == first
