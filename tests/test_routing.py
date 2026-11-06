"""Проверки маршрутизации.

Роутер выбирает, кто работает следующим. Его решение может быть
ошибочным — и тогда его надо **исправить**, а не выполнить: бриф прямо
требует «исправления ошибочного выбора».

Также проверяется подключение человека: агент сам ведёт работу, но
останавливается там, где без решения человека работа потеряла бы смысл.
"""

from __future__ import annotations

from typing import Any

from pf.schemas import NextStep
from pf.state import Phase, SessionState
from pf.workflow import StepResult, Workflow


class StubClient:
    """Модель, которая отвечает заранее заданным JSON."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    def complete(self, system: str, user: str) -> Any:
        from pf.llm import LLMResponse, LLMUsage

        role = "unknown"
        for marker, name in (
            ("интик", "intake"),
            ("планировщик", "planner"),
            ("сборщик", "collector"),
            ("аналитик", "analyst"),
            ("генератор", "options"),
            ("критик", "redteam"),
            ("стратег", "strategy"),
            ("decision-keeper", "keeper"),
            ("роутер", "router"),
        ):
            if marker in system:
                role = name
                break
        return LLMResponse(
            content=__import__("json").dumps(self.answers[role], ensure_ascii=False),
            usage=LLMUsage(prompt_tokens=10, completion_tokens=10),
        )


def _workflow(answers: dict[str, Any]) -> Workflow:
    return Workflow(client=StubClient(answers), config=None)  # type: ignore[arg-type]


def _result(data: Any) -> StepResult:
    return StepResult(role="router", phase=Phase.ROUTING, data=data)


def _state(phase: Phase = Phase.ROUTING) -> SessionState:
    return SessionState(session_id="t1", phase=phase, question="задача")


def test_router_choice_is_respected() -> None:
    """Роутер выбрал роль — работа идёт к ней."""
    workflow = _workflow({})
    state = _state()
    result = _result(NextStep(role="options", reason="нужны варианты"))

    assert workflow.route_next(state, result) is Phase.OPTIONS
    assert "options" in state.results["routing_note"]


def test_unknown_role_falls_back_to_order() -> None:
    """Неизвестная роль — работа идёт по порядку."""
    workflow = _workflow({})
    state = _state()
    result = _result(NextStep(role="дирижёр", reason=""))

    assert workflow.route_next(state, result) is Phase.COLLECTING
    assert "неизвестен" in state.results["routing_note"]


def test_backward_route_is_corrected() -> None:
    """Маршрут назад — исправляется, работа продолжается по порядку."""
    workflow = _workflow({})
    state = _state(Phase.STRATEGY)
    result = _result(NextStep(role="collector", reason="надо пересобрать"))

    assert workflow.route_next(state, result) is Phase.KEEPER
    assert "назад" in state.results["routing_note"]


def test_same_phase_is_corrected() -> None:
    """Маршрут в ту же фазу — это зацикливание, и оно исправляется."""
    workflow = _workflow({})
    state = _state(Phase.ANALYZING)
    result = _result(NextStep(role="analyst", reason="ещё раз"))

    assert workflow.route_next(state, result) is Phase.OPTIONS
    assert "назад" in state.results["routing_note"]


def test_non_router_result_keeps_order() -> None:
    """Если результат не от роутера — порядок фаз не меняется."""
    workflow = _workflow({})
    state = _state(Phase.ANALYZING)
    result = StepResult(role="analyst", phase=Phase.ANALYZING, data={"facts": []})

    assert workflow.route_next(state, result) is Phase.OPTIONS


def test_router_can_ask_human() -> None:
    """Роутер вправе попросить человека о существенном выборе."""
    workflow = _workflow({})
    result = _result(
        NextStep(role="options", reason="выбор неоднозначен", need_human=True, question="А или Б?")
    )

    assert workflow.route_next(_state(), result) is Phase.OPTIONS
    assert result.data.need_human is True
    assert result.data.question == "А или Б?"
