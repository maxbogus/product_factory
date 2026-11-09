"""Ограничения задачи: модель «действие + полярность» и проверка направления.

Ограничения собираются на входе (``TaskFrame.constraints``) и раньше только
подставлялись в промпт. В демо-прогоне до ``demo-v9`` ограничение «цену
снижать нельзя» потерялось, и выбранным направлением стало «снизить цену» —
никто этого не заметил.

Здесь проверяется модель: запрет разбирается в структуру (действие, объект,
полярность), а выбранное направление сверяется с запретом по понятиям, а не
по точной строке. Ложное срабатывание опаснее пропуска: запрет «не менять
цену» не должен блокировать «сделать тариф гибким».
"""

from __future__ import annotations

from pf.constraints import ConstraintKind, check_direction, parse_constraints
from pf.report import SessionReport
from pf.schemas import CommunicationStrategy, ProductStrategy, Strategy, VerificationPlan
from pf.state import Phase, SessionState
from pf.workflow import Workflow


def _strategy(direction: str) -> Strategy:
    """Стратегия с выбранным направлением и заполненными обязательными блоками.

    Args:
        direction: Выбранное направление.

    Returns:
        Заполненная стратегия.
    """
    return Strategy(
        product=ProductStrategy(audience="клиенты тарифа", chosen_direction=direction),
        communication=CommunicationStrategy(
            audience="клиенты тарифа",
            barrier="цена выше альтернатив",
            key_message="мы держим цену под контролем",
        ),
        verification=VerificationPlan(
            expected_change="удержание вырастет на 2 п.п.",
            compare_against="уровень прошлого квартала",
            observation_period="3 месяца",
            adverse_effects=["маржа снизится"],
            success_criteria=["Рост удержания клиентов"],
        ),
    )


def _state(*constraints: str) -> SessionState:
    """Состояние сессии с ограничениями в постановке задачи.

    Args:
        constraints: Ограничения задачи.

    Returns:
        Состояние на фазе входа с заполненным кадром задачи.
    """
    state = SessionState(session_id="s", phase=Phase.INTAKE)
    state.results[Phase.INTAKE.value] = {"constraints": list(constraints)}
    return state


def test_constraint_model_parses_prohibition() -> None:
    """«цену снижать нельзя» — запрет действия «снизить цену»."""
    constraints = parse_constraints(["цену снижать нельзя"])

    assert len(constraints) == 1
    parsed = constraints[0]
    assert parsed.kind is ConstraintKind.FORBIDDEN
    assert parsed.action == "снизить"
    assert parsed.target == "цена"


def test_constraint_model_separates_limit_from_prohibition() -> None:
    """«учитывать ограничения» — рамка, а не запрет действия."""
    parsed = parse_constraints(
        [
            "Необходимо учитывать бюджетные ограничения",
            "цену снижать нельзя",
        ]
    )
    kinds = {constraint.text: constraint.kind for constraint in parsed}

    assert kinds["Необходимо учитывать бюджетные ограничения"] is ConstraintKind.LIMITED
    assert kinds["цену снижать нельзя"] is ConstraintKind.FORBIDDEN


def test_constraint_model_finds_violation_across_word_forms() -> None:
    """«снижение» и «снизить» — одно действие, а не разные строки."""
    constraints = parse_constraints(["снижение цены запрещено"])

    violations = check_direction("Провести снижение цены", constraints)

    assert len(violations) == 1
    assert violations[0].constraint is constraints[0]
    assert violations[0].claim == "Провести снижение цены"


def test_constraint_model_does_not_block_other_formulation() -> None:
    """Запрет менять цену не запрещает сделать тариф гибким."""
    constraints = parse_constraints(["цену менять нельзя"])

    assert check_direction("Сделать тариф гибким", constraints) == []


def test_constraint_model_respects_negated_direction() -> None:
    """Направление, которое само отрицает запрещённое действие, — не нарушение."""
    constraints = parse_constraints(["цену снижать нельзя"])

    assert check_direction("Не снижать цену, а повысить качество", constraints) == []


def test_constraint_model_ignores_non_action_constraint() -> None:
    """Ограничение без действия не порождает нарушения."""
    constraints = parse_constraints(["Сроки анализа ограничены до конца месяца"])

    assert check_direction("Запустить новую линейку тарифов", constraints) == []


def test_direction_respects_constraints() -> None:
    """Направление против запрета — названное нарушение, а не тихий пропуск."""
    state = _state("цену снижать нельзя")

    check = Workflow._check_constraints(_strategy("Снизить цену на тарифы"), state)

    assert check["checked"] == 1
    assert check["violations"]
    assert "цену снижать нельзя" in check["violations"][0]


def test_direction_respects_constraints_when_formulation_changes() -> None:
    """Запрет «не менять цену» не блокирует изменение формы тарифа."""
    state = _state("цену менять нельзя")

    check = Workflow._check_constraints(_strategy("Сделать тариф гибким"), state)

    assert check["checked"] == 1
    assert check["violations"] == []


def test_direction_respects_constraints_named_in_report() -> None:
    """Нарушение называет отчёт отдельной строкой."""
    report = SessionReport(
        session_id="s",
        constraints=[
            "направление «Снизить цену» выполняет действие, которое ограничение "
            "«цену снижать нельзя» запрещает"
        ],
    )

    assert "Ограничения задачи нарушены" in report.as_markdown()
