"""Проверки критериев успеха плана проверки.

План проверки отвечает на вопрос «как поймём, что сработало». В
демо-прогоне план был про удержание, а критерий успеха — «снижение
текучести кадров»: показатель другой, проверять его будет нечем.
Схема такое пропускала, потому что список критериев свободный.
"""

from __future__ import annotations

from pf.schemas import (
    CommunicationStrategy,
    ProductStrategy,
    Strategy,
    VerificationPlan,
)
from pf.workflow import Workflow


def _strategy(criteria: list[str], expected: str) -> Strategy:
    """Стратегия с заданным планом проверки.

    Args:
        criteria: Критерии успеха.
        expected: Ожидаемое изменение.

    Returns:
        Заполненная стратегия.
    """
    return Strategy(
        product=ProductStrategy(audience="клиенты тарифа", chosen_direction="снизить цену"),
        communication=CommunicationStrategy(
            audience="клиенты тарифа",
            barrier="цена выше альтернатив",
            key_message="мы держим цену под контролем",
        ),
        verification=VerificationPlan(
            expected_change=expected,
            compare_against="уровень прошлого квартала",
            observation_period="3 месяца",
            adverse_effects=["маржа снизится"],
            success_criteria=criteria,
        ),
    )


def test_success_criterion_must_reference_metric() -> None:
    """Критерий про другой показатель отклоняется."""
    strategy = _strategy(["Снижение текучести кадров"], "Удержание клиентов вырастет на 2 п.п.")

    notes = Workflow._check_success_criteria(strategy)

    assert notes
    assert "не относится к показателю" in notes[0]


def test_success_criterion_about_metric_passes() -> None:
    """Критерий о том же показателе принимается — иначе правило бесполезно."""
    strategy = _strategy(
        ["Увеличение удержания клиентов", "Рост удовлетворённости клиентов"],
        "Удержание клиентов вырастет на 2 п.п.",
    )

    assert Workflow._check_success_criteria(strategy) == []
