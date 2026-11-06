"""Проверки схем вывода: агент не может выдать бессмысленный результат.

Схемы — это не описание, а **ограничения**. Каждый тест ниже проверяет,
что нарушение формы отвергается, а не молча проходит.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pf.schemas import (
    Analysis,
    Claim,
    CommunicationStrategy,
    DecisionRecord,
    NextStep,
    ProductStrategy,
    ResearchPlan,
    Strategy,
    TaskFrame,
    Variant,
    Variants,
    VerificationPlan,
)


def _variant(title: str, **kwargs: object) -> Variant:
    return Variant(title=title, summary="суть", objections=["возражение"], **kwargs)


def test_fact_without_source_is_rejected() -> None:
    """Факт без ссылки на источник — не факт."""
    with pytest.raises(ValidationError):
        Claim(kind="факт", text="утверждение")


def test_hypothesis_without_source_is_allowed() -> None:
    """Гипотеза может быть без источника — она и есть предположение."""
    claim = Claim(kind="гипотеза", text="предположение")
    assert claim.sources == []


def test_empty_analysis_is_rejected() -> None:
    """Пустой разбор — не результат."""
    with pytest.raises(ValidationError):
        Analysis()


def test_fewer_than_three_variants_is_rejected() -> None:
    """Требование: не меньше трёх содержательно разных вариантов."""
    with pytest.raises(ValidationError):
        Variants(variants=[_variant("А")])


def test_variant_without_objection_is_rejected() -> None:
    """Вариант без сильного возражения не проверен."""
    with pytest.raises(ValidationError):
        Variant(title="А", summary="суть", objections=[])


def test_identical_variant_titles_are_rejected() -> None:
    """Три одинаково названных варианта — это один вариант."""
    with pytest.raises(ValidationError):
        Variants(variants=[_variant("Одинаковый")] * 3)


def test_three_distinct_variants_pass() -> None:
    """Три разных варианта проходят."""
    variants = Variants(variants=[_variant("А"), _variant("Б"), _variant("В")])
    assert len(variants.variants) == 3


def test_task_frame_requires_core_fields() -> None:
    """Постановка задачи без цели бесполезна."""
    with pytest.raises(ValidationError):
        TaskFrame(question="вопрос", goal="", audience="кто-то")


def test_research_plan_requires_questions() -> None:
    """План без вопросов — не план."""
    with pytest.raises(ValidationError):
        ResearchPlan(questions=[])


def test_verification_plan_requires_all_four_parts() -> None:
    """План проверки обязан содержать изменение, сравнение, срок."""
    with pytest.raises(ValidationError):
        VerificationPlan(expected_change="", compare_against="база", observation_period="квартал")


def test_strategy_requires_all_three_parts() -> None:
    """Стратегия без коммуникационной части не принимается."""
    from pf.schemas import CommunicationStrategy

    with pytest.raises(ValidationError):
        Strategy(
            product=ProductStrategy(audience="а", chosen_direction="б"),
            communication=None,  # type: ignore[arg-type]
            verification=VerificationPlan(
                expected_change="а", compare_against="б", observation_period="в"
            ),
        )

    with pytest.raises(ValidationError):
        Strategy(
            product=ProductStrategy(audience="а", chosen_direction="б"),
            communication=CommunicationStrategy(audience="а", barrier="б", key_message="в"),
            verification=None,  # type: ignore[arg-type]
        )


def test_strategy_accepts_lists_where_string_expected() -> None:
    """Модель отвечает списком там, где ждём строку — это не ошибка."""
    strategy = Strategy(
        product=ProductStrategy(
            audience="а",
            chosen_direction="б",
            value_to_cost=["x", "y"],
            alternatives=[{"title": "Альтернатива 1"}],
        ),
        communication=CommunicationStrategy(
            audience="а",
            barrier="б",
            key_message="в",
            channels=["канал 1", "канал 2"],
            proof=["подтверждение 1"],
        ),
        verification=VerificationPlan(
            expected_change="а",
            compare_against="б",
            observation_period="в",
            adverse_effects=[{"title": "Побочный эффект"}],
        ),
    )
    assert strategy.product.value_to_cost == "x; y"
    assert strategy.product.alternatives == ["Альтернатива 1"]
    assert strategy.communication.channels == "канал 1; канал 2"
    assert strategy.verification.adverse_effects == ["Побочный эффект"]


def test_empty_strategy_block_is_rejected() -> None:
    """Пустой блок стратегии отвергается — шаг будет повторён с подсказкой."""
    with pytest.raises(ValidationError):
        Strategy(
            product=ProductStrategy(audience="а", chosen_direction="б"),
            communication=CommunicationStrategy(audience="", barrier="", key_message=""),
            verification=VerificationPlan(
                expected_change="а", compare_against="б", observation_period="в"
            ),
        )


def test_decision_record_requires_decision() -> None:
    """Запись о решении без самого решения бессмысленна."""
    with pytest.raises(ValidationError):
        DecisionRecord(decision="")


def test_next_step_requires_role() -> None:
    """Маршрут без роли — не маршрут."""
    with pytest.raises(ValidationError):
        NextStep(role="")
