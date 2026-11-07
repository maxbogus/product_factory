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


def test_variants_accept_bare_list() -> None:
    """Голый список вариантов принимается.

    Модель часто отвечает именно так — и раньше это давало ошибку
    валидации вместо принятого ответа.
    """
    variants = Variants.model_validate(
        [
            {"title": "А", "summary": "с", "objections": ["о"]},
            {"title": "Б", "summary": "с", "objections": ["о"]},
            {"title": "В", "summary": "с", "objections": ["о"]},
        ]
    )
    assert len(variants.variants) == 3
    assert all(item.objections for item in variants.variants)


def test_key_message_cannot_name_a_competitor() -> None:
    """Ключевое сообщение не должно называть чужой бренд.

    Коммуникация обращается к аудитории нашего продукта. Если в
    сообщении назван конкурент, это не наша коммуникация — так в
    демонстрации вышло «МТС предлагает прозрачные тарифы» для задачи
    Фабрики Мегафона. Ловим кодом, а не просьбой модели.
    """
    from pf.schemas import CommunicationStrategy

    with pytest.raises(ValidationError):
        CommunicationStrategy(
            audience="Клиенты",
            barrier="Низкая прозрачность тарифов",
            key_message="МТС предлагает прозрачные тарифы и гибкие условия",
            proof="Информирование о подписках",
        )


def test_key_message_without_competitor_is_accepted() -> None:
    """Сообщение без чужого бренда принимается."""
    from pf.schemas import CommunicationStrategy

    strategy = CommunicationStrategy(
        audience="Клиенты",
        barrier="Низкая прозрачность тарифов",
        key_message="Мы сделали тарифы прозрачными: вы видите все подписки",
        proof="Информирование о подписках",
    )
    assert "МТС" not in strategy.key_message


def test_barrier_may_mention_competitor() -> None:
    """В барьере конкурентMention конкурента можно: это контекст рынка."""
    from pf.schemas import CommunicationStrategy

    strategy = CommunicationStrategy(
        audience="Клиенты",
        barrier="Рынок привык к навязчивым подпискам МТС, и доверие к обещаниям падает",
        key_message="Мы показываем все подписки до подтверждения",
        proof="Информирование о подписках",
    )
    assert "МТС" in strategy.barrier


def test_verification_requires_adverse_effects() -> None:
    """План проверки без нежелательных последствий неполон.

    Требование брифа: «план проверки — ожидаемое изменение, с чем
    сравниваем, срок наблюдения и нежелательные последствия». Пустой
    список последствий — это не «последствий нет», это отсутствующая
    часть плана.
    """
    from pf.schemas import VerificationPlan

    with pytest.raises(ValidationError):
        VerificationPlan(
            expected_change="Рост удержания на 3 %",
            compare_against="База января",
            observation_period="3 месяца",
            adverse_effects=[],
        )


def test_verification_accepts_adverse_effects() -> None:
    """С нежелательными последствиями план принимается."""
    from pf.schemas import VerificationPlan

    plan = VerificationPlan(
        expected_change="Рост удержания на 3 %",
        compare_against="База января",
        observation_period="3 месяца",
        adverse_effects=["Падение маржи ниже 2 % — откатываем на сегмент A"],
    )
    assert plan.adverse_effects


def test_analysis_accepts_metrics_with_provenance() -> None:
    """Разбор должен принимать расчётные метрики со ссылкой на строку."""
    from pf.schemas import Analysis

    analysis = Analysis(
        facts=[],
        hypotheses=[],
        forecasts=[],
        gaps=[],
        metrics=[
            {
                "name": "Абонентская база, 2026 Q1",
                "value": 110.7,
                "unit": "млн",
                "source_id": "project/data/numbers.csv",
                "row": 4,
                "formula": "значение из строки 5",
                "inputs": {"значение": "110,7"},
            }
        ],
    )
    assert len(analysis.metrics) == 1
    metric = analysis.metrics[0]
    assert metric.value == pytest.approx(110.7)
    assert metric.source_id == "project/data/numbers.csv"
    assert metric.citation == "project/data/numbers.csv#row4"


def test_analysis_metric_without_source_is_marked_not_reproducible() -> None:
    """Метрика без источника принимается, но помечается как непроверяемая."""
    from pf.schemas import Analysis

    analysis = Analysis(
        metrics=[{"name": "Что-то", "value": 1.0, "unit": "шт"}],
    )
    assert analysis.metrics[0].is_reproducible is False


def test_strategy_accepts_flat_product_answer() -> None:
    """Плоский ответ стратегии должен собираться в блоки.

    Модель иногда отвечает так: `product` — короткая строка, а поля
    разложены на верхний уровень ответа. Без поддержки этого весь
    шаг терялся: стратегия отбрасывалась, хотя по сути была заполнена.
    """
    from pf.schemas import Strategy

    strategy = Strategy.model_validate(
        {
            "product": "тариф",
            "audience": "Абоненты мобильного тарифа",
            "alternatives": ["Оставить как есть"],
            "chosen_direction": "Прозрачность тарифов",
            "deliberate_omissions": ["Не снижать цену"],
            "value_to_cost": "Рост удержания окупает потерю маржи",
            "evidence": [],
            "communication": {
                "audience": "Клиенты",
                "barrier": "Непрозрачность подписок",
                "key_message": "Покажем все подписки до подтверждения",
            },
            "verification": {
                "expected_change": "Рост удержания на 3 %",
                "compare_against": "База января",
                "observation_period": "6 месяцев",
                "adverse_effects": ["Падение маржи"],
            },
        }
    )
    assert strategy.product.audience == "Абоненты мобильного тарифа"
    assert strategy.product.chosen_direction == "Прозрачность тарифов"
    assert strategy.communication.key_message


def test_strategy_accepts_field_synonyms() -> None:
    """Синонимы полей продуктовой стратегии принимаются."""
    from pf.schemas import Strategy

    strategy = Strategy.model_validate(
        {
            "product": {
                "аудитория": "Клиенты",
                "name": "Прозрачность тарифов",
                "отказы": ["Не снижать цену"],
            },
            "communication": {
                "audience": "Клиенты",
                "barrier": "Непрозрачность",
                "key_message": "Покажем подписки",
            },
            "verification": {
                "expected_change": "Рост",
                "compare_against": "База",
                "observation_period": "6 месяцев",
                "adverse_effects": ["Маржа"],
            },
        }
    )
    assert strategy.product.audience == "Клиенты"
    assert strategy.product.chosen_direction == "Прозрачность тарифов"
