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
    MetricRef,
    NextStep,
    ProductStrategy,
    ResearchPlan,
    Strategy,
    TaskFrame,
    Variant,
    Variants,
    VerificationPlan,
)


def test_metric_without_formula_is_rejected() -> None:
    """Метрика без формулы — не расчёт: пересчитать её нечем.

    Докстринг схемы обещает «обязательно формулу», а поле принимало
    пустую строку. Модель этим пользовалась: присылала значение и
    строку, а формулу опускала — и число попадало в отчёт как
    подтверждённое, хотя проверить его пересчётом было нельзя.
    """
    with pytest.raises(ValidationError):
        MetricRef(
            name="Абонентская база МТС",
            value=82.0,
            unit="млн",
            source_id="project/data/numbers.csv",
            row=0,
        )


def test_metric_with_formula_is_accepted() -> None:
    """С формулой метрика принимается: её можно пересчитать."""
    metric = MetricRef(
        name="Абонентская база МТС",
        value=82.0,
        unit="млн",
        source_id="project/data/numbers.csv",
        row=0,
        formula="значение из колонки «значение», строка 1",
        inputs={"значение": "82"},
    )
    assert metric.formula


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
    """Три одинаково названных варианта — это один вариант.

    Причина названа точно: проверка полюсов идёт следом и тоже отклонила бы
    такой набор, но тогда ошибка говорила бы не о том, что случилось.
    """
    with pytest.raises(ValidationError, match="различаться"):
        Variants(variants=[_variant("Одинаковый")] * 3)


def test_three_distinct_variants_pass() -> None:
    """Три разных варианта с обоими полюсами проходят."""
    variants = Variants(
        variants=[
            _variant("Оставить как есть"),
            _variant("Закрыть направление"),
            _variant("Снизить цену"),
        ]
    )
    assert len(variants.variants) == 3


# ── Полюса вариантов ─────────────────────────────────────────────────────────
# Полюс — свойство набора, а не декларации. Проверялось обратное: набор
# «А/Б/В» проходил схему с пустыми маркерами, а обработчики eval-кейсов
# `D-05`/`D-06` возвращали ровно ту строку, которую искал кейс, — упасть они
# не могли. В двух сохранённых прогонах из 25 маркер «Закрыть направление»
# стоял при наборе, где варианта с таким названием нет.


def _variant_list(*titles: str) -> list[Variant]:
    """Собрать набор вариантов по названиям.

    Args:
        *titles: Названия вариантов.

    Returns:
        Список вариантов с возражениями.
    """
    return [_variant(title) for title in titles]


def test_poles_missing_keep_current_is_rejected() -> None:
    """Набор без «оставить как есть» — не с чем сравнивать предлагаемое."""
    with pytest.raises(ValidationError, match="оставить как есть"):
        Variants(variants=_variant_list("Снизить цену", "Усилить сервис", "Закрыть направление"))


def test_poles_missing_close_direction_is_rejected() -> None:
    """Набор без «закрыть направление» не допускает, что направление не работает."""
    with pytest.raises(ValidationError, match="закрыть направление"):
        Variants(variants=_variant_list("Снизить цену", "Усилить сервис", "Оставить как есть"))


def test_poles_full_set_is_accepted() -> None:
    """Оба полюса каноническими названиями — набор проходит."""
    variants = Variants(
        variants=_variant_list("Оставить как есть", "Закрыть направление", "Снизить цену")
    )
    assert variants.keep_current_marker == "Оставить как есть"
    assert variants.close_direction_marker == "Закрыть направление"


def test_poles_synonym_title_is_a_pole() -> None:
    """Синоним полюса принимается, а маркер получает фактическое название.

    Маркер не подменяется канонической строкой: в журнале и отчёте должно
    стоять название, которое есть в наборе, иначе документ сошлётся на
    вариант, которого нет.
    """
    variants = Variants(
        variants=_variant_list("Оставить без изменений", "Свернуть направление", "Снизить цену")
    )
    assert variants.keep_current_marker == "Оставить без изменений"
    assert variants.close_direction_marker == "Свернуть направление"


def test_poles_declared_marker_does_not_replace_variant() -> None:
    """Объявленный маркер не заменяет вариант.

    В прогоне `session-bench-agent-qwen2-5-7b-215753-1` маркер «Закрыть
    направление» стоял при трёх вариантах без такого названия. Проверка шла
    по маркеру, поэтому расхождение не было видно.
    """
    payload = {
        "variants": [
            variant.model_dump()
            for variant in _variant_list("Оставить как есть", "Снизить цену", "Усилить сервис")
        ],
        "keep_current_marker": "Оставить как есть",
        "close_direction_marker": "Закрыть направление",
    }
    with pytest.raises(ValidationError, match="закрыть направление"):
        Variants.model_validate(payload)


def test_poles_error_names_both_required_concepts() -> None:
    """Отказ называет оба обязательных понятия.

    Модель исправляет ответ по тексту ошибки: «нужен полюс» без названий
    заставил бы её угадывать формулировку, и шаг тратил бы попытки.
    """
    with pytest.raises(ValidationError) as error:
        Variants(variants=_variant_list("Снизить цену", "Усилить сервис", "Повысить прозрачность"))
    message = str(error.value).lower().replace("ё", "е")
    assert "оставить как есть" in message
    assert "закрыть направление" in message


def test_poles_negated_title_is_not_a_pole() -> None:
    """«Не закрывать направление» — не полюс, хотя формулировка внутри есть."""
    with pytest.raises(ValidationError, match="закрыть направление"):
        Variants(
            variants=_variant_list("Оставить как есть", "Не закрывать направление", "Снизить цену")
        )


def test_poles_formulations_do_not_overlap() -> None:
    """Формулировки полюсов не пересекаются.

    Если одна формулировка попадёт в оба списка, вариант закроет сразу два
    полюса и набор с одним вариантом пройдёт проверку — тавтология вернётся
    с другой стороны.
    """
    from pf.schemas import (
        CLOSE_DIRECTION_TITLES,
        KEEP_CURRENT_TITLES,
        POLE_CLOSE_DIRECTION,
        POLE_KEEP_CURRENT,
        pole_of,
    )

    for title in KEEP_CURRENT_TITLES:
        assert pole_of(title) == POLE_KEEP_CURRENT
    for title in CLOSE_DIRECTION_TITLES:
        assert pole_of(title) == POLE_CLOSE_DIRECTION
    assert not set(KEEP_CURRENT_TITLES) & set(CLOSE_DIRECTION_TITLES)


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
    валидации вместо принятого ответа. Полюса при этом обязательны и в
    голом списке: форма ответа не отменяет требования к набору.
    """
    variants = Variants.model_validate(
        [
            {"title": "Оставить как есть", "summary": "с", "objections": ["о"]},
            {"title": "Закрыть направление", "summary": "с", "objections": ["о"]},
            {"title": "Снизить цену", "summary": "с", "objections": ["о"]},
        ]
    )
    assert len(variants.variants) == 3
    assert all(item.objections for item in variants.variants)
    assert variants.keep_current_marker == "Оставить как есть"


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
    """Метрика без источника принимается, но помечается как непроверяемая.

    Формула при этом обязательна: без неё метрика не проходит схему
    вообще. Здесь проверяется другое — отсутствие источника делает
    расчёт непроверяемым, хотя формально он валиден.
    """
    from pf.schemas import Analysis

    analysis = Analysis(
        metrics=[
            {
                "name": "Что-то",
                "value": 1.0,
                "unit": "шт",
                "formula": "значение из строки 1",
            }
        ],
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
