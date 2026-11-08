"""Проверки критериев успеха плана проверки.

План проверки отвечает на вопрос «как поймём, что сработало». В
демо-прогоне план был про удержание, а критерий успеха — «снижение
текучести кадров»: показатель другой, проверять его будет нечем.
Схема такое пропускала, потому что список критериев свободный.

Второй дефект — числа. Проверка сравнивала только основы слов, поэтому
критерий «увеличение удержания на 5 %» проходил по слову «удержание»,
а само число не сверялось ни с чем: в `demo-v9` из плана стояло
«удержание на уровне 87 %», хотя 87 — абонентская база МТС. Число в
плане и в обязательстве — такое же утверждение, как факт, и требует
расчёта.
"""

from __future__ import annotations

from pf.analysis import Metric
from pf.schemas import (
    Commitment,
    CommunicationStrategy,
    DecisionRecord,
    ProductStrategy,
    Strategy,
    VerificationPlan,
)
from pf.workflow import Workflow


def _strategy(
    criteria: list[str],
    expected: str,
    compare_against: str = "уровень прошлого квартала",
) -> Strategy:
    """Стратегия с заданным планом проверки.

    Args:
        criteria: Критерии успеха.
        expected: Ожидаемое изменение.
        compare_against: С чем сравниваем.

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
            compare_against=compare_against,
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


# ── Числа плана и обязательств ───────────────────────────────────────────────


def _metric(
    name: str,
    value: float,
    unit: str = "",
    source_id: str = "",
    row: int | None = None,
) -> Metric:
    """Расчёт с показателем в названии.

    Args:
        name: Название расчёта.
        value: Значение.
        unit: Единица измерения.
        source_id: Материал, по которому считали.
        row: Номер строки материала.

    Returns:
        Готовая метрика.
    """
    return Metric(
        name=name,
        value=value,
        unit=unit,
        formula="значение из строки таблицы",
        source_id=source_id,
        row=row,
    )


def _base_metrics() -> list[Metric]:
    """Расчёты демо-прогона: абонентская база есть, удержания — нет.

    Ссылка настоящая: показатель расчёта берётся из строки, на которую он
    ссылается, — иначе имя метрики (заголовок материала) выдавало бы
    расчёт про абонентскую базу за расчёт про удержание.
    """
    return [
        _metric(
            "Открытые метрики абонентской базы операторов связи",
            87.0,
            "млн",
            "project/data/telecom_open_metrics.csv",
            0,
        ),
        _metric(
            "Открытые метрики абонентской базы операторов связи",
            82.0,
            "млн",
            "project/data/telecom_open_metrics.csv",
            1,
        ),
    ]


def test_criterion_numbers_must_come_from_calculations() -> None:
    """Число критерия без расчёта — замечание, даже если показатель тот же.

    Слово «удержание» есть и в плане, и в критерии, и прежняя проверка
    этим удовлетворялась. Откуда взялись 5 % — из ответа не следует:
    удержание в материалах не измерено ни разу.
    """
    strategy = _strategy(["Увеличение удержания на 5%"], "Повышение удержания клиентов")

    notes = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert notes, "число без расчёта должно давать замечание"
    assert "5" in notes[0]
    assert "расчёта" in notes[0]


def test_baseline_numbers_of_other_indicator_are_rejected() -> None:
    """Случай `demo-v9`: 87 % — это абонентская база, а не удержание.

    Число в источнике есть, и по нему даже есть расчёт. Но расчёт — про
    другой показатель: подтверждать им «удержание на уровне 87 %» нельзя.
    """
    strategy = _strategy(
        ["Увеличение удержания клиентов"],
        "Повышение удержания клиентов",
        compare_against="Удержание на уровне 87%",
    )

    notes = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert notes
    assert "87" in notes[0]
    assert "другой показатель" in notes[0]


def test_criterion_numbers_matching_calculation_pass() -> None:
    """Число, подтверждённое расчётом того же показателя, замечаний не даёт."""
    strategy = _strategy(["Удержание клиентов не ниже 87 %"], "Повышение удержания клиентов")
    metrics = [_metric("Удержание клиентов", 87.0, "%")]

    assert Workflow._check_plan_numbers(strategy, metrics=metrics) == []


def test_commitment_numbers_must_come_from_calculations() -> None:
    """Обязательство с числом проверяется так же, как план.

    Фаза решения не проверялась вовсе: число в обязательстве попадало в
    зафиксированное решение без единого расчёта.
    """
    record = DecisionRecord(
        decision="Оставляем тариф, возвращаем удержание",
        commitments=[
            Commitment(
                statement="Повысить удержание на 5 %",
                owner="маркетинг",
                metric="удержание",
                review_condition="если через 3 месяца не выросло — откатить",
            )
        ],
    )

    notes = Workflow._check_plan_numbers(record, metrics=_base_metrics())

    assert notes
    assert "5" in notes[0]


def test_plan_without_numbers_has_no_notes() -> None:
    """Срок и формулировки без чисел не считаются неподтверждённой величиной.

    Иначе проверка краснела бы на «3 месяца» и обесценилась бы: замечание,
    которое срабатывает всегда, не читают.
    """
    strategy = _strategy(["Увеличение удержания клиентов"], "Повышение удержания клиентов")

    assert Workflow._check_plan_numbers(strategy, metrics=_base_metrics()) == []
