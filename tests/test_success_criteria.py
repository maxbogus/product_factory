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
from pf.ground import GroundTally
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

    check = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert check.notes, "число без расчёта должно давать замечание"
    assert "5" in check.notes[0].text
    assert "расчёта" in check.notes[0].text
    # Цель плана — не утверждение о мире: замечание есть, статус она не понижает.
    assert check.unsupported == 0


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

    check = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert check.notes
    assert "87" in check.notes[0].text
    assert "другой показатель" in check.notes[0].text
    # База сравнения — утверждение о текущем уровне, а не цель: 87 % ни из
    # чего не следует, значит статус вывода обязан это знать.
    assert check.unsupported == 1


def test_criterion_numbers_matching_calculation_pass() -> None:
    """Число, подтверждённое расчётом того же показателя, замечаний не даёт."""
    strategy = _strategy(["Удержание клиентов не ниже 87 %"], "Повышение удержания клиентов")
    metrics = [_metric("Удержание клиентов", 87.0, "%")]

    check = Workflow._check_plan_numbers(strategy, metrics=metrics)

    assert check.notes == ()
    assert check.unsupported == 0


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

    check = Workflow._check_plan_numbers(record, metrics=_base_metrics())

    assert check.notes
    assert "5" in check.notes[0].text


def test_plan_without_numbers_has_no_notes() -> None:
    """Срок и формулировки без чисел не считаются неподтверждённой величиной.

    Иначе проверка краснела бы на «3 месяца» и обесценилась бы: замечание,
    которое срабатывает всегда, не читают.
    """
    strategy = _strategy(["Увеличение удержания клиентов"], "Повышение удержания клиентов")

    check = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert check.notes == ()
    assert check.unsupported == 0


# ── Тот ли показатель сравнивается ───────────────────────────────────────────
#
# В прогоне `cloud-deepseek-2` база сравнения называла рыночный фон и несла
# свою ссылку на строку расчёта:
#
#   7,39 млн за 8 месяцев, изменение по ряду 84,62 %
#   [project/data/mnp_stats.csv#row2] и [project/data/mnp_stats.csv#row1]
#
# Оба числа подтверждены расчётами MNP. Но в пуле был ещё расчёт по ключевой
# ставке, и его имя («…: изменение за период») делило с текстом слово
# «период». Проверка считала этот расчёт «тем же показателем», не находила в
# нём чисел и сообщала «не совпадает с расчётом по показателю «Ключевая
# ставка Банка России…»». То есть замечание называло **не тот** показатель, а
# до настоящего совпадения дело не доходило: ветка `others` стояла после
# `continue`.

#: База сравнения из прогона `cloud-deepseek-2` — дословно, вместе с якорями.
MARKET_BASE = (
    "Базовый период того же тарифа до включения предупреждений плюс, если удастся "
    "выделить, часть потока, не получавшая уведомление. Рыночный фон (заявки MNP: "
    "7,39 млн за 8 месяцев, изменение по ряду 84,62 % "
    "[project/data/mnp_stats.csv#row2] и [project/data/mnp_stats.csv#row1]) — не база "
    "для удержания, а объяснение, почему общерыночный сдвиг нельзя записывать в эффект."
)


def _market_metrics() -> list[Metric]:
    """Расчёты рыночного фона: те же строки, что названы в тексте базы.

    Returns:
        Два расчёта MNP — значение строки и изменение по ряду.
    """
    return [
        _metric(
            "Заявки на перенос номера за 8 месяцев, млн",
            7.39,
            "млн",
            "project/data/mnp_stats.csv",
            2,
        ),
        _metric(
            "Заявки на перенос номера за год, млн",
            84.62,
            "%",
            "project/data/mnp_stats.csv",
            1,
        ),
    ]


def _rate_metric() -> Metric:
    """Расчёт с служебным словом «период» в названии.

    Returns:
        Расчёт по ключевой ставке — тот самый ложный «тот же показатель».
    """
    return _metric(
        "Ключевая ставка Банка России, январь-март 2026: изменение за период",
        78.79,
        "%",
    )


def test_numbers_are_not_compared_with_foreign_indicator() -> None:
    """Число со своей ссылкой на строку расчёта не отклоняется чужой метрикой.

    Ожидание — ни одного замечания: оба числа подтверждены расчётами MNP, а
    расчёт по ключевой ставке не «тот же показатель», хотя в его имени есть
    слово «период».
    """
    strategy = _strategy(
        ["Замер удержания сделан по зафиксированному определению"],
        "Сначала измерить текущий уровень удержания",
        compare_against=MARKET_BASE,
    )

    check = Workflow._check_plan_numbers(strategy, metrics=[*_market_metrics(), _rate_metric()])

    assert check.notes == (), f"верные числа отклонены: {[note.text for note in check.notes]}"


def test_service_word_does_not_make_calculation_same_indicator() -> None:
    """Слова «изменение за период» не делают расчёт по ставке показателем плана."""
    strategy = _strategy(
        ["Изменение удержания за период: 12 %"],
        "Измерение удержания клиентов",
    )

    check = Workflow._check_plan_numbers(strategy, metrics=[_rate_metric()])

    assert check.notes, "число без расчёта должно давать замечание"
    assert "ничем не подтверждено" in check.notes[0].text
    assert "Ключевая ставка" not in check.notes[0].text


def test_number_missing_in_same_indicator_is_mismatch() -> None:
    """Показатель тот же, а числа в расчёте нет — это расхождение, не подмена."""
    strategy = _strategy(
        ["Удержание клиентов не ниже 95 %"],
        "Повышение удержания клиентов",
    )
    metrics = [_metric("Удержание клиентов", 87.0, "%")]

    check = Workflow._check_plan_numbers(strategy, metrics=metrics)

    assert check.notes
    assert "не совпадает" in check.notes[0].text
    assert "95" in check.notes[0].text


# ── Связка с статусом вывода ─────────────────────────────────────────────────
#
# Замечание проверки само по себе статус не меняло: в `demo-v9` «удержание на
# уровне 87 %» стояло в `rejected_details`, а рядом — «подкреплено». Замер на
# пяти журналах (T-48b) показал, что понижать статус должна не любая величина:
# цель плана — выбранное число, а не утверждение о мире. Иначе «вырастет на
# 5 %» понижало бы каждый вывод, где есть план.


def test_unsupported_number_demotes_status() -> None:
    """Число базы сравнения без своего расчёта понижает статус вывода.

    База сравнения — утверждение о текущем уровне («удержание на уровне
    87 %»), и 87 в материалах — абонентская база. Вывод не может быть
    «подкреплено», пока это число ни из чего не следует.
    """
    strategy = _strategy(
        ["Увеличение удержания клиентов"],
        "Повышение удержания клиентов",
        compare_against="Удержание на уровне 87%",
    )

    check = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert check.unsupported == 1
    tally = GroundTally(checked=2, confirmed=2, rejected=0, needs_review=0)
    verdict = Workflow._conclusion_verdict(tally, {}, unsupported_numbers=check.unsupported)

    assert verdict is not None
    assert verdict["status"] == "подкреплено частично"
    assert verdict["unsupported_numbers"] == 1


def test_plan_goal_numbers_are_not_facts() -> None:
    """Цель «на 5 %» остаётся замечанием, но статус не понижает.

    Требовать расчёт для выбранной цели значило бы понижать каждый вывод, у
    которого есть план: цель не утверждение о мире. В `demo-v12` так были бы
    понижены оба шага, где нет ни одного неподтверждённого факта.
    """
    strategy = _strategy(["Увеличение удержания на 5%"], "Повышение удержания клиентов")

    check = Workflow._check_plan_numbers(strategy, metrics=_base_metrics())

    assert check.notes, "цель с числом — всё ещё замечание для человека"
    assert check.unsupported == 0


def test_number_range_is_not_negative() -> None:
    """Диапазон «5-10 %» не превращается в −10 %.

    Знак минус в диапазоне — дефис, а не отрицательная величина. Пока это
    было замечанием в `rejected_details`, ошибка не влияла на статус; после
    связки выдуманный минус понижал бы вывод.
    """
    hyphen = [value for value, _ in Workflow._magnitudes("Удержание вырастет на 5-10%")]
    dash = [value for value, _ in Workflow._magnitudes("Удержание вырастет на 5–10%")]

    assert hyphen == [10.0]
    assert dash == [10.0]
    assert all(value > 0 for value in [*hyphen, *dash])


# ── Готовый агрегат: расчёт даёт система ─────────────────────────────────────
#
# Агрегат и его первая строка делят один якорь `[… #rowN]`: у среднего
# `source_id` и `row` взяты у первого наблюдения (`analysis.summarize_series`).
# Поэтому значение строки и значение среднего различимы только по виду
# расчёта. Модель, назвавшая средним значение строки (87), складывает сама:
# среднее посчитал код, и оно равно 84,5. Общий якорь не должен подтверждать
# подмену (`T-62`).


def _row_and_mean_metrics() -> list[Metric]:
    """Строка таблицы и её среднее — делят строку № 0 и якорь.

    Returns:
        Расчёт строки (87) и среднее по ряду (84,5).
    """
    row = _metric(
        "Открытые метрики: база, МТС",
        87.0,
        "млн",
        "project/data/telecom_open_metrics.csv",
        0,
    )
    mean = Metric(
        name="Открытые метрики: база, МТС: среднее",
        value=84.5,
        unit="млн",
        source_id="project/data/telecom_open_metrics.csv",
        row=0,
        formula="сумма / количество",
        inputs={"наблюдения": ["87", "82"]},
    )
    return [row, mean]


def test_aggregate_anchor_does_not_confirm_row_value() -> None:
    """Значение строки не подтверждает среднее, даже если якорь общий.

    «Среднее 87 млн» — это арифметика модели: готовый агрегат системы равен
    84,5. Оба расчёта ссылаются на `#row0`, и прежняя сверка принимала 87
    по строке, не замечая, что назван агрегат.
    """
    strategy = _strategy(
        ["Средняя абонентская база"],
        "Повышение удержания клиентов",
        compare_against="Среднее 87 млн [project/data/telecom_open_metrics.csv#row0]",
    )

    check = Workflow._check_plan_numbers(strategy, metrics=_row_and_mean_metrics())

    assert check.notes, "значение строки не должно подтверждать среднее"
    assert "не совпадает" in check.notes[0].text


def test_aggregate_anchor_confirms_ready_aggregate_value() -> None:
    """Ссылка на готовый агрегат (84,5) подтверждается — иначе правило бесполезно."""
    strategy = _strategy(
        ["Средняя абонентская база"],
        "Повышение удержания клиентов",
        compare_against="Среднее 84,5 млн [project/data/telecom_open_metrics.csv#row0]",
    )

    check = Workflow._check_plan_numbers(strategy, metrics=_row_and_mean_metrics())

    assert check.notes == (), f"готовый агрегат отклонён: {[n.text for n in check.notes]}"
