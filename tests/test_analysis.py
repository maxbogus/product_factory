"""Проверки расчётов: числа вместо соображений.

Требование брифа к плану проверки: «ожидаемое изменение, с чем
сравниваем, срок наблюдения». Обещание «повышение на 5–10 %» без
baseline — не план, а мечта. Значит, агент должен **считать** по данным,
а не опираться на оценку модели.

Каждая метрика несёт provenance: строку источника, формулу и входные
значения. Метрика без источника — не расчёт, а пересказ: её нельзя ни
проверить, ни пересчитать при обновлении данных.

Здесь проверяется именно воспроизводимость: подмена строки исходных
данных должна ломать пересчёт.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.analysis import (
    Metric,
    cagr,
    change_percent,
    compute_from_rows,
    mean,
    median,
    percentile,
)


@pytest.fixture
def _rows() -> list[dict[str, str]]:
    """Строки таблицы с числовым рядом."""
    return [
        {"период": "2025 Q1", "значение": "112,4"},
        {"период": "2025 Q2", "значение": "111,8"},
        {"период": "2025 Q3", "значение": "111,2"},
        {"период": "2025 Q4", "значение": "110,9"},
        {"период": "2026 Q1", "значение": "110,7"},
    ]


def test_median_of_numeric_values(_rows: list[dict[str, str]]) -> None:
    """Медиана считается по числам, а не по строкам."""
    values = [float(row["значение"].replace(",", ".")) for row in _rows]
    assert median(values) == pytest.approx(111.2)


def test_mean_of_numeric_values(_rows: list[dict[str, str]]) -> None:
    """Среднее считается по числам."""
    values = [float(row["значение"].replace(",", ".")) for row in _rows]
    assert mean(values) == pytest.approx(111.4)


def test_percentile_is_interpolated() -> None:
    """Перцентиль интерполируется, а не берётся из соседнего элемента."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == pytest.approx(2.5)


def test_change_percent_computes_relative_change() -> None:
    """Изменение в процентах считается от базы."""
    assert change_percent(100.0, 110.0) == pytest.approx(10.0)
    assert change_percent(100.0, 90.0) == pytest.approx(-10.0)


def test_change_percent_handles_zero_base() -> None:
    """Нулевая база не даёт деления на ноль."""
    assert change_percent(0.0, 5.0) == 0.0


def test_cagr_computes_annual_rate() -> None:
    """CAGR считается по числу периодов."""
    # 100 → 121 за 2 года = 10 % в год
    assert cagr(100.0, 121.0, years=2) == pytest.approx(10.0, rel=1e-3)


def test_cagr_rejects_non_positive_values() -> None:
    """CAGR не считается для неположительных значений."""
    assert cagr(0.0, 121.0, years=2) == 0.0
    assert cagr(100.0, -1.0, years=2) == 0.0


def test_metric_carries_provenance() -> None:
    """Метрика несёт источник, формулу и входные значения."""
    metric = Metric(
        name="Абонентская база, 2026 Q1",
        value=110.7,
        unit="млн",
        source_id="project/data/numbers.csv",
        row=4,
        formula="значение из строки 4",
        inputs={"значение": "110,7"},
    )
    assert metric.citation == "project/data/numbers.csv#row4"
    assert metric.is_reproducible


def test_metric_without_source_is_not_reproducible() -> None:
    """Метрика без источника не считается воспроизводимой."""
    metric = Metric(name="Что-то", value=1.0, unit="шт")
    assert not metric.is_reproducible
    assert metric.citation == ""


def test_compute_from_rows_returns_reproducible_metrics(
    _rows: list[dict[str, str]],
) -> None:
    """Расчёт по строкам таблицы даёт метрику с provenance."""
    metrics = compute_from_rows(
        _rows,
        value_column="значение",
        source_id="project/data/numbers.csv",
        name="Абонентская база",
        unit="млн",
    )
    assert len(metrics) == 5
    # Метрика соответствует своей строке: значение и номер строки.
    latest = metrics[-1]
    assert latest.value == pytest.approx(110.7)
    assert latest.row == 4
    assert latest.source_id == "project/data/numbers.csv"
    assert latest.citation == "project/data/numbers.csv#row4"
    assert latest.is_reproducible
    assert latest.inputs["значение"] == "110,7"


def test_compute_from_rows_skips_unparsable_values() -> None:
    """Нечисловые значения пропускаются, а не ломают расчёт."""
    rows = [
        {"период": "2026 Q1", "значение": "н/д"},
        {"период": "2026 Q2", "значение": "12,5"},
    ]
    metrics = compute_from_rows(
        rows,
        value_column="значение",
        source_id="project/data/x.csv",
        name="Метрика",
        unit="млн",
    )
    assert len(metrics) == 1
    assert metrics[0].value == pytest.approx(12.5)


def test_metric_recomputes_from_inputs() -> None:
    """Метрику можно пересчитать по её же входным значениям."""
    metric = Metric(
        name="Изменение",
        value=change_percent(100.0, 110.0),
        unit="%",
        source_id="project/data/x.csv",
        row=1,
        formula="(110 - 100) / 100 * 100",
        inputs={"база": "100", "новое": "110"},
    )
    recomputed = change_percent(
        float(metric.inputs["база"]),
        float(metric.inputs["новое"]),
    )
    assert recomputed == pytest.approx(metric.value)


def test_table_rows_are_readable_from_material(tmp_path: Path) -> None:
    """Из материала-таблицы читаются строки для расчёта."""
    import csv

    from pf.materials import load_materials

    path = tmp_path / "numbers.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["период", "значение"])
        writer.writerow(["2026 Q1", "110,7"])

    materials = load_materials(tmp_path)
    assert materials
    rows = materials[0].table_rows()
    assert rows == [{"период": "2026 Q1", "значение": "110,7"}]


def test_metric_is_confirmed_when_value_matches_row(tmp_path: Path) -> None:
    """Метрика подтверждается, если значение совпадает со строкой."""
    import csv

    from pf.materials import load_materials
    from pf.sources import verify_metric

    path = tmp_path / "numbers.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["период", "значение"])
        writer.writerow(["2026 Q1", "110,7"])

    materials = load_materials(tmp_path)
    metric = Metric(
        name="Абонентская база",
        value=110.7,
        unit="млн",
        source_id=materials[0].source_id,
        row=0,
        formula="значение из строки 1",
        inputs={"значение": "110,7"},
    )
    support = verify_metric(metric, materials)
    assert support.is_usable, f"метрика должна подтверждаться: {support.detail}"


def test_metric_is_rejected_when_value_differs(tmp_path: Path) -> None:
    """Метрика с неверным значением отклоняется."""
    import csv

    from pf.materials import load_materials
    from pf.sources import verify_metric

    path = tmp_path / "numbers.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["период", "значение"])
        writer.writerow(["2026 Q1", "110,7"])

    materials = load_materials(tmp_path)
    metric = Metric(
        name="Абонентская база",
        value=47.0,
        unit="млн",
        source_id=materials[0].source_id,
        row=0,
        formula="значение из строки 1",
        inputs={"значение": "110,7"},
    )
    support = verify_metric(metric, materials)
    assert not support.is_usable, "неверное значение не должно подтверждаться"


def test_grounding_metrics_are_computed() -> None:
    """Метрики качества оснований считаются по прогону.

    Аналог citation_precision в mlaw-rag: доля оснований, которые
    действительно подтверждены. Без такой метрики «качество оснований»
    — это самооценка.
    """
    from pf.ground import GroundEntry, GroundReport

    report = GroundReport(
        entries=[
            GroundEntry(citation="a", status="confirmed", kind="факт"),
            GroundEntry(citation="b", status="no_data", kind="факт"),
            GroundEntry(citation="c", status="confirmed", kind="метрика"),
            GroundEntry(citation="d", status="needs_review", kind="метрика"),
        ]
    )
    assert report.confirmed == 2
    assert report.total == 4
    assert report.grounding_rate == pytest.approx(0.5)
    assert report.fact_grounding_rate == pytest.approx(0.5)
    assert report.metric_grounding_rate == pytest.approx(0.5)


def test_grounding_metrics_handle_empty_report() -> None:
    """Пустой прогон даёт нули, а не деление на ноль."""
    from pf.ground import GroundReport

    report = GroundReport(entries=[])
    assert report.total == 0
    assert report.grounding_rate == 0.0


def test_control_test_detects_broken_grounding() -> None:
    """Контрольный тест: при случайных основаниях метрика должна пасть.

    Если метрика не падает на мусоре, она ничего не измеряет. Это
    прямой аналог контрольного теста в mlaw-rag.
    """
    from pf.ground import GroundEntry, GroundReport

    # Настоящий прогон: основания подтверждены.
    real = GroundReport(
        entries=[
            GroundEntry(citation="a", status="confirmed", kind="факт"),
            GroundEntry(citation="b", status="confirmed", kind="факт"),
        ]
    )
    # Контроль: те же объёмы, но основания не подтверждаются.
    control = GroundReport(
        entries=[
            GroundEntry(citation="x", status="no_data", kind="факт"),
            GroundEntry(citation="y", status="no_data", kind="факт"),
        ]
    )
    assert real.grounding_rate > control.grounding_rate
    assert control.grounding_rate < 0.1


def test_control_grounding_collapses_on_fabricated_citations() -> None:
    """Контрольный тест: выдуманные ссылки должны обрушить метрику.

    Принцип из mlaw-rag: если метрика не падает на мусоре, она ничего
    не измеряет. Здесь проверяется, что подтверждение основания —
    реальная проверка, а не счётчик.
    """
    from pf.ground import verify_quote

    source = "По состоянию на июнь 2024 года компания обслужила 77,26 млн абонентов."

    # Настоящая цитата подтверждается.
    real = verify_quote(source, "компания обслужила 77,26 млн абонентов", "test/s")
    assert real.status.value == "confirmed"

    # Выдуманная цитата — нет.
    fabricated = verify_quote(source, "компания обслужила 99,99 млн абонентов", "test/s")
    assert fabricated.status.value == "no_data"


def test_control_grounding_rate_collapses() -> None:
    """Доля подтверждённых падает с 100 % до нуля на выдумках."""
    from pf.ground import GroundEntry, GroundReport

    real = GroundReport(
        entries=[
            GroundEntry(citation="a", status="confirmed", kind="факт"),
            GroundEntry(citation="b", status="confirmed", kind="факт"),
        ]
    )
    fabricated = GroundReport(
        entries=[
            GroundEntry(citation="x", status="no_data", kind="факт"),
            GroundEntry(citation="y", status="no_data", kind="факт"),
        ]
    )
    assert real.grounding_rate == 1.0
    assert fabricated.grounding_rate == 0.0
    assert fabricated.grounding_rate < 0.1


def test_series_are_grouped_before_change_is_computed() -> None:
    """Изменение считается по одному ряду, а не по всем строкам.

    Таблица может содержать разные операторов и разные годы. Считать
    изменение по всем строкам сразу бессмысленно: из 87 млн абонентов
    МТС и 5,7 млн абонентов МТС Беларусь получается «-16,67 %», что
    не означает ничего. Ряды надо разделять по подписи и считать
    изменение внутри каждого.
    """
    from pf.analysis import group_series, summarize_groups

    rows = [
        {"оператор": "МТС", "значение": "87"},
        {"оператор": "МТС", "значение": "82"},
        {"оператор": "МТС Беларусь", "значение": "5.7"},
    ]
    groups = group_series(rows, label_column="оператор")
    assert len(groups) == 2

    summaries = summarize_groups(
        groups,
        value_column="значение",
        source_id="project/data/x.csv",
        name="Абонентская база",
    )
    changes = [m for m in summaries if "изменение" in m.name]
    assert len(changes) == 2
    # Ряд МТС: 87 -> 82 = -5,75 %. Ряд МТС Беларусь: одна точка, ноль.
    by_name = {m.name: m.value for m in changes}
    mts = next(v for k, v in by_name.items() if "МТС:" in k or "(МТС)" in k)
    belarus = next(v for k, v in by_name.items() if "Беларусь" in k)
    assert mts == pytest.approx(-5.747, abs=0.01)
    assert belarus == 0.0


def test_group_summary_row_points_to_table_row() -> None:
    """Сводка по ряду ссылается на свою строку исходной таблицы.

    Первый ряд занимает строки 0 и 1, второй — строку 2. Раньше сводка
    второго ряда ссылалась на строку 0: индекс считался внутри ряда, а
    не по таблице. Из-за этого «38,9 млн» указывало на строку, где
    лежит 87, и проверка справедливо отвергала верный расчёт.
    """
    from pf.analysis import group_series, summarize_groups

    rows = [
        {"оператор": "МТС", "значение": "87"},
        {"оператор": "МТС", "значение": "82"},
        {"оператор": "Tele2", "значение": "38.9"},
    ]
    groups = group_series(rows, label_column="оператор")
    summaries = summarize_groups(
        groups,
        value_column="значение",
        source_id="project/data/x.csv",
        name="Абонентская база",
    )

    tele2 = [m for m in summaries if "Tele2" in m.name]
    assert tele2, "сводка для Tele2 не найдена"
    for metric in tele2:
        assert metric.row == 2, f"сводка Tele2 должна ссылаться на строку 2, а не {metric.row}"


def test_group_summary_rows_follow_period_order() -> None:
    """Номера строк в сводке следуют за порядком наблюдений.

    Ряд упорядочивается по периоду, и наблюдения меняются местами
    относительно таблицы: строка 1 — 2023 год, строка 0 — 2024. Значит
    «изменение за период» ссылается на последнее наблюдение (строка 0),
    а «среднее» — на первое (строка 1). Индексы остаются индексами
    таблицы, а не позициями внутри отсортированного ряда.
    """
    from pf.analysis import group_series, summarize_groups

    rows = [
        {"оператор": "МТС", "период": "2024", "значение": "87"},
        {"оператор": "МТС", "период": "2023", "значение": "80"},
    ]
    groups = group_series(rows, label_column="оператор")
    summaries = summarize_groups(
        groups,
        value_column="значение",
        source_id="project/data/x.csv",
        name="Абонентская база",
    )

    change = next(m for m in summaries if "изменение" in m.name)
    average = next(m for m in summaries if "среднее" in m.name)
    assert change.row == 0, (
        f"изменение должно ссылаться на последнее наблюдение (строка 0), а не {change.row}"
    )
    assert average.row == 1, (
        f"среднее должно ссылаться на первое наблюдение (строка 1), а не {average.row}"
    )


def test_change_within_one_series_is_computed() -> None:
    """Внутри ряда изменение считается по первому и последнему."""
    from pf.analysis import group_series, summarize_groups

    rows = [
        {"оператор": "МТС", "значение": "100"},
        {"оператор": "МТС", "значение": "110"},
    ]
    groups = group_series(rows, label_column="оператор")
    summaries = summarize_groups(
        groups,
        value_column="значение",
        source_id="project/data/x.csv",
        name="Абонентская база",
    )
    change = next(m for m in summaries if "изменение" in m.name)
    assert change.value == pytest.approx(10.0)


def test_aggregate_carries_observations_not_only_count() -> None:
    """Сводная метрика несёт сами наблюдения, а не их количество.

    Среднее по ряду — это 8,75 из четырёх наблюдений 6, 7, 10, 12.
    Если в inputs лежит только «наблюдений: 4», пересчитать среднее
    невозможно: чисел для сложения нет. Значит метрику нечем
    подтвердить, и она отбрасывается вопреки тому, что расчёт верен.

    Поэтому в inputs кладутся **наблюдения ряда**: по ним пересчёт
    воспроизводится, и агент может показать, из чего сложил.
    """
    from pf.analysis import group_series, summarize_groups

    rows = [
        {"оператор": "Виртуальные операторы", "период": "2017", "значение": "6"},
        {"оператор": "Виртуальные операторы", "период": "2018", "значение": "7"},
        {"оператор": "Виртуальные операторы", "период": "2019", "значение": "10"},
        {"оператор": "Виртуальные операторы", "период": "2020", "значение": "12"},
    ]
    groups = group_series(rows, label_column="оператор")
    summaries = summarize_groups(
        groups,
        value_column="значение",
        source_id="project/data/x.csv",
        name="Численность",
    )

    average = next(m for m in summaries if "среднее" in m.name)
    numbers = []
    for value in average.inputs.values():
        if isinstance(value, (list, tuple)):
            numbers = [float(v) for v in value]
            break
    assert numbers == [6.0, 7.0, 10.0, 12.0], (
        f"среднее должно нести наблюдения ряда, а не их количество: {average.inputs}"
    )


def test_period_column_is_not_used_as_series_label() -> None:
    """Колонка-период не выбирается подписью ряда.

    Подпись ряда — это «что за величина» (оператор, показатель), а не
    «когда». Если подписью стала колонка с датой, то каждая дата
    образовала свой ряд из одной строки: изменение всегда 0 %, а
    среднее всегда равно единственному значению. Ряд перестаёт быть
    рядом.

    Поэтому колонка с периодом исключается из выбора подписи.
    """
    from pf.analysis import label_column_for

    columns = ["дата", "значение_процентов"]
    assert label_column_for(columns, value_column="значение_процентов") == ""

    columns = ["оператор", "показатель", "значение", "период", "источник"]
    assert label_column_for(columns, value_column="значение") == "оператор"


def test_aggregate_metric_is_marked_as_aggregate() -> None:
    """Агрегат отличается от значения в строке.

    Среднее, медиана, размах и изменение вычисляются по нескольким
    наблюдениям, и их значения нет ни в одной строке таблицы. Такие
    метрики нельзя сверить со строкой — их проверяют пересчётом по
    входам, поэтому входы должны быть видны и в блоке расчётов.
    """
    from pf.analysis import Metric

    aggregate = Metric(
        name="Абонентская база: среднее",
        value=8.75,
        unit="млн",
        source_id="project/data/x.csv",
        row=0,
        formula="сумма / количество",
        inputs={"наблюдения": ["6", "7", "10", "12"]},
    )
    plain = Metric(
        name="Абонентская база",
        value=82.0,
        unit="млн",
        source_id="project/data/x.csv",
        row=1,
        formula="значение из колонки «значение», строка 2",
        inputs={"значение": "82"},
    )
    assert aggregate.is_aggregate
    assert not plain.is_aggregate


# ── Порядок ряда по периоду ────────────────────────────────────────────────
#
# Период — это время, а не число из цифр. «1 кв. 2017» содержит цифры 1 и
# 2017; если склеить их в 12017, первый квартал 2018 года (12018) встанет
# раньше четвёртого квартала 2017-го (42017). Ряд перестаёт быть рядом:
# «изменение за период» считается между двумя случайными строками, и число
# выглядит правдоподобно. Поэтому квартал разбирается как часть года.


def _quarter_rows() -> list[dict[str, str]]:
    """Три квартала одного показателя в порядке таблицы.

    Returns:
        Строки с периодами в русской записи — как их печатает источник.
    """
    return [
        {"оператор": "ВымпелКом", "период": "1 кв. 2017", "значение": "100"},
        {"оператор": "ВымпелКом", "период": "4 кв. 2017", "значение": "110"},
        {"оператор": "ВымпелКом", "период": "1 кв. 2018", "значение": "121"},
    ]


def test_period_order_quarters_sort_in_time() -> None:
    """Кварталы упорядочиваются по времени, а не по склеенным цифрам.

    «1 кв. 2017» < «4 кв. 2017» < «1 кв. 2018» — если разбор читает
    квартал как часть года. Разбор по цифрам даёт 12017 < 12018 < 42017 и
    ставит первый квартал 2018 года между первым и четвёртым кварталами
    2017-го.
    """
    from pf.analysis import _period_key

    rows = _quarter_rows()
    ordered = [row["период"] for row in sorted(rows, key=_period_key)]
    assert ordered == ["1 кв. 2017", "4 кв. 2017", "1 кв. 2018"]


def test_period_order_change_uses_last_quarter() -> None:
    """Изменение по квартальному ряду считается от последнего квартала.

    Ряд 100 → 110 → 121: изменение от первого квартала 2017 к первому
    кварталу 2018 — 21 %. Порядок по цифрам берёт последним «4 кв. 2017» и
    выдаёт 10 % — правдоподобное, но неверное число.
    """
    from pf.analysis import group_series, summarize_groups

    groups = group_series(_quarter_rows(), label_column="оператор")
    summaries = summarize_groups(
        groups,
        value_column="значение",
        source_id="project/data/x.csv",
        name="ARPU",
    )

    change = next(metric for metric in summaries if "изменение" in metric.name)
    assert change.value == pytest.approx(21.0), change.inputs


def test_period_order_quarter_forms_give_one_key() -> None:
    """Одна и та же четверть в разных записях даёт один ключ периода.

    Источники пишут квартал по-разному: «2017-Q1», «Q1 2017», «1 кв. 2017»,
    «1-й квартал 2017», «I квартал 2017», «первый квартал 2017». Если формы
    дают разные ключи, один ряд распадается на несколько, и «изменение за
    период» считается внутри четверти.
    """
    from pf.analysis import _period_key

    keys = {
        _period_key({"период": form})[:2]
        for form in (
            "2017-Q1",
            "Q1 2017",
            "2017 Q1",
            "1 кв. 2017",
            "1-й квартал 2017",
            "I квартал 2017",
            "1 квартал 2017 г.",
            "первый квартал 2017",
        )
    }
    assert len(keys) == 1, keys
    assert _period_key({"период": "3-й квартал 2024"})[:2] == _period_key({"период": "2024-Q3"})[:2]


def test_period_order_year_goes_before_its_quarters() -> None:
    """Год и его кварталы сравнимы: год раньше кварталов, четвёртый — раньше следующего года.

    Разная крупность периода не повод терять порядок: ряд «2017», «2017-Q1»,
    «2017-Q4», «2018» — это движение во времени. Квартал живёт в шкале
    года (год + квартал/5), поэтому ничьи между годом и его первым
    кварталом не возникает.
    """
    from pf.analysis import _period_key

    year = _period_key({"период": "2017"})[:2]
    first = _period_key({"период": "2017-Q1"})[:2]
    fourth = _period_key({"период": "2017-Q4"})[:2]
    next_year = _period_key({"период": "2018"})[:2]

    assert year < first < fourth < next_year
