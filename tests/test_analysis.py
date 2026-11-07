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
