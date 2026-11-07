"""Проверки смысла расчётной метрики.

Подтверждение метрики пересчётом ловит выдуманное число, но не подмену
показателя. В демо-прогоне метрика называлась «Удержание клиентов», а
значение 87 млн было абонентской базой из строки таблицы: цитата
дословная, число есть в строке — проверка проходила, хотя под словом
«удержание» стояла база.

Второй случай того же рода — формула-тождество: «Сумма / количество» по
одному наблюдению (87 / 1 = 87). Формально формула есть, фактически число
просто переписано из строки.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.analysis import Metric
from pf.materials import load_materials
from pf.sources import verify_metric


@pytest.fixture
def materials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Материалы с одной строкой таблицы.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.

    Returns:
        Загруженные материалы.
    """
    monkeypatch.setattr("pf.config.settings.project_dir", tmp_path)
    (tmp_path / "base.csv").write_text(
        "оператор,показатель,значение\nМТС,абонентская база,87\n",
        encoding="utf-8",
    )
    return load_materials(tmp_path)


def test_metric_name_must_match_column(materials: list[object]) -> None:
    """Показатель в имени метрики обязан быть в строке источника.

    Args:
        materials: Материалы теста.
    """
    metric = Metric(
        name="Удержание клиентов (МТС, абонентская база)",
        value=87.0,
        unit="млн",
        source_id="project/data/base.csv#row0",
        formula="значение из колонки «значение»",
        inputs={"значение": "87"},
    )

    support = verify_metric(metric, materials)

    assert not support.is_usable
    assert "не совпало с колонкой" in support.detail


def test_metric_name_matching_is_tolerant(materials: list[object]) -> None:
    """Название показателя словами человека подтверждается.

    Args:
        materials: Материалы теста.
    """
    metric = Metric(
        name="Абонентская база МТС",
        value=87.0,
        unit="млн",
        source_id="project/data/base.csv#row0",
        formula="значение из колонки «значение»",
        inputs={"значение": "87"},
    )

    support = verify_metric(metric, materials)

    assert support.is_usable, f"метрика должна подтверждаться: {support.detail}"


def test_identity_formula_is_not_a_calculation(materials: list[object]) -> None:
    """«Сумма / количество» по одному наблюдению — не расчёт.

    Args:
        materials: Материалы теста.
    """
    metric = Metric(
        name="Абонентская база МТС",
        value=87.0,
        unit="млн",
        source_id="project/data/base.csv#row0",
        formula="Сумма / количество",
        inputs={"сумма": "87", "количество": "1"},
    )

    support = verify_metric(metric, materials)

    assert not support.is_usable
    assert "тождество" in support.detail


def test_real_aggregate_still_confirms(materials: list[object]) -> None:
    """Настоящий агрегат по нескольким наблюдениям подтверждается.

    Args:
        materials: Материалы теста.
    """
    metric = Metric(
        name="Абонентская база МТС",
        value=43.5,
        unit="млн",
        source_id="project/data/base.csv#row0",
        formula="Среднее значение",
        inputs={"наблюдения": ["87", "0"]},
    )

    support = verify_metric(metric, materials)

    assert support.is_usable, f"агрегат должен подтверждаться: {support.detail}"
